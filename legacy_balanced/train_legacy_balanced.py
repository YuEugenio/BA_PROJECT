"""
train_legacy_balanced.py - Legacy training with imbalance-aware optimizations
"""

import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from dataset import create_dataloaders, summarize_dataset, PES_COLUMNS
from model_legacy import create_model, PES_TASK_NAMES


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for multi-class classification.
    """

    def __init__(self, class_counts, beta: float = 0.999, gamma: float = 1.5, eps: float = 1e-8):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)

        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, eps)
        weights = weights / np.maximum(weights.sum(), eps) * len(weights)

        self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs[torch.arange(targets.size(0), device=targets.device), targets]
        target_probs = probs[torch.arange(targets.size(0), device=targets.device), targets]

        alpha = self.class_weights[targets]
        focal_term = torch.pow(1.0 - target_probs, self.gamma)
        loss = -alpha * focal_term * target_log_probs
        return loss.mean()


def _counts_from_split_info(split_info: dict) -> dict:
    counts_info = split_info.get('split_stats', {}).get('counts', {})
    train_counts = {}
    val_counts = {}
    for task in PES_COLUMNS:
        task_info = counts_info.get(task, {})
        train_counts[task] = [int(x) for x in task_info.get('train_counts', [0, 0, 0])]
        val_counts[task] = [int(x) for x in task_info.get('val_counts', [0, 0, 0])]
    return {'train': train_counts, 'val': val_counts}


def _distribution_dict(count_list: list) -> dict:
    return {idx: int(count_list[idx]) for idx in range(len(count_list))}


def compute_auc_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]

    macro_auc = None
    try:
        macro_auc = float(roc_auc_score(one_hot, probs, multi_class='ovr', average='macro'))
    except ValueError:
        pass

    per_class_auc = {}
    roc_curve_data = {}
    valid_auc_values = []

    for cls in range(n_classes):
        class_key = str(cls)
        binary_true = (labels == cls).astype(int)
        class_scores = probs[:, cls]

        if binary_true.min() == binary_true.max():
            per_class_auc[class_key] = None
            roc_curve_data[class_key] = {'fpr': [], 'tpr': [], 'auc': None}
            continue

        class_auc = float(roc_auc_score(binary_true, class_scores))
        fpr, tpr, _ = roc_curve(binary_true, class_scores)
        per_class_auc[class_key] = class_auc
        roc_curve_data[class_key] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': class_auc,
        }
        valid_auc_values.append(class_auc)

    if macro_auc is None:
        macro_auc = float(np.mean(valid_auc_values)) if valid_auc_values else 0.0

    return {
        'auc': macro_auc,
        'per_class_auc': per_class_auc,
        'roc_curve': roc_curve_data,
    }


def compute_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> dict:
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    auc_metrics = compute_auc_metrics(labels, probs)

    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'auc': auc_metrics['auc'],
        'per_class_auc': auc_metrics['per_class_auc'],
        'roc_curve': auc_metrics['roc_curve'],
        'confusion_matrix': cm.tolist(),
    }


def train_epoch(model: nn.Module, train_loader, criterion_dict: dict, optimizer, device: str) -> dict:
    model.train()
    task_losses = {name: [] for name in PES_TASK_NAMES}

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        implant, control, global_view, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        global_view = global_view.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(implant, control, global_view)

        total_loss = 0
        for index, task_name in enumerate(PES_TASK_NAMES):
            task_labels = labels[:, index]
            task_logits = outputs[task_name]
            loss = criterion_dict[task_name](task_logits, task_labels)
            total_loss += loss
            task_losses[task_name].append(loss.item())

        total_loss.backward()
        optimizer.step()

        avg_loss = total_loss.item() / len(PES_TASK_NAMES)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    avg_losses = {name: np.mean(losses) for name, losses in task_losses.items()}
    avg_losses['total'] = np.mean([np.mean(losses) for losses in task_losses.values()])
    return avg_losses


@torch.no_grad()
def validate_epoch(model: nn.Module, val_loader, criterion_dict: dict, device: str) -> dict:
    model.eval()
    task_losses = {name: [] for name in PES_TASK_NAMES}
    task_preds = {name: [] for name in PES_TASK_NAMES}
    task_labels = {name: [] for name in PES_TASK_NAMES}
    task_probs = {name: [] for name in PES_TASK_NAMES}

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        implant, control, global_view, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        global_view = global_view.to(device)
        labels = labels.to(device)

        outputs = model(implant, control, global_view)

        for index, task_name in enumerate(PES_TASK_NAMES):
            task_label = labels[:, index]
            task_logits = outputs[task_name]
            loss = criterion_dict[task_name](task_logits, task_label)
            task_losses[task_name].append(loss.item())

            preds = task_logits.argmax(dim=-1)
            probs = torch.softmax(task_logits, dim=-1)
            task_preds[task_name].extend(preds.cpu().numpy())
            task_labels[task_name].extend(task_label.cpu().numpy())
            task_probs[task_name].extend(probs.cpu().numpy())

    results = {}
    for task_name in PES_TASK_NAMES:
        preds = np.array(task_preds[task_name])
        labels = np.array(task_labels[task_name])
        probs = np.array(task_probs[task_name])
        metrics = compute_metrics(preds, labels, probs)
        metrics['loss'] = np.mean(task_losses[task_name])
        results[task_name] = metrics

    results['avg_accuracy'] = float(np.mean([results[name]['accuracy'] for name in PES_TASK_NAMES]))
    results['avg_f1'] = float(np.mean([results[name]['f1'] for name in PES_TASK_NAMES]))
    results['avg_auc'] = float(np.mean([results[name]['auc'] for name in PES_TASK_NAMES]))
    results['avg_loss'] = float(np.mean([results[name]['loss'] for name in PES_TASK_NAMES]))
    return results


def train_legacy_balanced(
    data_dir: str,
    label_file: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    t_max: int = 30,
    eta_min_ratio: float = 0.01,
    test_size: float = 0.2,
    split_seed: int = 42,
    auto_search_split: bool = True,
    split_search_trials: int = 500,
    split_seed_base: int = 42,
    min_val_per_class: int = 1,
    min_train_per_class: int = 1,
    use_weighted_sampler: bool = True,
    sampler_strategy: str = 'max_task',
    loss_type: str = 'cb_focal',
    cb_beta: float = 0.999,
    focal_gamma: float = 1.5,
    selection_metric: str = 'avg_f1',
    num_workers: int = 4,
    device: str = 'cuda',
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Output directory: {run_dir}')
    print(f'Device: {device}')

    print('\n=== Creating Model ===')
    model, preprocess = create_model(device=device)

    print('\n=== Loading Data ===')
    train_loader, val_loader, class_weights, split_info = create_dataloaders(
        data_dir=data_dir,
        label_file=label_file,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        test_size=test_size,
        random_state=split_seed,
        auto_search_split=auto_search_split,
        split_search_trials=split_search_trials,
        split_seed_base=split_seed_base,
        min_val_per_class=min_val_per_class,
        min_train_per_class=min_train_per_class,
        train_augment=False,
        use_weighted_sampler=use_weighted_sampler,
        sampler_strategy=sampler_strategy,
    )

    data_summary = summarize_dataset(data_dir, label_file)
    split_counts = _counts_from_split_info(split_info)
    split_summary = {
        'mode': split_info.get('mode'),
        'seed': split_info.get('seed'),
        'score': split_info.get('score'),
        'feasible_trials': split_info.get('feasible_trials'),
        'trials': split_info.get('trials'),
        'sampler_strategy': split_info.get('sampler_strategy'),
        'train_samples': int(sum(split_counts['train'][PES_COLUMNS[0]])),
        'val_samples': int(sum(split_counts['val'][PES_COLUMNS[0]])),
        'train_label_distribution': {
            task: _distribution_dict(split_counts['train'][task])
            for task in PES_COLUMNS
        },
        'val_label_distribution': {
            task: _distribution_dict(split_counts['val'][task])
            for task in PES_COLUMNS
        },
    }

    criterion_dict = {}
    for task_name in PES_TASK_NAMES:
        if loss_type == 'cb_focal':
            class_counts = split_counts['train'][task_name]
            criterion = CBFocalLoss(class_counts=class_counts, beta=cb_beta, gamma=focal_gamma).to(device)
            criterion_dict[task_name] = criterion
            print(f'CB-Focal counts for {task_name}: {class_counts}')
        else:
            weights = class_weights[task_name].to(device)
            criterion_dict[task_name] = nn.CrossEntropyLoss(weight=weights)
            print(f'CE class weights for {task_name}: {weights.cpu().numpy()}')

    optimizer = AdamW(model.get_trainable_params(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr * eta_min_ratio)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'best_epoch': 0,
        'best_accuracy': 0.0,
        'best_f1': 0.0,
        'best_auc': 0.0,
        'best_selection_metric': selection_metric,
        'best_selection_value': float('-inf'),
    }

    run_metadata = {
        'model_name': 'model_legacy_balanced',
        'timestamp': timestamp,
        'data_dir': data_dir,
        'label_file': label_file,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        't_max': t_max,
        'eta_min_ratio': eta_min_ratio,
        'test_size': test_size,
        'split_seed': split_seed,
        'auto_search_split': auto_search_split,
        'split_search_trials': split_search_trials,
        'split_seed_base': split_seed_base,
        'min_val_per_class': min_val_per_class,
        'min_train_per_class': min_train_per_class,
        'use_weighted_sampler': use_weighted_sampler,
        'sampler_strategy': sampler_strategy,
        'loss_type': loss_type,
        'cb_beta': cb_beta,
        'focal_gamma': focal_gamma,
        'selection_metric': selection_metric,
        'num_workers': num_workers,
        'device': device,
    }

    with open(os.path.join(run_dir, 'model_meta.json'), 'w', encoding='utf-8') as file_obj:
        json.dump(run_metadata, file_obj, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, 'data_summary.json'), 'w', encoding='utf-8') as file_obj:
        json.dump({'dataset': data_summary, 'split': split_summary}, file_obj, ensure_ascii=False, indent=2)

    best_selection_value = float('-inf')

    print('\n=== Starting Training ===')
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')

        train_losses = train_epoch(model, train_loader, criterion_dict, optimizer, device)
        print(f"Train Loss: {train_losses['total']:.4f}")

        val_results = validate_epoch(model, val_loader, criterion_dict, device)
        print(f"Val Loss: {val_results['avg_loss']:.4f}")
        print(f"Val Accuracy: {val_results['avg_accuracy']:.4f}")
        print(f"Val F1: {val_results['avg_f1']:.4f}")
        print(f"Val AUC: {val_results['avg_auc']:.4f}")

        for task_name in PES_TASK_NAMES:
            metrics = val_results[task_name]
            print(
                f"  {task_name}: "
                f"Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"AUC={metrics['auc']:.4f}"
            )

        scheduler.step()

        history['train_loss'].append(float(train_losses['total']))
        history['val_loss'].append(float(val_results['avg_loss']))
        history['val_accuracy'].append(float(val_results['avg_accuracy']))
        history['val_f1'].append(float(val_results['avg_f1']))
        history['val_auc'].append(float(val_results['avg_auc']))

        current_selection_value = float(val_results[selection_metric])
        if current_selection_value > best_selection_value:
            best_selection_value = current_selection_value
            history['best_epoch'] = epoch
            history['best_accuracy'] = float(val_results['avg_accuracy'])
            history['best_f1'] = float(val_results['avg_f1'])
            history['best_auc'] = float(val_results['avg_auc'])
            history['best_selection_value'] = best_selection_value

            best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
            model.save_model(best_model_path)
            best_results_path = os.path.join(run_dir, 'best_results.json')
            with open(best_results_path, 'w', encoding='utf-8') as file_obj:
                json.dump(val_results, file_obj, ensure_ascii=False, indent=2)
            print(
                f"*** New best model saved ({selection_metric}={best_selection_value:.4f}, "
                f"Acc={val_results['avg_accuracy']:.4f}, AUC={val_results['avg_auc']:.4f}) ***"
            )

        last_model_path = os.path.join(ckpt_dir, 'last_model.pth')
        model.save_model(last_model_path)

        history_path = os.path.join(run_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as file_obj:
            json.dump(history, file_obj, ensure_ascii=False, indent=2)

    history_path = os.path.join(run_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as file_obj:
        json.dump(history, file_obj, ensure_ascii=False, indent=2)

    print('\n=== Training Complete ===')
    print(
        f"Best model at epoch {history['best_epoch']} "
        f"({selection_metric}={history['best_selection_value']:.4f}, "
        f"Acc={history['best_accuracy']:.4f}, AUC={history['best_auc']:.4f})"
    )
    print(f'Results saved to {run_dir}')

    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        from visualize_legacy import plot_training_curves, plot_confusion_matrices, plot_auc_curves
        curves_path = os.path.join(run_dir, 'training_curves.png')
        plot_training_curves(history_path, curves_path, show=False)
        best_results_path = os.path.join(run_dir, 'best_results.json')
        if os.path.exists(best_results_path):
            conf_path = os.path.join(run_dir, 'confusion_matrices.png')
            plot_confusion_matrices(best_results_path, conf_path, show=False)
            auc_path = os.path.join(run_dir, 'auc_curves.png')
            plot_auc_curves(best_results_path, auc_path, show=False)
    except Exception as exc:
        print(f'Visualization failed: {exc}')

    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Train PES Legacy with imbalance-aware optimization')
    parser.add_argument('--data_dir', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/红白美学标注',
                        help='Data directory path')
    parser.add_argument('--label_file', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/两次重新标注后校正.xlsx',
                        help='Label file path')
    parser.add_argument('--output_dir', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/legacy_balanced/outputs',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--t_max', type=int, default=30,
                        help='T_max for cosine scheduler')
    parser.add_argument('--eta_min_ratio', type=float, default=0.01,
                        help='Eta_min / lr ratio for cosine scheduler')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for fixed patient split')
    parser.add_argument('--auto_search_split', dest='auto_search_split', action='store_true',
                        help='Enable automatic seed search for balanced patient split')
    parser.add_argument('--no_auto_search_split', dest='auto_search_split', action='store_false',
                        help='Disable automatic split seed search')
    parser.add_argument('--split_search_trials', type=int, default=500,
                        help='Number of seeds to search when auto split is enabled')
    parser.add_argument('--split_seed_base', type=int, default=42,
                        help='Start seed for automatic split search')
    parser.add_argument('--min_val_per_class', type=int, default=1,
                        help='Minimum validation samples per class per task in split search')
    parser.add_argument('--min_train_per_class', type=int, default=1,
                        help='Minimum training samples per class per task in split search')
    parser.add_argument('--use_weighted_sampler', dest='use_weighted_sampler', action='store_true',
                        help='Use weighted random sampler to reduce class imbalance')
    parser.add_argument('--no_weighted_sampler', dest='use_weighted_sampler', action='store_false',
                        help='Disable weighted random sampler')
    parser.add_argument('--sampler_strategy', type=str, default='max_task',
                        choices=['max_task', 'avg_task', 'sum_task'],
                        help='How to aggregate per-task inverse frequencies into sample weight')
    parser.add_argument('--loss', dest='loss_type', type=str, default='cb_focal',
                        choices=['ce', 'cb_focal'],
                        help='Training loss type')
    parser.add_argument('--cb_beta', type=float, default=0.999,
                        help='Beta for class-balanced focal loss')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                        help='Gamma for focal loss')
    parser.add_argument('--selection_metric', type=str, default='avg_f1',
                        choices=['avg_accuracy', 'avg_f1', 'avg_auc'],
                        help='Metric used to select best checkpoint')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    parser.set_defaults(auto_search_split=True, use_weighted_sampler=True)

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'

    train_legacy_balanced(
        data_dir=args.data_dir,
        label_file=args.label_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        t_max=args.t_max,
        eta_min_ratio=args.eta_min_ratio,
        test_size=args.test_size,
        split_seed=args.split_seed,
        auto_search_split=args.auto_search_split,
        split_search_trials=args.split_search_trials,
        split_seed_base=args.split_seed_base,
        min_val_per_class=args.min_val_per_class,
        min_train_per_class=args.min_train_per_class,
        use_weighted_sampler=args.use_weighted_sampler,
        sampler_strategy=args.sampler_strategy,
        loss_type=args.loss_type,
        cb_beta=args.cb_beta,
        focal_gamma=args.focal_gamma,
        selection_metric=args.selection_metric,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == '__main__':
    main()
