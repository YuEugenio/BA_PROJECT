"""
Train baseline-style two-stream ResNet50 model.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_data_items, compute_label_distribution, create_dataloaders, patient_group_split, summarize_dataset
from model_baseline import PES_TASK_NAMES, create_model


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

    return {'auc': macro_auc, 'per_class_auc': per_class_auc, 'roc_curve': roc_curve_data}


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
        implant, control, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(implant, control)

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
        implant, control, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        labels = labels.to(device)

        outputs = model(implant, control)

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


def train_baseline(
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
    train_loader, val_loader, class_weights = create_dataloaders(
        data_dir=data_dir,
        label_file=label_file,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        test_size=test_size,
        random_state=split_seed,
        train_augment=False,
        use_weighted_sampler=False,
    )

    data_summary = summarize_dataset(data_dir, label_file)
    data_items = build_data_items(data_dir, label_file)
    split_train, split_val = patient_group_split(data_items, test_size, split_seed)
    split_summary = {
        'train_samples': len(split_train),
        'val_samples': len(split_val),
        'train_label_distribution': compute_label_distribution(split_train),
        'val_label_distribution': compute_label_distribution(split_val),
    }

    criterion_dict = {}
    for task_name in PES_TASK_NAMES:
        weights = class_weights[task_name].to(device)
        criterion_dict[task_name] = nn.CrossEntropyLoss(weight=weights)
        print(f'Class weights for {task_name}: {weights.cpu().numpy()}')

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
        'best_auc': 0.0,
    }

    run_metadata = {
        'model_name': 'baseline_resnet50_crossattn_linear',
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
        'num_workers': num_workers,
        'device': device,
        'streams': ['implant', 'control'],
        'fusion': 'cross_attention_bidirectional',
        'head': 'single_linear',
        'backbone': 'ResNet50 ImageNet pretrained',
    }

    with open(os.path.join(run_dir, 'model_meta.json'), 'w', encoding='utf-8') as file_obj:
        json.dump(run_metadata, file_obj, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, 'data_summary.json'), 'w', encoding='utf-8') as file_obj:
        json.dump({'dataset': data_summary, 'split': split_summary}, file_obj, ensure_ascii=False, indent=2)

    best_accuracy = 0.0

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

        if val_results['avg_accuracy'] > best_accuracy:
            best_accuracy = float(val_results['avg_accuracy'])
            history['best_epoch'] = epoch
            history['best_accuracy'] = best_accuracy
            history['best_auc'] = float(val_results['avg_auc'])

            best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
            model.save_model(best_model_path)

            best_results_path = os.path.join(run_dir, 'best_results.json')
            with open(best_results_path, 'w', encoding='utf-8') as file_obj:
                json.dump(val_results, file_obj, ensure_ascii=False, indent=2)
            print(f"*** New best model saved (Accuracy: {best_accuracy:.4f}, AUC: {val_results['avg_auc']:.4f}) ***")

        last_model_path = os.path.join(ckpt_dir, 'last_model.pth')
        model.save_model(last_model_path)

        history_path = os.path.join(run_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as file_obj:
            json.dump(history, file_obj, ensure_ascii=False, indent=2)

    history_path = os.path.join(run_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as file_obj:
        json.dump(history, file_obj, ensure_ascii=False, indent=2)

    print('\n=== Training Complete ===')
    print(f"Best model at epoch {history['best_epoch']} with accuracy {history['best_accuracy']:.4f}")
    print(f"Best model avg AUC: {history['best_auc']:.4f}")
    print(f'Results saved to {run_dir}')

    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        from visualize import plot_auc_curves, plot_confusion_matrices, plot_training_curves

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
    parser = argparse.ArgumentParser(description='Train baseline ResNet50 (cross-attn + linear)')
    parser.add_argument('--data_dir', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/红白美学标注',
                        help='Data directory path')
    parser.add_argument('--label_file', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/两次重新标注后校正.xlsx',
                        help='Label file path')
    parser.add_argument('--output_dir', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/Baseline/outputs',
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
                        help='Random seed for patient split')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'

    train_baseline(
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
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == '__main__':
    main()
