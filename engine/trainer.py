"""Unified training and validation engine."""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score, roc_curve


def compute_auc_metrics(labels, probs):
    """Compute macro AUC and per-class ROC curves."""
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]

    macro_auc = None
    try:
        macro_auc = float(roc_auc_score(one_hot, probs, multi_class='ovr', average='macro'))
    except ValueError:
        pass

    per_class_auc = {}
    roc_curve_data = {}
    valid_aucs = []

    for cls in range(n_classes):
        key = str(cls)
        binary_true = (labels == cls).astype(int)
        scores = probs[:, cls]

        if binary_true.min() == binary_true.max():
            per_class_auc[key] = None
            roc_curve_data[key] = {'fpr': [], 'tpr': [], 'auc': None}
            continue

        auc_val = float(roc_auc_score(binary_true, scores))
        fpr, tpr, _ = roc_curve(binary_true, scores)
        per_class_auc[key] = auc_val
        roc_curve_data[key] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc_val}
        valid_aucs.append(auc_val)

    if macro_auc is None:
        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    return {'auc': macro_auc, 'per_class_auc': per_class_auc, 'roc_curve': roc_curve_data}


def compute_metrics(preds, labels, probs):
    """Compute accuracy, F1, recall, AUC, and confusion matrix."""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    auc_metrics = compute_auc_metrics(labels, probs)
    return {
        'accuracy': acc, 'f1': f1, 'recall': recall,
        'auc': auc_metrics['auc'],
        'per_class_auc': auc_metrics['per_class_auc'],
        'roc_curve': auc_metrics['roc_curve'],
        'confusion_matrix': cm.tolist(),
    }


def train_epoch(model, train_loader, criterion_dict, optimizer, device, task_keys):
    """Run one training epoch."""
    model.train()
    task_losses = {key: [] for key in task_keys}

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        *inputs, labels = batch
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(*inputs)

        total_loss = 0
        for i, key in enumerate(task_keys):
            loss = criterion_dict[key](outputs[key], labels[:, i])
            total_loss += loss
            task_losses[key].append(loss.item())

        total_loss.backward()
        optimizer.step()

        avg_loss = total_loss.item() / len(task_keys)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    avg_losses = {key: np.mean(losses) for key, losses in task_losses.items()}
    avg_losses['total'] = np.mean(list(avg_losses.values()))
    return avg_losses


@torch.no_grad()
def validate_epoch(model, val_loader, criterion_dict, device, task_keys):
    """Run one validation epoch."""
    model.eval()
    task_losses = {key: [] for key in task_keys}
    task_preds = {key: [] for key in task_keys}
    task_labels = {key: [] for key in task_keys}
    task_probs = {key: [] for key in task_keys}

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        *inputs, labels = batch
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        outputs = model(*inputs)

        for i, key in enumerate(task_keys):
            task_label = labels[:, i]
            logits = outputs[key]
            loss = criterion_dict[key](logits, task_label)
            task_losses[key].append(loss.item())

            preds = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)
            task_preds[key].extend(preds.cpu().numpy())
            task_labels[key].extend(task_label.cpu().numpy())
            task_probs[key].extend(probs.cpu().numpy())

    results = {}
    for key in task_keys:
        preds = np.array(task_preds[key])
        lbls = np.array(task_labels[key])
        prbs = np.array(task_probs[key])
        metrics = compute_metrics(preds, lbls, prbs)
        metrics['loss'] = np.mean(task_losses[key])
        results[key] = metrics

    results['avg_accuracy'] = np.mean([results[k]['accuracy'] for k in task_keys])
    results['avg_f1'] = np.mean([results[k]['f1'] for k in task_keys])
    results['avg_auc'] = np.mean([results[k]['auc'] for k in task_keys])
    results['avg_loss'] = np.mean([results[k]['loss'] for k in task_keys])

    return results
