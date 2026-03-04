"""Unified visualization for training outputs."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAME_MAP = {
    '0': 'Class 0 (Poor)',
    '1': 'Class 1 (Fair)',
    '2': 'Class 2 (Good)',
}


def plot_training_curves(history_path, output_path=None, show=False):
    """Plot training loss, val accuracy, val F1, and val AUC curves."""
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    has_auc = 'val_auc' in history and len(history['val_auc']) == len(history['train_loss'])
    selection_metric = history.get('selection_metric', 'avg_accuracy')
    best_epoch = history.get('best_epoch', 0)

    n_cols = 4 if has_auc else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_accuracy'], 'g-', linewidth=2)
    best_acc = history.get('best_accuracy', None)
    if best_acc is not None:
        if selection_metric == 'avg_auc':
            acc_label = f"Acc@BestAUC: {best_acc:.4f}"
        else:
            acc_label = f"Best Acc: {best_acc:.4f}"
        axes[1].axhline(y=best_acc, color='r', linestyle='--', label=acc_label)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_f1'], 'm-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].grid(True, alpha=0.3)

    if has_auc:
        axes[3].plot(epochs, history['val_auc'], color='#8E44AD', linewidth=2)
        if 'best_auc' in history:
            if selection_metric == 'avg_auc':
                auc_label = f"Best AUC: {history['best_auc']:.4f}"
            else:
                auc_label = f"AUC@BestAcc: {history['best_auc']:.4f}"
            axes[3].axhline(y=history['best_auc'], color='#1ABC9C', linestyle='--',
                           label=auc_label)
            axes[3].legend()
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('AUC')
        axes[3].set_title('Validation AUC')
        axes[3].grid(True, alpha=0.3)

    if isinstance(best_epoch, int) and best_epoch > 0:
        for ax in axes:
            ax.axvline(best_epoch, color='gray', linestyle=':', linewidth=1, alpha=0.8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(results_path, task_keys, display_names, output_path=None, show=False):
    """Plot confusion matrices for each task."""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    n_tasks = len(task_keys)
    n_cols = min(n_tasks, 4)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    class_labels = ['0 (Poor)', '1 (Fair)', '2 (Good)']

    for i, (key, name) in enumerate(zip(task_keys, display_names)):
        if i >= len(axes):
            break
        ax = axes[i]
        cm = np.array(results[key]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=ax, annot_kws={'size': 12})
        acc = results[key]['accuracy']
        f1 = results[key]['f1']
        auc = results[key].get('auc', None)
        if auc is not None:
            ax.set_title(f'{name}\nAcc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}', fontsize=10)
        else:
            ax.set_title(f'{name}\nAcc: {acc:.3f}, F1: {f1:.3f}', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrices saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_auc_curves(results_path, task_keys, display_names, output_path=None, show=False):
    """Plot ROC/AUC curves for each task."""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    n_tasks = len(task_keys)
    n_cols = min(n_tasks, 4)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (key, name) in enumerate(zip(task_keys, display_names)):
        if i >= len(axes):
            break
        ax = axes[i]
        task_result = results.get(key, {})
        roc_data = task_result.get('roc_curve', {})

        plotted = False
        for cls_key, cls_name in CLASS_NAME_MAP.items():
            curve = roc_data.get(cls_key, {})
            fpr, tpr = curve.get('fpr', []), curve.get('tpr', [])
            cls_auc = curve.get('auc', None)
            if fpr and tpr and cls_auc is not None:
                ax.plot(fpr, tpr, linewidth=2, label=f'{cls_name} (AUC={cls_auc:.3f})')
                plotted = True

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        macro_auc = task_result.get('auc', None)
        if macro_auc is not None:
            ax.set_title(f'{name} (Macro AUC={macro_auc:.3f})', fontsize=10)
        else:
            ax.set_title(f'{name} ROC', fontsize=10)

        if plotted:
            ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"AUC curves saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()
