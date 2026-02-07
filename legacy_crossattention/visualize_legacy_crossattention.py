"""
visualize_legacy.py - Visualization module for legacy training outputs
"""

import json
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_legacy_crossattention import PES_TASK_NAMES


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TASK_NAME_MAP = {
    '近中牙龈乳头': 'Mesial Papilla',
    '远中牙龈乳头': 'Distal Papilla',
    '软组织形态': 'Soft Tissue Contour',
    '粘膜颜色': 'Mucosal Color',
    '黏膜颜色': 'Mucosal Color',
}

CLASS_NAME_MAP = {
    '0': 'Class 0 (Poor)',
    '1': 'Class 1 (Fair)',
    '2': 'Class 2 (Good)',
}


def display_task_name(name: str) -> str:
    return TASK_NAME_MAP.get(name, name)


def plot_training_curves(
    history_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    with open(history_path, 'r', encoding='utf-8') as file_obj:
        history = json.load(file_obj)

    epochs = range(1, len(history['train_loss']) + 1)
    has_auc = 'val_auc' in history and len(history['val_auc']) == len(history['train_loss'])

    n_cols = 4 if has_auc else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history['val_accuracy'], 'g-', linewidth=2)
    ax.axhline(y=history['best_accuracy'], color='r', linestyle='--',
               label=f'Best: {history["best_accuracy"]:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history['val_f1'], 'm-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score')
    ax.grid(True, alpha=0.3)

    if has_auc:
        ax = axes[3]
        ax.plot(epochs, history['val_auc'], color='#8E44AD', linewidth=2)
        if 'best_auc' in history:
            ax.axhline(y=history['best_auc'], color='#1ABC9C', linestyle='--',
                       label=f'Best model AUC: {history["best_auc"]:.4f}')
            ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('Validation AUC')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Training curves saved to {output_path}')

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    results_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    with open(results_path, 'r', encoding='utf-8') as file_obj:
        results = json.load(file_obj)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    class_labels = ['0 (Poor)', '1 (Fair)', '2 (Good)']

    for index, task_name in enumerate(PES_TASK_NAMES):
        ax = axes[index]
        cm = np.array(results[task_name]['confusion_matrix'])

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            annot_kws={'size': 12}
        )

        acc = results[task_name]['accuracy']
        f1 = results[task_name]['f1']
        auc = results[task_name].get('auc', None)
        title_name = display_task_name(task_name)
        if auc is None:
            ax.set_title(f'{title_name}\nAcc: {acc:.3f}, F1: {f1:.3f}', fontsize=12)
        else:
            ax.set_title(f'{title_name}\nAcc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}', fontsize=12)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Confusion matrices saved to {output_path}')

    if show:
        plt.show()
    else:
        plt.close()


def plot_auc_curves(
    results_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    with open(results_path, 'r', encoding='utf-8') as file_obj:
        results = json.load(file_obj)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for index, task_name in enumerate(PES_TASK_NAMES):
        ax = axes[index]
        task_result = results.get(task_name, {})
        roc_data = task_result.get('roc_curve', {})

        plotted = False
        for class_key, class_name in CLASS_NAME_MAP.items():
            class_curve = roc_data.get(class_key, {})
            fpr = class_curve.get('fpr', [])
            tpr = class_curve.get('tpr', [])
            class_auc = class_curve.get('auc', None)

            if len(fpr) > 0 and len(tpr) > 0 and class_auc is not None:
                ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={class_auc:.3f})')
                plotted = True

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        macro_auc = task_result.get('auc', None)
        title_name = display_task_name(task_name)
        if macro_auc is None:
            ax.set_title(f'{title_name} ROC Curves')
        else:
            ax.set_title(f'{title_name} ROC Curves (Macro AUC={macro_auc:.3f})')

        if plotted:
            ax.legend(loc='lower right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'ROC unavailable\n(single-class labels in val set)',
                    ha='center', va='center', fontsize=10, alpha=0.7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'AUC curves saved to {output_path}')

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualization for legacy training outputs')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'confusion', 'auc'],
                        help='Visualization mode')
    parser.add_argument('--history', type=str, help='Path to training_history.json')
    parser.add_argument('--results', type=str, help='Path to validation results JSON')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--no_show', action='store_true', help='Do not display the plot')

    args = parser.parse_args()
    show = not args.no_show

    if args.mode == 'training':
        if not args.history:
            print('Please specify --history for training curve visualization')
            return
        plot_training_curves(args.history, args.output, show)
    elif args.mode == 'confusion':
        if not args.results:
            print('Please specify --results for confusion matrix visualization')
            return
        plot_confusion_matrices(args.results, args.output, show)
    elif args.mode == 'auc':
        if not args.results:
            print('Please specify --results for AUC curve visualization')
            return
        plot_auc_curves(args.results, args.output, show)


if __name__ == '__main__':
    main()
