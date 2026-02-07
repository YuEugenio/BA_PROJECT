"""
visualize.py - Visualization module
Visualization module for PES Multi-Task Classification

Features:
1. Training curve visualization
2. Confusion matrix visualization
3. Prediction result visualization with ROI overlay
"""

import os
import json
import argparse
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

from dataset import parse_labelme_json, PES_COLUMNS
from model import PES_TASK_NAMES


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Color configuration
COLORS = {
    'implant': '#FF6B6B',
    'control': '#4ECDC4',
    'global': '#45B7D1',
}

# PES score colors
SCORE_COLORS = {
    0: '#E74C3C',
    1: '#F39C12',
    2: '#27AE60',
}

TASK_NAME_MAP = {
    '近中牙龈乳头': 'Mesial Papilla',
    '远中牙龈乳头': 'Distal Papilla',
    '软组织形态': 'Soft Tissue Contour',
    '粘膜颜色': 'Mucosal Color',
    '黏膜颜色': 'Mucosal Color',
}

ROI_NAME_MAP = {
    'implant': 'Implant ROI',
    'control': 'Control ROI',
    'global': 'Global ROI',
}


def display_task_name(name: str) -> str:
    return TASK_NAME_MAP.get(name, name)


def display_column_name(name: str) -> str:
    return TASK_NAME_MAP.get(name, name)


def plot_training_curves(
    history_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves
    
    Args:
        history_path: Path to training_history.json
        output_path: Output image path (optional)
        show: Whether to display the plot
    """
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss curve
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, history['val_accuracy'], 'g-', linewidth=2)
    ax.axhline(y=history['best_accuracy'], color='r', linestyle='--', 
               label=f'Best: {history["best_accuracy"]:.4f}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # F1 score curve
    ax = axes[2]
    ax.plot(epochs, history['val_f1'], 'm-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Validation F1 Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    results_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot confusion matrices for each task
    
    Args:
        results_path: Path to validation results JSON
        output_path: Output image path (optional)
        show: Whether to display the plot
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    class_labels = ['0 (Poor)', '1 (Fair)', '2 (Good)']
    
    for idx, task_name in enumerate(PES_TASK_NAMES):
        ax = axes[idx]
        cm = np.array(results[task_name]['confusion_matrix'])
        
        # Plot heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            annot_kws={'size': 14}
        )
        
        # Set labels
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        title_name = display_task_name(task_name)
        ax.set_title(f'{title_name}\nAcc: {results[task_name]["accuracy"]:.3f}, F1: {results[task_name]["f1"]:.3f}',
                     fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrices saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_prediction(
    image_path: str,
    json_path: str,
    prediction: Dict,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize prediction result for a single image
    
    Args:
        image_path: Image file path
        json_path: LabelMe JSON path
        prediction: Prediction result dict
        output_path: Output image path (optional)
        show: Whether to display the plot
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Parse ROIs
    rois = parse_labelme_json(json_path)
    
    # Draw ROI bounding boxes
    for roi_name, bbox in rois.items():
        x1, y1, x2, y2 = [int(c) for c in bbox]
        color = COLORS.get(roi_name, '#FFFFFF')
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = ROI_NAME_MAP.get(roi_name, roi_name)
        draw.text((x1, y1 - 20), label, fill=color)
    
    # Create combined view
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Image
    axes[0].imshow(np.array(image))
    axes[0].set_title('Image with ROIs', fontsize=14)
    axes[0].axis('off')
    
    # Right: Predictions
    ax = axes[1]
    ax.axis('off')
    
    # Build result text
    result_text = "PES Prediction Results\n" + "=" * 40 + "\n\n"
    
    for task_name in PES_TASK_NAMES:
        if task_name in prediction.get('predictions', {}):
            pred = prediction['predictions'][task_name]
            score = pred['class']
            conf = pred['confidence']
            result_text += f"{display_task_name(task_name)}:\n"
            result_text += f"  Score: {score} ({['Poor', 'Fair', 'Good'][score]})\n"
            result_text += f"  Confidence: {conf:.1%}\n\n"
    
    if 'total_pes_score' in prediction:
        result_text += "=" * 40 + "\n"
        result_text += f"Total PES Score: {prediction['total_pes_score']}/{prediction['max_possible_score']}\n"
    
    ax.text(0.1, 0.9, result_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Draw score bar chart
    scores = [prediction['predictions'][name]['class'] for name in PES_TASK_NAMES]
    colors = [SCORE_COLORS[s] for s in scores]
    
    # Add bar chart at bottom right
    ax_bar = fig.add_axes([0.55, 0.1, 0.4, 0.3])
    bars = ax_bar.bar(range(4), scores, color=colors)
    ax_bar.set_xticks(range(4))
    ax_bar.set_xticklabels([display_task_name(name) for name in PES_TASK_NAMES], rotation=30, ha='right')
    ax_bar.set_ylabel('Score')
    ax_bar.set_ylim(0, 2.5)
    ax_bar.set_title('PES Scores')
    
    # Add values on bars
    for bar, score in zip(bars, scores):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(score), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    data_dir: str,
    label_file: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot class distribution
    
    Args:
        data_dir: Data directory
        label_file: Label file path
        output_path: Output image path (optional)
        show: Whether to display the plot
    """
    import pandas as pd
    
    df = pd.read_excel(label_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#E74C3C', '#F39C12', '#27AE60']
    
    for idx, col in enumerate(PES_COLUMNS):
        ax = axes[idx]
        counts = df[col].value_counts().sort_index()
        
        bars = ax.bar(counts.index, counts.values, color=colors)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(display_column_name(col), fontsize=14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['0 (Poor)', '1 (Fair)', '2 (Good)'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Class Distribution for 4 PES Tasks', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='PES Model Visualization')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'confusion', 'prediction', 'distribution'],
                        help='Visualization mode')
    parser.add_argument('--history', type=str,
                        help='Path to training_history.json')
    parser.add_argument('--results', type=str,
                        help='Path to validation results JSON')
    parser.add_argument('--image', type=str,
                        help='Path to image for prediction visualization')
    parser.add_argument('--json', type=str,
                        help='Path to LabelMe JSON')
    parser.add_argument('--prediction', type=str,
                        help='Path to prediction JSON')
    parser.add_argument('--data_dir', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/红白美学标注',
                        help='Data directory')
    parser.add_argument('--label_file', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/两次重新标注后校正.xlsx',
                        help='Label file path')
    parser.add_argument('--output', type=str,
                        help='Output image path')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display the plot')
    
    args = parser.parse_args()
    show = not args.no_show
    
    if args.mode == 'training':
        if not args.history:
            print("Please specify --history for training curve visualization")
            return
        plot_training_curves(args.history, args.output, show)
        
    elif args.mode == 'confusion':
        if not args.results:
            print("Please specify --results for confusion matrix visualization")
            return
        plot_confusion_matrices(args.results, args.output, show)
        
    elif args.mode == 'prediction':
        if not all([args.image, args.json, args.prediction]):
            print("Please specify --image, --json, and --prediction for prediction visualization")
            return
        with open(args.prediction, 'r', encoding='utf-8') as f:
            prediction = json.load(f)
        visualize_prediction(args.image, args.json, prediction, args.output, show)
        
    elif args.mode == 'distribution':
        plot_class_distribution(args.data_dir, args.label_file, args.output, show)


if __name__ == '__main__':
    main()
