"""
visualize.py - 可视化模块
Visualization module for PES Multi-Task Classification

功能 / Features:
1. 训练曲线可视化 / Training curve visualization
2. 混淆矩阵可视化 / Confusion matrix visualization
3. 预测结果可视化 / Prediction result visualization with ROI overlay
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


# 设置中文字体 / Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置 / Color configuration
COLORS = {
    'implant': '#FF6B6B',   # 红色 / Red
    'control': '#4ECDC4',   # 青色 / Cyan
    'global': '#45B7D1',    # 蓝色 / Blue
}

# PES分数颜色 / PES score colors
SCORE_COLORS = {
    0: '#E74C3C',  # 红色 - 较差 / Red - Poor
    1: '#F39C12',  # 橙色 - 中等 / Orange - Fair  
    2: '#27AE60',  # 绿色 - 良好 / Green - Good
}


def plot_training_curves(
    history_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制训练曲线
    Plot training curves
    
    Args:
        history_path: 训练历史JSON文件路径
        output_path: 输出图像路径（可选）
        show: 是否显示图像
    """
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 损失曲线 / Loss curve
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 准确率曲线 / Accuracy curve
    ax = axes[1]
    ax.plot(epochs, history['val_accuracy'], 'g-', linewidth=2)
    ax.axhline(y=history['best_accuracy'], color='r', linestyle='--', 
               label=f'Best: {history["best_accuracy"]:.4f}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # F1分数曲线 / F1 score curve
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
    绘制混淆矩阵
    Plot confusion matrices for each task
    
    Args:
        results_path: 验证结果JSON文件路径
        output_path: 输出图像路径（可选）
        show: 是否显示图像
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    class_labels = ['0 (较差)', '1 (中等)', '2 (良好)']
    
    for idx, task_name in enumerate(PES_TASK_NAMES):
        ax = axes[idx]
        cm = np.array(results[task_name]['confusion_matrix'])
        
        # 绘制热力图 / Plot heatmap
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
        
        # 设置标签 / Set labels
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{task_name}\nAcc: {results[task_name]["accuracy"]:.3f}, F1: {results[task_name]["f1"]:.3f}', 
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
    可视化单张图像的预测结果
    Visualize prediction result for a single image
    
    Args:
        image_path: 图像文件路径
        json_path: LabelMe JSON文件路径
        prediction: 预测结果字典
        output_path: 输出图像路径（可选）
        show: 是否显示图像
    """
    # 加载图像 / Load image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # 解析ROI / Parse ROIs
    rois = parse_labelme_json(json_path)
    
    # 绘制ROI边界框 / Draw ROI bounding boxes
    roi_names_cn = {'implant': '种植牙', 'control': '对侧牙', 'global': '上颌前牙'}
    for roi_name, bbox in rois.items():
        x1, y1, x2, y2 = [int(c) for c in bbox]
        color = COLORS.get(roi_name, '#FFFFFF')
        
        # 绘制矩形 / Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签 / Draw label
        label = roi_names_cn.get(roi_name, roi_name)
        draw.text((x1, y1 - 20), label, fill=color)
    
    # 创建图像和结果的组合视图 / Create combined view
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左侧：图像 / Left: Image
    axes[0].imshow(np.array(image))
    axes[0].set_title('Image with ROIs', fontsize=14)
    axes[0].axis('off')
    
    # 右侧：预测结果 / Right: Predictions
    ax = axes[1]
    ax.axis('off')
    
    # 构建结果文本 / Build result text
    result_text = "PES Prediction Results\n" + "=" * 40 + "\n\n"
    
    for task_name in PES_TASK_NAMES:
        if task_name in prediction.get('predictions', {}):
            pred = prediction['predictions'][task_name]
            score = pred['class']
            conf = pred['confidence']
            color_name = ['Red', 'Orange', 'Green'][score]
            result_text += f"{task_name}:\n"
            result_text += f"  Score: {score} ({['Poor', 'Fair', 'Good'][score]})\n"
            result_text += f"  Confidence: {conf:.1%}\n\n"
    
    if 'total_pes_score' in prediction:
        result_text += "=" * 40 + "\n"
        result_text += f"Total PES Score: {prediction['total_pes_score']}/{prediction['max_possible_score']}\n"
    
    ax.text(0.1, 0.9, result_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 绘制分数条形图 / Draw score bar chart
    scores = [prediction['predictions'][name]['class'] for name in PES_TASK_NAMES]
    colors = [SCORE_COLORS[s] for s in scores]
    
    # 在右侧下方添加条形图 / Add bar chart at bottom right
    ax_bar = fig.add_axes([0.55, 0.1, 0.4, 0.3])
    bars = ax_bar.bar(range(4), scores, color=colors)
    ax_bar.set_xticks(range(4))
    ax_bar.set_xticklabels([name[:4] for name in PES_TASK_NAMES], rotation=45)
    ax_bar.set_ylabel('Score')
    ax_bar.set_ylim(0, 2.5)
    ax_bar.set_title('PES Scores')
    
    # 在条形图上添加数值 / Add values on bars
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
    绘制类别分布图
    Plot class distribution
    
    Args:
        data_dir: 数据目录
        label_file: 标签文件路径
        output_path: 输出图像路径（可选）
        show: 是否显示图像
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
        ax.set_title(f'{col}', fontsize=14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['0 (较差)', '1 (中等)', '2 (良好)'])
        
        # 添加数值标签 / Add value labels
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
