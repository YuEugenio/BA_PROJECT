"""
train.py - PES多任务分类模型训练脚本
Training script for PES Multi-Task Classification Model

功能 / Features:
1. 加权交叉熵损失 / Weighted CrossEntropy Loss
2. AdamW优化器 + Cosine Annealing / AdamW + Cosine Annealing scheduler
3. 多指标评估 / Multi-metric evaluation (Accuracy, F1, Recall, Confusion Matrix)
4. 最佳模型保存 / Best model checkpointing
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from tqdm import tqdm

from dataset import create_dataloaders, PES_COLUMNS
from model import create_model, PES_TASK_NAMES


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    计算分类指标
    Compute classification metrics
    
    Args:
        preds: 预测标签 [N]
        labels: 真实标签 [N]
        
    Returns:
        包含各项指标的字典
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    
    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion_dict: dict,
    optimizer,
    device: str
) -> dict:
    """
    训练一个epoch
    Train for one epoch
    
    Returns:
        各任务的平均损失
    """
    model.train()
    task_losses = {name: [] for name in PES_TASK_NAMES}
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        implant, control, global_view, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        global_view = global_view.to(device)
        labels = labels.to(device)  # [B, 4]
        
        optimizer.zero_grad()
        
        # 前向传播 / Forward pass
        outputs = model(implant, control, global_view)
        
        # 计算各任务损失 / Compute loss for each task
        total_loss = 0
        for i, task_name in enumerate(PES_TASK_NAMES):
            task_labels = labels[:, i]  # [B]
            task_logits = outputs[task_name]  # [B, 3]
            loss = criterion_dict[task_name](task_logits, task_labels)
            total_loss += loss
            task_losses[task_name].append(loss.item())
        
        # 反向传播 / Backward pass
        total_loss.backward()
        optimizer.step()
        
        # 更新进度条
        avg_loss = total_loss.item() / len(PES_TASK_NAMES)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # 计算各任务平均损失 / Compute average loss for each task
    avg_losses = {name: np.mean(losses) for name, losses in task_losses.items()}
    avg_losses['total'] = np.mean([np.mean(losses) for losses in task_losses.values()])
    
    return avg_losses


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader,
    criterion_dict: dict,
    device: str
) -> dict:
    """
    验证一个epoch
    Validate for one epoch
    
    Returns:
        各任务的损失和指标
    """
    model.eval()
    task_losses = {name: [] for name in PES_TASK_NAMES}
    task_preds = {name: [] for name in PES_TASK_NAMES}
    task_labels = {name: [] for name in PES_TASK_NAMES}
    
    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        implant, control, global_view, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        global_view = global_view.to(device)
        labels = labels.to(device)
        
        # 前向传播 / Forward pass
        outputs = model(implant, control, global_view)
        
        # 计算各任务损失和预测 / Compute loss and predictions for each task
        for i, task_name in enumerate(PES_TASK_NAMES):
            task_label = labels[:, i]
            task_logits = outputs[task_name]
            loss = criterion_dict[task_name](task_logits, task_label)
            task_losses[task_name].append(loss.item())
            
            preds = task_logits.argmax(dim=-1)
            task_preds[task_name].extend(preds.cpu().numpy())
            task_labels[task_name].extend(task_label.cpu().numpy())
    
    # 计算各任务指标 / Compute metrics for each task
    results = {}
    for task_name in PES_TASK_NAMES:
        preds = np.array(task_preds[task_name])
        labels = np.array(task_labels[task_name])
        metrics = compute_metrics(preds, labels)
        metrics['loss'] = np.mean(task_losses[task_name])
        results[task_name] = metrics
    
    # 计算总体平均准确率 / Compute overall average accuracy
    results['avg_accuracy'] = np.mean([results[name]['accuracy'] for name in PES_TASK_NAMES])
    results['avg_f1'] = np.mean([results[name]['f1'] for name in PES_TASK_NAMES])
    results['avg_loss'] = np.mean([results[name]['loss'] for name in PES_TASK_NAMES])
    
    return results


def train(
    data_dir: str,
    label_file: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_workers: int = 4,
    device: str = 'cuda'
):
    """
    训练主函数
    Main training function
    
    Args:
        data_dir: 数据目录
        label_file: 标签文件路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        num_workers: 数据加载进程数
        device: 设备
    """
    # 创建输出目录 / Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    print(f"Device: {device}")
    
    # 创建模型 / Create model
    print("\n=== Creating Model ===")
    model, preprocess = create_model(device=device)
    
    # 创建数据加载器 / Create data loaders
    print("\n=== Loading Data ===")
    train_loader, val_loader, class_weights = create_dataloaders(
        data_dir=data_dir,
        label_file=label_file,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # 创建各任务的加权损失函数 / Create weighted loss for each task
    criterion_dict = {}
    for task_name in PES_TASK_NAMES:
        weights = class_weights[task_name].to(device)
        criterion_dict[task_name] = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights for {task_name}: {weights.cpu().numpy()}")
    
    # 创建优化器和调度器 / Create optimizer and scheduler
    trainable_params = model.get_trainable_params()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # 训练记录 / Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'best_epoch': 0,
        'best_accuracy': 0
    }
    
    best_accuracy = 0
    
    print("\n=== Starting Training ===")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 训练 / Train
        train_losses = train_epoch(model, train_loader, criterion_dict, optimizer, device)
        print(f"Train Loss: {train_losses['total']:.4f}")
        
        # 验证 / Validate
        val_results = validate_epoch(model, val_loader, criterion_dict, device)
        print(f"Val Loss: {val_results['avg_loss']:.4f}")
        print(f"Val Accuracy: {val_results['avg_accuracy']:.4f}")
        print(f"Val F1: {val_results['avg_f1']:.4f}")
        
        # 打印各任务指标 / Print metrics for each task
        for task_name in PES_TASK_NAMES:
            metrics = val_results[task_name]
            print(f"  {task_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")
        
        # 更新学习率 / Update learning rate
        scheduler.step()
        
        # 记录历史 / Record history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_results['avg_loss'])
        history['val_accuracy'].append(val_results['avg_accuracy'])
        history['val_f1'].append(val_results['avg_f1'])
        
        # 保存最佳模型 / Save best model
        if val_results['avg_accuracy'] > best_accuracy:
            best_accuracy = val_results['avg_accuracy']
            history['best_epoch'] = epoch
            history['best_accuracy'] = best_accuracy
            
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            model.save_model(best_model_path)
            print(f"*** New best model saved (Accuracy: {best_accuracy:.4f}) ***")
            
            # 保存最佳结果 / Save best results
            best_results_path = os.path.join(run_dir, 'best_results.json')
            with open(best_results_path, 'w', encoding='utf-8') as f:
                json.dump(val_results, f, ensure_ascii=False, indent=2)
    
    # 保存训练历史 / Save training history
    history_path = os.path.join(run_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print("\n=== Training Complete ===")
    print(f"Best model at epoch {history['best_epoch']} with accuracy {history['best_accuracy']:.4f}")
    print(f"Results saved to {run_dir}")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Train PES Multi-Task Classification Model')
    parser.add_argument('--data_dir', type=str, 
                        default='/data15/data15_5/yujun26/BA_PROJECT/红白美学标注',
                        help='Data directory path')
    parser.add_argument('--label_file', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/两次重新标注后校正.xlsx',
                        help='Label file path')
    parser.add_argument('--output_dir', type=str,
                        default='/data15/data15_5/yujun26/BA_PROJECT/outputs',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 检查设备 / Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    train(
        data_dir=args.data_dir,
        label_file=args.label_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device
    )


if __name__ == '__main__':
    main()
