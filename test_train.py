"""
test_train.py - 训练流程最小可验证测试 (1 epoch sanity check)
Training pipeline sanity check script
"""

import sys
sys.path.insert(0, '/data15/data15_5/yujun26/BA_PROJECT')

import os
import torch
import torch.nn as nn
from torch.optim import AdamW


def test_train():
    """测试训练流程 / Test training pipeline"""
    
    print("="*60)
    print("Testing Training Pipeline (1 epoch sanity check)")
    print("="*60)
    
    # 检查设备 / Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    DATA_DIR = '/data15/data15_5/yujun26/BA_PROJECT/红白美学标注'
    LABEL_FILE = '/data15/data15_5/yujun26/BA_PROJECT/两次重新标注后校正.xlsx'
    
    # 1. 创建模型 / Create model
    print("\n[1] Creating model...")
    try:
        from model import create_model, PES_TASK_NAMES
        model, preprocess = create_model(device=device)
        print("    ✓ Model created")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 创建数据加载器 / Create data loaders
    print("\n[2] Creating data loaders...")
    try:
        from dataset import create_dataloaders
        
        train_loader, val_loader, class_weights = create_dataloaders(
            data_dir=DATA_DIR,
            label_file=LABEL_FILE,
            preprocess=preprocess,
            batch_size=4,
            num_workers=0  # 使用0以便调试
        )
        print(f"    ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 创建损失函数和优化器 / Create loss and optimizer
    print("\n[3] Creating loss functions and optimizer...")
    try:
        criterion_dict = {}
        for task_name in PES_TASK_NAMES:
            weights = class_weights[task_name].to(device)
            criterion_dict[task_name] = nn.CrossEntropyLoss(weight=weights)
            print(f"    {task_name}: weights={weights.cpu().numpy().round(3)}")
        
        optimizer = AdamW(model.get_trainable_params(), lr=1e-4)
        print("    ✓ Loss and optimizer created")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 4. 训练1个batch / Train 1 batch
    print("\n[4] Training 1 batch...")
    try:
        model.train()
        batch = next(iter(train_loader))
        implant, control, global_view, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        global_view = global_view.to(device)
        labels = labels.to(device)
        
        print(f"    Input shapes: implant={implant.shape}, control={control.shape}, global={global_view.shape}")
        print(f"    Labels shape: {labels.shape}")
        
        optimizer.zero_grad()
        outputs = model(implant, control, global_view)
        
        total_loss = 0
        for i, task_name in enumerate(PES_TASK_NAMES):
            task_labels = labels[:, i]
            task_logits = outputs[task_name]
            loss = criterion_dict[task_name](task_logits, task_labels)
            total_loss += loss
            print(f"    {task_name}: loss={loss.item():.4f}")
        
        total_loss.backward()
        optimizer.step()
        
        print(f"    Total loss: {total_loss.item():.4f}")
        print("    ✓ Training step successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 验证1个batch / Validate 1 batch
    print("\n[5] Validating 1 batch...")
    try:
        model.eval()
        batch = next(iter(val_loader))
        implant, control, global_view, labels = batch
        implant = implant.to(device)
        control = control.to(device)
        global_view = global_view.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(implant, control, global_view)
        
        # 计算准确率 / Calculate accuracy
        correct = 0
        total = 0
        for i, task_name in enumerate(PES_TASK_NAMES):
            preds = outputs[task_name].argmax(dim=-1)
            task_labels = labels[:, i]
            correct += (preds == task_labels).sum().item()
            total += len(task_labels)
        
        accuracy = correct / total
        print(f"    Batch accuracy: {accuracy:.4f}")
        print("    ✓ Validation step successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 6. 测试模型保存/加载 / Test model save/load
    print("\n[6] Testing model save/load...")
    try:
        test_path = '/data15/data15_5/yujun26/BA_PROJECT/test_model_temp.pth'
        model.save_model(test_path)
        
        # 创建新模型并加载 / Create new model and load
        model2, _ = create_model(device=device)
        model2.load_model(test_path)
        
        # 清理 / Cleanup
        os.remove(test_path)
        print("    ✓ Model save/load successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("Training Pipeline Test: PASSED ✓")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_train()
    sys.exit(0 if success else 1)
