"""
test_model.py - 模型模块最小可验证测试
Model module sanity check script
"""

import sys
sys.path.insert(0, '/data15/data15_5/yujun26/BA_PROJECT')

import torch


def test_model():
    """测试模型模块 / Test model module"""
    
    print("="*60)
    print("Testing Model Module")
    print("="*60)
    
    # 检查设备 / Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 1. 测试模型创建 / Test model creation
    print("\n[1] Testing create_model...")
    try:
        from model import create_model, PES_TASK_NAMES
        
        model, preprocess = create_model(device=device)
        print("    ✓ Model created successfully")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 测试参数统计 / Test parameter counting
    print("\n[2] Testing parameter count...")
    try:
        params = model.count_parameters()
        print(f"    Total parameters: {params['total']:,}")
        print(f"    Trainable parameters: {params['trainable']:,}")
        print(f"    Frozen parameters: {params['frozen']:,}")
        print("    ✓ Parameter count successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 3. 测试前向传播 / Test forward pass
    print("\n[3] Testing forward pass...")
    try:
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_input, dummy_input, dummy_input)
        
        print(f"    Output tasks: {list(outputs.keys())}")
        for task_name, logits in outputs.items():
            assert logits.shape == (batch_size, 3), f"Wrong output shape for {task_name}"
            print(f"    {task_name}: shape={logits.shape}")
        
        print("    ✓ Forward pass successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试预处理函数 / Test preprocessing function
    print("\n[4] Testing preprocessing function...")
    try:
        from PIL import Image
        import numpy as np
        
        # 创建测试图像 / Create test image
        test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        tensor = preprocess(test_img)
        print(f"    Input image size: 256x256")
        print(f"    Output tensor shape: {tensor.shape}")
        assert tensor.shape == (3, 224, 224), "Wrong preprocessed tensor shape"
        print("    ✓ Preprocessing successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 5. 测试梯度流 / Test gradient flow
    print("\n[5] Testing gradient flow...")
    try:
        model.train()
        dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True).to(device)
        
        outputs = model(dummy_input, dummy_input, dummy_input)
        loss = sum(out.mean() for out in outputs.values())
        loss.backward()
        
        # 检查是否有梯度 / Check if gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found"
        print("    ✓ Gradient flow successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("Model Module Test: PASSED ✓")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
