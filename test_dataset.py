"""
test_dataset.py - 数据模块最小可验证测试
Dataset module sanity check script
"""

import sys
sys.path.insert(0, '/data15/data15_5/yujun26/BA_PROJECT')

from dataset import (
    build_data_items,
    patient_group_split,
    parse_labelme_json,
    compute_class_weights,
    PES_COLUMNS
)
import numpy as np


def test_dataset():
    """测试数据模块 / Test dataset module"""
    
    print("="*60)
    print("Testing Dataset Module")
    print("="*60)
    
    DATA_DIR = '/data15/data15_5/yujun26/BA_PROJECT/红白美学标注'
    LABEL_FILE = '/data15/data15_5/yujun26/BA_PROJECT/两次重新标注后校正.xlsx'
    
    # 1. 测试数据项构建 / Test data item building
    print("\n[1] Testing build_data_items...")
    try:
        data_items = build_data_items(DATA_DIR, LABEL_FILE)
        print(f"    ✓ Built {len(data_items)} data items")
        assert len(data_items) > 0, "No data items built"
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 2. 验证数据项结构 / Verify data item structure
    print("\n[2] Verifying data item structure...")
    try:
        item = data_items[0]
        required_keys = ['image_path', 'json_path', 'labels', 'patient']
        for key in required_keys:
            assert key in item, f"Missing key: {key}"
        print(f"    ✓ Data item has all required keys: {required_keys}")
        print(f"    Example: patient={item['patient']}, labels={item['labels']}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 3. 测试患者级别划分 / Test patient-level split
    print("\n[3] Testing patient_group_split...")
    try:
        train_items, val_items = patient_group_split(data_items)
        print(f"    ✓ Train: {len(train_items)}, Val: {len(val_items)}")
        
        # 验证患者不重叠 / Verify no patient overlap
        train_patients = set(item['patient'] for item in train_items)
        val_patients = set(item['patient'] for item in val_items)
        overlap = train_patients & val_patients
        assert len(overlap) == 0, f"Patient overlap found: {overlap}"
        print(f"    ✓ No patient overlap between train/val")
        print(f"    Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 4. 测试JSON解析 / Test JSON parsing
    print("\n[4] Testing parse_labelme_json...")
    try:
        rois = parse_labelme_json(data_items[0]['json_path'])
        print(f"    ✓ Parsed ROIs: {list(rois.keys())}")
        for roi_name, bbox in rois.items():
            print(f"      {roi_name}: {bbox}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    # 5. 测试类别权重计算 / Test class weight computation
    print("\n[5] Testing compute_class_weights...")
    try:
        labels = np.array([item['labels'] for item in train_items])
        print(f"    Labels shape: {labels.shape}")
        
        for i, col in enumerate(PES_COLUMNS):
            weights = compute_class_weights(labels[:, i])
            print(f"    {col}: weights={weights.numpy().round(4)}")
        print("    ✓ Class weights computed successfully")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("Dataset Module Test: PASSED ✓")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_dataset()
    sys.exit(0 if success else 1)
