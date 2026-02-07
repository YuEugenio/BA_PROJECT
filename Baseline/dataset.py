"""
dataset.py - PES数据加载与预处理模块
Dataset module for PES (Pink Esthetic Score) multi-task classification

功能 / Features:
1. 患者级别数据划分 (Patient-level train/val split)
2. LabelMe JSON解析与ROI裁剪 (ROI extraction from LabelMe annotations)
3. BioMedCLIP官方预处理 (Official BioMedCLIP preprocessing)
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


# 4个PES评估子项列名
# Column names for 4 PES evaluation sub-items
PES_COLUMNS = ['近中牙龈乳头', '远中牙龈乳头', '软组织形态', '粘膜颜色']

# ROI标签映射 (LabelMe中的标签名 → 标准名称)
# ROI label mapping (LabelMe label names → standard names)
ROI_LABELS = {
    '种植牙': 'implant',    # Implant region
    '对侧牙': 'control',    # Contralateral tooth (control)
}


def get_patient_name(folder_name: str) -> str:
    """
    从文件夹名称提取患者姓名（用于患者级别分组）
    Extract patient name from folder name (for patient-level grouping)
    
    例如 / Example: "何宁 11 （4）" → "何宁"
    """
    # 文件夹名格式通常是 "姓名 牙位 （序号）"
    # Folder name format is typically "Name ToothPosition (Index)"
    parts = folder_name.strip().split()
    if len(parts) >= 1:
        return parts[0]  # 返回第一部分作为患者姓名
    return folder_name


def parse_labelme_json(json_path: str) -> Dict[str, List[float]]:
    """
    解析LabelMe JSON文件，提取ROI矩形坐标
    Parse LabelMe JSON file to extract ROI rectangle coordinates
    
    Args:
        json_path: LabelMe JSON文件路径
        
    Returns:
        Dict[roi_name, [x1, y1, x2, y2]] - ROI名称到坐标的映射
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rois = {}
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])
        
        if shape_type == 'rectangle' and len(points) == 2 and label in ROI_LABELS:
            # 矩形由两个对角点定义
            # Rectangle defined by two corner points
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 确保坐标顺序正确 (左上到右下)
            # Ensure correct coordinate order (top-left to bottom-right)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            rois[ROI_LABELS[label]] = [x1, y1, x2, y2]
    
    return rois


def crop_roi(image: Image.Image, bbox: List[float]) -> Image.Image:
    """
    从图像中裁剪ROI区域
    Crop ROI region from image
    
    Args:
        image: PIL图像
        bbox: [x1, y1, x2, y2] 边界框坐标
        
    Returns:
        裁剪后的PIL图像
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    # 确保坐标在图像范围内
    # Ensure coordinates are within image bounds
    w, h = image.size
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    return image.crop((x1, y1, x2, y2))


def patient_group_split(
    data_items: List[Dict],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    患者级别的数据划分，确保同一患者的所有图像在同一集合中
    Patient-level data split, ensuring all images from same patient are in same set
    
    Args:
        data_items: 数据项列表，每项包含 'patient' 键
        test_size: 验证集比例
        random_state: 随机种子
        
    Returns:
        (train_items, val_items)
    """
    if len(data_items) == 0:
        return [], []
    
    # 提取患者分组信息
    # Extract patient grouping information
    groups = [item['patient'] for item in data_items]
    indices = np.arange(len(data_items))
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    
    train_items = [data_items[i] for i in train_idx]
    val_items = [data_items[i] for i in val_idx]
    
    return train_items, val_items


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    计算类别权重用于加权交叉熵损失
    Compute class weights for weighted cross-entropy loss
    
    Args:
        labels: shape (N,) 的标签数组
        
    Returns:
        torch.Tensor: 各类别权重
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    n_classes = len(unique)
    weights = total / (n_classes * counts)
    
    # 创建完整的权重数组 (确保3类都有权重)
    full_weights = np.ones(3)
    for cls, w in zip(unique, weights):
        full_weights[int(cls)] = w
    
    return torch.FloatTensor(full_weights)


def build_train_preprocess(
    preprocess,
    enable_augment: bool = True,
    flip_prob: float = 0.5,
    jitter_strength: float = 0.05,
    rotation_deg: float = 8.0,
    blur_prob: float = 0.1
):
    """
    基于BioMedCLIP预处理创建训练时的数据增强
    Build train-time augmentations on top of BioMedCLIP preprocessing
    """
    if not enable_augment:
        return preprocess

    aug = T.Compose([
        T.RandomHorizontalFlip(p=flip_prob),
        T.ColorJitter(
            brightness=jitter_strength,
            contrast=jitter_strength,
            saturation=jitter_strength
        ),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=blur_prob
        ),
        T.RandomRotation(degrees=rotation_deg, fill=(0, 0, 0))
    ])

    def _aug_preprocess(img: Image.Image):
        return preprocess(aug(img))

    return _aug_preprocess


class PESDataset(Dataset):
    """
    PES多任务分类数据集
    PES Multi-task Classification Dataset
    
    每个样本返回:
    - 两路ROI图像 (implant, control)
    - 4个PES子任务标签
    """
    
    def __init__(
        self,
        data_items: List[Dict],
        preprocess,
        data_root: str
    ):
        """
        Args:
            data_items: 数据项列表，每项包含 'image_path', 'json_path', 'labels', 'patient'
            preprocess: BioMedCLIP预处理函数
            data_root: 数据根目录
        """
        self.data_items = data_items
        self.preprocess = preprocess
        self.data_root = data_root
        
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回两路ROI张量和4个标签
        Return two ROI tensors and 4 labels
        
        Returns:
            (implant_tensor, control_tensor, labels_tensor)
        """
        item = self.data_items[idx]
        
        # 加载原始图像 / Load original image
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # 解析JSON获取ROI坐标 / Parse JSON to get ROI coordinates
        json_path = item['json_path']
        rois = parse_labelme_json(json_path)
        
        # 裁剪两个ROI区域 / Crop two ROI regions
        roi_tensors = {}
        for roi_name in ['implant', 'control']:
            if roi_name in rois:
                roi_image = crop_roi(image, rois[roi_name])
            else:
                # 如果ROI不存在，使用整张图像作为fallback
                # If ROI doesn't exist, use full image as fallback
                roi_image = image
            
            # 应用BioMedCLIP预处理 / Apply BioMedCLIP preprocessing
            roi_tensor = self.preprocess(roi_image)
            roi_tensors[roi_name] = roi_tensor
        
        # 获取标签 / Get labels
        labels = torch.LongTensor(item['labels'])
        
        return (
            roi_tensors['implant'],
            roi_tensors['control'],
            labels
        )


def build_data_items(
    data_dir: str,
    label_file: str
) -> List[Dict]:
    """
    构建数据项列表
    Build list of data items
    
    Args:
        data_dir: 包含患者文件夹的数据目录
        label_file: 标签Excel文件路径
        
    Returns:
        数据项列表 [{'image_path', 'json_path', 'labels', 'patient'}, ...]
    """
    # 读取标签文件 / Read label file
    df = pd.read_excel(label_file)
    
    # 创建图像ID到标签的映射 / Create image ID to labels mapping
    # 图像列包含不带扩展名的文件名
    label_map = {}
    for _, row in df.iterrows():
        image_id = str(row['图像']).strip()
        labels = [int(row[col]) for col in PES_COLUMNS]
        label_map[image_id] = labels
    
    data_items = []
    
    # 遍历患者文件夹 / Traverse patient folders
    for patient_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(folder_path):
            continue
        
        patient_name = get_patient_name(patient_folder)
        
        # 查找图像和对应的JSON文件 / Find images and corresponding JSON files
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_id = os.path.splitext(filename)[0]
                json_filename = image_id + '.json'
                json_path = os.path.join(folder_path, json_filename)
                
                # 检查JSON文件和标签是否存在
                # Check if JSON file and labels exist
                if os.path.exists(json_path) and image_id in label_map:
                    data_items.append({
                        'image_path': os.path.join(folder_path, filename),
                        'json_path': json_path,
                        'labels': label_map[image_id],
                        'patient': patient_name
                    })
    
    return data_items


def compute_label_distribution(data_items: List[Dict]) -> Dict[str, Dict[int, int]]:
    """
    统计每个任务的类别分布
    Compute class distribution per task
    """
    dist = {col: {0: 0, 1: 0, 2: 0} for col in PES_COLUMNS}
    for item in data_items:
        labels = item['labels']
        for i, col in enumerate(PES_COLUMNS):
            dist[col][int(labels[i])] += 1
    return dist


def summarize_dataset(data_dir: str, label_file: str) -> Dict:
    """
    汇总数据集情况（图像/标签/ROI缺失等）
    Summarize dataset stats (images, labels, ROI missing)
    """
    image_ids = []
    patient_stats = {}
    for patient_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(folder_path):
            continue
        patient_name = get_patient_name(patient_folder)
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_ids.append(os.path.splitext(filename)[0])
                count += 1
        if count > 0:
            patient_stats[patient_name] = patient_stats.get(patient_name, 0) + count

    df = pd.read_excel(label_file)
    label_ids = [str(x).strip() for x in df['图像'].tolist()]

    image_set = set(image_ids)
    label_set = set(label_ids)
    images_with_labels = image_set & label_set
    labels_without_images = label_set - image_set
    images_without_labels = image_set - label_set

    data_items = build_data_items(data_dir, label_file)

    roi_missing = {roi: 0 for roi in ROI_LABELS.values()}
    for item in data_items:
        rois = parse_labelme_json(item['json_path'])
        for roi in ROI_LABELS.values():
            if roi not in rois:
                roi_missing[roi] += 1

    summary = {
        'total_images': len(image_ids),
        'total_labels': len(label_ids),
        'images_with_labels': len(images_with_labels),
        'labels_without_images': len(labels_without_images),
        'images_without_labels': len(images_without_labels),
        'data_items': len(data_items),
        'roi_missing_counts': roi_missing,
        'roi_missing_ratio': {
            roi: (roi_missing[roi] / len(data_items)) if data_items else 0.0
            for roi in roi_missing
        },
        'patient_count': len(patient_stats),
        'patient_image_stats': {
            'min': min(patient_stats.values()) if patient_stats else 0,
            'max': max(patient_stats.values()) if patient_stats else 0,
            'mean': (sum(patient_stats.values()) / len(patient_stats)) if patient_stats else 0
        },
        'label_distribution': compute_label_distribution(data_items)
    }
    return summary


def create_dataloaders(
    data_dir: str,
    label_file: str,
    preprocess,
    batch_size: int = 8,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    train_augment: bool = True,
    use_weighted_sampler: bool = False,
    flip_prob: float = 0.5,
    jitter_strength: float = 0.05,
    rotation_deg: float = 8.0,
    blur_prob: float = 0.1
) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    创建训练和验证数据加载器
    Create train and validation data loaders
    
    Args:
        data_dir: 数据目录
        label_file: 标签文件路径
        preprocess: BioMedCLIP预处理函数
        batch_size: 批次大小
        num_workers: 数据加载进程数
        test_size: 验证集比例
        random_state: 随机种子
        
    Returns:
        (train_loader, val_loader, class_weights_dict)
    """
    # 构建数据项 / Build data items
    data_items = build_data_items(data_dir, label_file)
    print(f"Total samples: {len(data_items)}")
    
    # 患者级别划分 / Patient-level split
    train_items, val_items = patient_group_split(data_items, test_size, random_state)
    print(f"Train samples: {len(train_items)}, Val samples: {len(val_items)}")
    
    # 计算类别权重（仅使用训练集）/ Compute class weights (training set only)
    train_labels = np.array([item['labels'] for item in train_items])
    class_weights = {}
    for i, col in enumerate(PES_COLUMNS):
        class_weights[col] = compute_class_weights(train_labels[:, i])
    
    # 创建数据集 / Create datasets
    train_preprocess = build_train_preprocess(
        preprocess,
        enable_augment=train_augment,
        flip_prob=flip_prob,
        jitter_strength=jitter_strength,
        rotation_deg=rotation_deg,
        blur_prob=blur_prob
    )
    train_dataset = PESDataset(train_items, train_preprocess, data_dir)
    val_dataset = PESDataset(val_items, preprocess, data_dir)
    
    # 创建数据加载器 / Create data loaders
    if use_weighted_sampler:
        label_freq = {col: np.bincount(train_labels[:, i], minlength=3) for i, col in enumerate(PES_COLUMNS)}
        inv_freq = {col: 1.0 / np.maximum(label_freq[col], 1) for col in PES_COLUMNS}
        sample_weights = []
        for item in train_items:
            w = 0.0
            for i, col in enumerate(PES_COLUMNS):
                w += inv_freq[col][int(item['labels'][i])]
            sample_weights.append(w / len(PES_COLUMNS))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_weights
