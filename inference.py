"""
inference.py - PES模型推理模块
Inference module for PES Multi-Task Classification Model

功能 / Features:
1. 加载训练好的模型权重 / Load trained model weights
2. 单张或批量图像推理 / Single or batch image inference
3. 输出4个PES子项的预测 / Output predictions for 4 PES sub-items
"""

import os
import argparse
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image

from dataset import parse_labelme_json, crop_roi, ROI_LABELS
from model import create_model, PES_TASK_NAMES


# PES分数到描述的映射 / PES score to description mapping
PES_SCORE_DESC = {
    0: '较差/Poor (0分)',
    1: '中等/Fair (1分)',
    2: '良好/Good (2分)'
}


class PESPredictor:
    """
    PES预测器类
    PES Predictor class
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: 模型权重文件路径
            device: 设备
        """
        self.device = device
        
        # 创建模型并加载权重 / Create model and load weights
        print(f"Loading model from {model_path}...")
        self.model, self.preprocess = create_model(device=device)
        self.model.load_model(model_path)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict_single(
        self,
        image_path: str,
        json_path: Optional[str] = None
    ) -> Dict:
        """
        对单张图像进行推理
        Predict for a single image
        
        Args:
            image_path: 图像文件路径
            json_path: LabelMe JSON文件路径（可选，不提供则使用整图）
            
        Returns:
            预测结果字典
        """
        # 加载图像 / Load image
        image = Image.open(image_path).convert('RGB')
        
        # 获取ROI / Get ROIs
        if json_path and os.path.exists(json_path):
            rois = parse_labelme_json(json_path)
        else:
            # 无JSON时使用整图作为所有ROI
            # Use full image as all ROIs when no JSON
            rois = {}
        
        # 裁剪和预处理三个ROI / Crop and preprocess three ROIs
        roi_tensors = {}
        for roi_name in ['implant', 'control', 'global']:
            if roi_name in rois:
                roi_image = crop_roi(image, rois[roi_name])
            else:
                roi_image = image
            roi_tensor = self.preprocess(roi_image).unsqueeze(0).to(self.device)
            roi_tensors[roi_name] = roi_tensor
        
        # 推理 / Inference
        with torch.no_grad():
            outputs = self.model(
                roi_tensors['implant'],
                roi_tensors['control'],
                roi_tensors['global']
            )
        
        # 处理输出 / Process outputs
        results = {
            'image_path': image_path,
            'predictions': {}
        }
        
        for task_name in PES_TASK_NAMES:
            logits = outputs[task_name][0]  # [3]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_class = int(logits.argmax().item())
            
            results['predictions'][task_name] = {
                'class': pred_class,
                'description': PES_SCORE_DESC[pred_class],
                'probabilities': {
                    0: float(probs[0]),
                    1: float(probs[1]),
                    2: float(probs[2])
                },
                'confidence': float(probs[pred_class])
            }
        
        # 计算总PES分数 / Calculate total PES score
        total_score = sum(
            results['predictions'][name]['class'] 
            for name in PES_TASK_NAMES
        )
        results['total_pes_score'] = total_score
        results['max_possible_score'] = 8  # 4个任务 × 2分
        
        return results
    
    def predict_batch(
        self,
        image_paths: List[str],
        json_paths: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        批量图像推理
        Batch image inference
        
        Args:
            image_paths: 图像文件路径列表
            json_paths: JSON文件路径列表（可选）
            
        Returns:
            预测结果列表
        """
        if json_paths is None:
            json_paths = [None] * len(image_paths)
        
        results = []
        for img_path, json_path in zip(image_paths, json_paths):
            try:
                result = self.predict_single(img_path, json_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_folder(
        self,
        folder_path: str,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        对文件夹中的所有图像进行推理
        Predict for all images in a folder
        
        Args:
            folder_path: 文件夹路径
            output_file: 输出JSON文件路径（可选）
            
        Returns:
            预测结果列表
        """
        # 查找所有图像和对应的JSON / Find all images and corresponding JSONs
        image_paths = []
        json_paths = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                
                # 查找对应的JSON / Find corresponding JSON
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(folder_path, json_filename)
                json_paths.append(json_path if os.path.exists(json_path) else None)
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        
        # 批量推理 / Batch inference
        results = self.predict_batch(image_paths, json_paths)
        
        # 保存结果 / Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def print_prediction(result: Dict):
    """
    打印预测结果
    Print prediction result
    """
    print(f"\n{'='*60}")
    print(f"Image: {result.get('image_path', 'Unknown')}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nPES Predictions:")
    print(f"{'-'*60}")
    
    for task_name in PES_TASK_NAMES:
        pred = result['predictions'][task_name]
        print(f"  {task_name}:")
        print(f"    Score: {pred['class']} - {pred['description']}")
        print(f"    Confidence: {pred['confidence']:.2%}")
        probs = pred['probabilities']
        print(f"    Probabilities: 0:{probs[0]:.2%}, 1:{probs[1]:.2%}, 2:{probs[2]:.2%}")
    
    print(f"{'-'*60}")
    print(f"Total PES Score: {result['total_pes_score']}/{result['max_possible_score']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='PES Model Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--image', type=str,
                        help='Path to single image')
    parser.add_argument('--json', type=str,
                        help='Path to LabelMe JSON for single image')
    parser.add_argument('--folder', type=str,
                        help='Path to folder containing images')
    parser.add_argument('--output', type=str,
                        help='Output JSON file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 检查设备 / Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 创建预测器 / Create predictor
    predictor = PESPredictor(args.model, device=args.device)
    
    if args.image:
        # 单张图像推理 / Single image inference
        result = predictor.predict_single(args.image, args.json)
        print_prediction(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {args.output}")
            
    elif args.folder:
        # 文件夹批量推理 / Folder batch inference
        results = predictor.predict_folder(args.folder, args.output)
        for result in results:
            print_prediction(result)
    else:
        print("Please specify --image or --folder")


if __name__ == '__main__':
    main()
