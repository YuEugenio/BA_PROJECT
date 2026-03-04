"""
Two-local input mode: implant ROI + control ROI.
Used by Baseline experiments and some legacy experiments without global view.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple


class TwoLocalDataset(Dataset):
    """Dataset returning (implant_tensor, control_tensor, labels_tensor)."""

    def __init__(self, data_items: List[Dict], preprocess, roi_parser):
        self.data_items = data_items
        self.preprocess = preprocess
        self.roi_parser = roi_parser

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data_items[idx]
        image = Image.open(item['image_path']).convert('RGB')
        rois = self.roi_parser(item['json_path'])

        tensors = {}
        for roi_name in ['implant', 'control']:
            if roi_name in rois:
                roi_img = _crop_roi(image, rois[roi_name])
            else:
                roi_img = image  # fallback
            tensors[roi_name] = self.preprocess(roi_img)

        labels = torch.LongTensor(item['labels'])
        return tensors['implant'], tensors['control'], labels


def _crop_roi(image: Image.Image, bbox: List[float]) -> Image.Image:
    x1, y1, x2, y2 = [int(c) for c in bbox]
    w, h = image.size
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    return image.crop((x1, y1, x2, y2))
