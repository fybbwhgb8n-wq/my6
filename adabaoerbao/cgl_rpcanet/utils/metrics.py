"""
Evaluation metrics for ISTD.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU).
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [1, H, W] in [0, 1]
        target: Ground truth mask [B, 1, H, W] or [1, H, W] in {0, 1}
        threshold: Threshold for binarizing prediction (default: 0.5)
    
    Returns:
        IoU score (scalar float)
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient (F1 score).
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [1, H, W] in [0, 1]
        target: Ground truth mask [B, 1, H, W] or [1, H, W] in {0, 1}
        threshold: Threshold for binarizing prediction (default: 0.5)
    
    Returns:
        Dice score (scalar float)
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    
    intersection = (pred_bin * target_bin).sum()
    denominator = pred_bin.sum() + target_bin.sum()
    
    if denominator == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection / denominator).item()


def compute_pixel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pixel-level precision, recall, F1.
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [1, H, W] in [0, 1]
        target: Ground truth mask [B, 1, H, W] or [1, H, W] in {0, 1}
        threshold: Threshold for binarizing prediction (default: 0.5)
    
    Returns:
        Dictionary with precision, recall, f1
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all metrics: IoU, Dice, precision, recall, F1.
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [1, H, W] in [0, 1]
        target: Ground truth mask [B, 1, H, W] or [1, H, W] in {0, 1}
        threshold: Threshold for binarizing prediction (default: 0.5)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "iou": compute_iou(pred, target, threshold),
        "dice": compute_dice(pred, target, threshold)
    }
    
    pixel_metrics = compute_pixel_metrics(pred, target, threshold)
    metrics.update(pixel_metrics)
    
    return metrics

