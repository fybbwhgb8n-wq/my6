#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation Script for RCBANet - IRSTD-1K Dataset
This script evaluates the trained model and saves visualization results.

Usage:
    python validate.py

Output:
    - Performance metrics printed to console
    - results/masks/          - Predicted mask images
    - results/comparisons/    - Input | Ground Truth | Prediction comparison images
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import sys
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for saving figures

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgl_rpcanet.models.rcbanet import RCBANet, RCBANetConfig
from cgl_rpcanet.data import ISTDDataset, get_dataloader
from cgl_rpcanet.utils.checkpoint import load_checkpoint


# ==================== Configuration ====================
# Modify these paths according to your setup

# Checkpoint path
CHECKPOINT_PATH = "./checkpoints/irstd_best_model.pth"

# Dataset configuration
DATA_ROOT = "/home/gsx/pycahrm_projects/LRCBA-5/datasets"  # Dataset root path
DATASET_NAME = "IRSTD-1k"  # Dataset name
IMAGE_SIZE = (256, 256)    # Image size (H, W)

# Output directories
OUTPUT_DIR = "./results"
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
COMPARISONS_DIR = os.path.join(OUTPUT_DIR, "comparisons")

# Model configuration (must match train.py configuration exactly)
# C1 = 32                      # Full resolution (1/1)
# C2 = 48                      # Half resolution (1/2)
# C3 = 64                      # Quarter resolution (1/4)
C1 = 32                      # Full resolution (1/1)
C2 = 48                      # Half resolution (1/2)
C3 = 64                      # Quarter resolution (1/4)
CAU_CH_S1 = 24               # CAU channels at scale 1 (1/1)
CAU_CH_S2 = 16               # CAU channels at scale 2 (1/2)
CAU_CH_S3 = 16               # CAU channels at scale 3 (1/4)
CAU_KERNELS = [3, 9]         # Multi-scale kernels for CAU
CAU_STRIDES = [8, 8, 8]      # Per-scale strides: [s3, s2, s1]
BASE_CH_S1 = 32              # Base channels at scale 1
BASE_CH_S2 = 32              # Base channels at scale 2
BASE_CH_S3 = 24              # Base channels at scale 3
MID_CH_S1 = 24               # Bottleneck channels at scale 1 (from train.py)
MID_CH_S2 = 24               # Bottleneck channels at scale 2 (from train.py)
MID_CH_S3 = 16               # Bottleneck channels at scale 3 (from train.py)
PROX_DEPTH = 2               # Depth of ProxNetB
GRAD_DEPTH = 2               # Depth of GradNet
DEEP_SUPERVISION = True      # Enable deep supervision
DEEP_SUP_WEIGHT = 0.15        # Weight for deep supervision losses

# Other configuration
BATCH_SIZE = 1  # Use batch size 1 for individual image saving
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================


# ==================== Evaluation Metrics ====================

def batch_pix_accuracy(output, target):
    """Calculate pixel accuracy"""
    if len(target.shape) == 3:
        target = target.unsqueeze(dim=0).float()
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0.5).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    """Calculate intersection over union"""
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0.5).float()
    if len(target.shape) == 3:
        target = target.unsqueeze(dim=0).float()
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class mIoU():
    """Calculate mIoU, pixAcc and F1 score"""

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct.cpu().numpy()
        self.total_label += labeled.cpu().numpy()
        self.total_inter += inter
        self.total_union += union

        # Calculate TP, FP, FN for F1
        pred_mask = (preds > 0.5).float()
        target_mask = labels.float()
        self.total_tp += (pred_mask * target_mask).sum().cpu().numpy()
        self.total_fp += (pred_mask * (1 - target_mask)).sum().cpu().numpy()
        self.total_fn += ((1 - pred_mask) * target_mask).sum().cpu().numpy()

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU_val = IoU.mean()

        # Calculate F1
        precision = self.total_tp / (self.total_tp + self.total_fp + np.spacing(1))
        recall = self.total_tp / (self.total_tp + self.total_fn + np.spacing(1))
        f1_score = 2.0 * precision * recall / (precision + recall + np.spacing(1))

        return float(pixAcc), float(mIoU_val), float(f1_score), float(precision), float(recall)

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0


class PD_FA():
    """Calculate Probability of Detection (PD) and False Alarm rate (FA)"""

    def __init__(self):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)
                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / (self.target + 1e-10)
        return float(Final_PD), float(Final_FA)

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0


# ==================== Visualization Functions ====================

def save_mask(pred_mask, save_path):
    """
    Save predicted mask as a grayscale image.
    
    Args:
        pred_mask: Numpy array [H, W] in [0, 1]
        save_path: Path to save the image
    """
    # Convert to 0-255 range
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode='L')
    img.save(save_path)


def save_comparison(input_img, gt_mask, pred_mask, save_path, img_name):
    """
    Save side-by-side comparison: Input | Ground Truth | Prediction
    
    Args:
        input_img: Numpy array [H, W] in [0, 1]
        gt_mask: Numpy array [H, W] in {0, 1}
        pred_mask: Numpy array [H, W] in [0, 1]
        save_path: Path to save the comparison image
        img_name: Image filename for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f'{img_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def save_overlay_comparison(input_img, gt_mask, pred_mask, save_path, img_name):
    """
    Save overlay comparison: Input with GT overlay | Input with Pred overlay | Difference
    
    Args:
        input_img: Numpy array [H, W] in [0, 1]
        gt_mask: Numpy array [H, W] in {0, 1}
        pred_mask: Numpy array [H, W] in [0, 1]
        save_path: Path to save the comparison image
        img_name: Image filename for title
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Convert to RGB for overlay
    input_rgb = np.stack([input_img] * 3, axis=-1)
    
    # Input image
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth overlay (green)
    gt_overlay = input_rgb.copy()
    gt_overlay[gt_mask > 0.5, 1] = np.clip(gt_overlay[gt_mask > 0.5, 1] + 0.5, 0, 1)
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth (Green)', fontsize=14)
    axes[1].axis('off')
    
    # Prediction overlay (red)
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    pred_overlay = input_rgb.copy()
    pred_overlay[pred_binary > 0.5, 0] = np.clip(pred_overlay[pred_binary > 0.5, 0] + 0.5, 0, 1)
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction (Red)', fontsize=14)
    axes[2].axis('off')
    
    # Difference: TP (white), FP (red), FN (blue)
    diff_img = np.zeros((input_img.shape[0], input_img.shape[1], 3))
    tp = (pred_binary > 0.5) & (gt_mask > 0.5)
    fp = (pred_binary > 0.5) & (gt_mask <= 0.5)
    fn = (pred_binary <= 0.5) & (gt_mask > 0.5)
    
    diff_img[tp] = [1, 1, 1]  # White for TP
    diff_img[fp] = [1, 0, 0]  # Red for FP
    diff_img[fn] = [0, 0, 1]  # Blue for FN
    
    axes[3].imshow(diff_img)
    axes[3].set_title('Diff: TP(white) FP(red) FN(blue)', fontsize=14)
    axes[3].axis('off')
    
    plt.suptitle(f'{img_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ==================== Main Validation Function ====================

@torch.no_grad()
def validate():
    """Main validation function"""
    print("\n" + "=" * 70)
    print("RCBANet Validation - IRSTD-1K Dataset")
    print("=" * 70)
    
    # Check device
    device = DEVICE
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(COMPARISONS_DIR, exist_ok=True)
    print(f"\nOutput directories:")
    print(f"  Masks: {MASKS_DIR}")
    print(f"  Comparisons: {COMPARISONS_DIR}")
    
    # Create model with same configuration as training
    print("\nCreating model...")
    cfg = RCBANetConfig(
        C1=C1,
        C2=C2,
        C3=C3,
        cau_ch_s1=CAU_CH_S1,
        cau_ch_s2=CAU_CH_S2,
        cau_ch_s3=CAU_CH_S3,
        cau_kernels=CAU_KERNELS,
        cau_strides=CAU_STRIDES,
        base_ch_s1=BASE_CH_S1,
        base_ch_s2=BASE_CH_S2,
        base_ch_s3=BASE_CH_S3,
        mid_ch_s1=MID_CH_S1,
        mid_ch_s2=MID_CH_S2,
        mid_ch_s3=MID_CH_S3,
        prox_depth=PROX_DEPTH,
        grad_depth=GRAD_DEPTH,
        use_gradient_checkpointing=False,  # Not needed for inference
        deep_supervision=DEEP_SUPERVISION,
        deep_sup_weight=DEEP_SUP_WEIGHT
    )
    
    model = RCBANet(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return
    
    checkpoint_info = load_checkpoint(model, CHECKPOINT_PATH, device=device)
    print(f"Checkpoint epoch: {checkpoint_info['epoch']}")
    if checkpoint_info['metrics']:
        print(f"Checkpoint metrics: {checkpoint_info['metrics']}")
    
    model.eval()
    
    # Load test dataset
    print("\nLoading test dataset...")
    dataset_path = Path(DATA_ROOT) / DATASET_NAME
    test_root = dataset_path / 'test'
    
    if not test_root.exists():
        print(f"❌ Error: Test dataset not found at {test_root}")
        print(f"Please ensure the dataset structure is correct:")
        print(f"  {dataset_path}/test/images/")
        print(f"  {dataset_path}/test/masks/")
        return
    
    test_dataset = ISTDDataset(
        root=str(test_root),
        split='val',  # No augmentation
        image_size=IMAGE_SIZE,
        use_augmentation=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize metrics
    eval_miou = mIoU()
    eval_pd_fa = PD_FA()
    
    # Validation loop
    print("\n" + "-" * 70)
    print("Running validation...")
    print("-" * 70)
    
    for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Validating")):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        # Forward pass
        outputs = model(images)
        pred_mask = outputs["mask"]  # [B, 1, H, W]
        
        # Update metrics
        eval_miou.update((pred_mask > 0.5), masks)
        
        # Update PD_FA (per sample)
        for i in range(pred_mask.shape[0]):
            size = (masks.shape[2], masks.shape[3])
            eval_pd_fa.update(
                (pred_mask[i, 0, :, :] > 0.5),
                masks[i, 0, :, :],
                size
            )
        
        # Save visualizations
        for i in range(images.shape[0]):
            img_idx = batch_idx * BATCH_SIZE + i
            img_name = test_dataset.image_names[img_idx]
            base_name = Path(img_name).stem
            
            # Convert to numpy
            input_np = images[i, 0].cpu().numpy()  # [H, W]
            gt_np = masks[i, 0].cpu().numpy()       # [H, W]
            pred_np = pred_mask[i, 0].cpu().numpy() # [H, W]
            
            # Save mask
            mask_save_path = os.path.join(MASKS_DIR, f"{base_name}.png")
            save_mask((pred_np > 0.5).astype(np.float32), mask_save_path)
            
            # Save comparison
            comparison_save_path = os.path.join(COMPARISONS_DIR, f"{base_name}_comparison.png")
            save_overlay_comparison(input_np, gt_np, pred_np, comparison_save_path, img_name)
    
    # Get final metrics
    pixAcc, mIoU_val, F1, precision, recall = eval_miou.get()
    PD, FA = eval_pd_fa.get()
    
    # Print results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    print(f"\n📊 Performance Metrics:")
    print(f"  ├── Pixel Accuracy:   {pixAcc * 100:.4f}%")
    print(f"  ├── mIoU:             {mIoU_val * 100:.4f}%")
    print(f"  ├── F1 Score:         {F1 * 100:.4f}%")
    print(f"  ├── Precision:        {precision * 100:.4f}%")
    print(f"  ├── Recall:           {recall * 100:.4f}%")
    print(f"  ├── PD (Prob. Det.):  {PD * 100:.4f}%")
    print(f"  └── FA (False Alarm): {FA * 1e6:.4f} × 10⁻⁶")
    
    print(f"\n📁 Results saved to:")
    print(f"  ├── Masks:       {MASKS_DIR} ({len(test_dataset)} images)")
    print(f"  └── Comparisons: {COMPARISONS_DIR} ({len(test_dataset)} images)")
    
    print("\n" + "=" * 70)
    print("✅ Validation completed!")
    print("=" * 70 + "\n")
    
    # Save metrics to file
    metrics_file = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("RCBANet Validation Results - IRSTD-1K Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Checkpoint Epoch: {checkpoint_info['epoch']}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Pixel Accuracy:   {pixAcc * 100:.4f}%\n")
        f.write(f"  mIoU:             {mIoU_val * 100:.4f}%\n")
        f.write(f"  F1 Score:         {F1 * 100:.4f}%\n")
        f.write(f"  Precision:        {precision * 100:.4f}%\n")
        f.write(f"  Recall:           {recall * 100:.4f}%\n")
        f.write(f"  PD:               {PD * 100:.4f}%\n")
        f.write(f"  FA:               {FA * 1e6:.4f} × 10^-6\n")
    
    print(f"📄 Metrics saved to: {metrics_file}")
    
    return {
        "pixAcc": pixAcc,
        "mIoU": mIoU_val,
        "F1": F1,
        "precision": precision,
        "recall": recall,
        "PD": PD,
        "FA": FA
    }


if __name__ == "__main__":
    validate()
