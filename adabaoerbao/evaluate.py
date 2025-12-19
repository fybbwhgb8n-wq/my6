#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCBANet Evaluation Script for IRSTD-1K Dataset
Evaluates trained model and saves predicted masks and comparison images.

Usage:
    python evaluate.py

Outputs:
    - Performance metrics (mIoU, F1, pixAcc, PD, FA)
    - results/masks/          - Predicted mask images
    - results/comparisons/    - Input-GT-Prediction comparison images
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
import sys
from datetime import datetime
from skimage import measure

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgl_rpcanet.models.rcbanet import RCBANet, RCBANetConfig
from cgl_rpcanet.data import ISTDDataset, get_dataloader
from cgl_rpcanet.utils.checkpoint import load_checkpoint


# ==================== Configuration ====================
# Modify these paths according to your setup

# Model checkpoint path
CHECKPOINT_PATH = "./checkpoints/best_model.pth"

# Dataset configuration (IRSTD-1K)
DATA_ROOT = "/home/gsx/pycahrm_projects/LRCBA-5/datasets"
DATASET_NAME = "IRSTD-1k"
IMAGE_SIZE = (256, 256)

# Model configuration (must match training config)
# These values are inferred from the checkpoint
C1 = 32
C2 = 48
C3 = 64
CAU_CH_S1 = 24
CAU_CH_S2 = 16
CAU_CH_S3 = 16
CAU_KERNELS = [3, 5]
CAU_STRIDES = [2, 1, 1]  # Default strides
# Stage 3: mid_ch_s3=16, base_ch_s3=24
# Stage 2: mid_ch_s2=24, base_ch_s2=32
# Stage 1: mid_ch_s1=24, base_ch_s1=32
BASE_CH_S1 = 32
BASE_CH_S2 = 32
BASE_CH_S3 = 24
MID_CH_S1 = 24
MID_CH_S2 = 24
MID_CH_S3 = 16
PROX_DEPTH = 2
GRAD_DEPTH = 2

# Evaluation settings
BATCH_SIZE = 1  # Use batch size 1 for per-image saving
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5  # Binary threshold for mask

# Output directories
OUTPUT_DIR = "./results"
MASK_DIR = OUTPUT_DIR + "/masks"
COMPARISON_DIR = OUTPUT_DIR + "/comparisons"


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
    """Save predicted binary mask as image"""
    # Convert to numpy and scale to 0-255
    mask = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(mask, mode='L')
    img.save(save_path)


def save_comparison(input_img, gt_mask, pred_mask, save_path):
    """
    Save comparison image showing input, ground truth, and prediction side by side.
    
    Layout: [Input Image] [Ground Truth] [Prediction]
    
    Args:
        input_img: Input image tensor [1, H, W]
        gt_mask: Ground truth mask tensor [1, H, W]
        pred_mask: Predicted mask tensor [1, H, W]
        save_path: Path to save comparison image
    """
    H, W = input_img.shape[1], input_img.shape[2]
    
    # Convert tensors to numpy (0-255)
    input_np = (input_img.squeeze().cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred_np = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    # Create colored versions for better visualization
    # Input: grayscale
    input_rgb = np.stack([input_np, input_np, input_np], axis=-1)
    
    # Ground truth: green channel for targets
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    gt_rgb[:, :, 0] = input_np  # Red channel - background
    gt_rgb[:, :, 1] = np.clip(input_np.astype(np.int32) + gt_np, 0, 255).astype(np.uint8)  # Green - enhanced for GT
    gt_rgb[:, :, 2] = input_np  # Blue channel - background
    
    # Prediction: red channel for predictions (to distinguish from GT)
    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    pred_rgb[:, :, 0] = np.clip(input_np.astype(np.int32) + pred_np, 0, 255).astype(np.uint8)  # Red - enhanced for pred
    pred_rgb[:, :, 1] = input_np  # Green channel - background
    pred_rgb[:, :, 2] = input_np  # Blue channel - background
    
    # Add border between images
    border = np.ones((H, 2, 3), dtype=np.uint8) * 255
    
    # Concatenate horizontally: Input | GT | Pred
    comparison = np.concatenate([input_rgb, border, gt_rgb, border, pred_rgb], axis=1)
    
    # Add labels
    # Create a label bar at the top
    label_bar_height = 25
    label_bar = np.ones((label_bar_height, comparison.shape[1], 3), dtype=np.uint8) * 50  # Dark gray
    
    # Final image with label bar
    final_img = np.concatenate([label_bar, comparison], axis=0)
    
    # Save image
    img = Image.fromarray(final_img, mode='RGB')
    
    # Add text labels using PIL
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Draw labels
    draw.text((W // 2 - 30, 5), "Input", fill=(255, 255, 255), font=font)
    draw.text((W + 2 + W // 2 - 50, 5), "Ground Truth", fill=(0, 255, 0), font=font)
    draw.text((2 * W + 4 + W // 2 - 45, 5), "Prediction", fill=(255, 100, 100), font=font)
    
    img.save(save_path)


def save_comparison_v2(input_img, gt_mask, pred_mask, save_path):
    """
    Alternative comparison visualization with overlay.
    
    Shows:
    - Row 1: Input | GT overlay | Pred overlay
    - Row 2: GT mask | Pred mask | Difference
    
    Args:
        input_img: Input image tensor [1, H, W]
        gt_mask: Ground truth mask tensor [1, H, W]
        pred_mask: Predicted mask tensor [1, H, W]
        save_path: Path to save comparison image
    """
    H, W = input_img.shape[1], input_img.shape[2]
    
    # Convert tensors to numpy (0-255)
    input_np = (input_img.squeeze().cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred_np = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    # Row 1: Input | GT overlay | Pred overlay
    # Input: grayscale to RGB
    input_rgb = np.stack([input_np, input_np, input_np], axis=-1)
    
    # GT overlay (green)
    gt_overlay = input_rgb.copy()
    gt_mask_bool = gt_np > 127
    gt_overlay[gt_mask_bool, 0] = np.clip(gt_overlay[gt_mask_bool, 0] * 0.5, 0, 255)
    gt_overlay[gt_mask_bool, 1] = np.clip(gt_overlay[gt_mask_bool, 1] * 0.5 + 180, 0, 255)
    gt_overlay[gt_mask_bool, 2] = np.clip(gt_overlay[gt_mask_bool, 2] * 0.5, 0, 255)
    
    # Pred overlay (red)
    pred_overlay = input_rgb.copy()
    pred_mask_bool = pred_np > 127
    pred_overlay[pred_mask_bool, 0] = np.clip(pred_overlay[pred_mask_bool, 0] * 0.5 + 180, 0, 255)
    pred_overlay[pred_mask_bool, 1] = np.clip(pred_overlay[pred_mask_bool, 1] * 0.5, 0, 255)
    pred_overlay[pred_mask_bool, 2] = np.clip(pred_overlay[pred_mask_bool, 2] * 0.5, 0, 255)
    
    # Row 2: GT mask | Pred mask | Difference
    gt_mask_rgb = np.stack([gt_np, gt_np, gt_np], axis=-1)
    pred_mask_rgb = np.stack([pred_np, pred_np, pred_np], axis=-1)
    
    # Difference: Green = TP, Red = FP, Blue = FN
    diff = np.zeros((H, W, 3), dtype=np.uint8)
    tp = gt_mask_bool & pred_mask_bool  # True Positive
    fp = ~gt_mask_bool & pred_mask_bool  # False Positive
    fn = gt_mask_bool & ~pred_mask_bool  # False Negative
    diff[tp, 1] = 255  # Green for TP
    diff[fp, 0] = 255  # Red for FP
    diff[fn, 2] = 255  # Blue for FN
    
    # Add border between images
    border_v = np.ones((H, 2, 3), dtype=np.uint8) * 255
    border_h = np.ones((2, W * 3 + 4, 3), dtype=np.uint8) * 255
    
    # Row 1
    row1 = np.concatenate([input_rgb, border_v, gt_overlay, border_v, pred_overlay], axis=1)
    
    # Row 2
    row2 = np.concatenate([gt_mask_rgb, border_v, pred_mask_rgb, border_v, diff], axis=1)
    
    # Combine rows
    comparison = np.concatenate([row1, border_h, row2], axis=0)
    
    # Add label bar
    label_bar_height = 25
    label_bar = np.ones((label_bar_height, comparison.shape[1], 3), dtype=np.uint8) * 50
    
    final_img = np.concatenate([label_bar, comparison], axis=0)
    
    # Save image
    img = Image.fromarray(final_img, mode='RGB')
    
    # Add text labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Labels for row 1
    draw.text((W // 2 - 20, 5), "Input", fill=(255, 255, 255), font=font)
    draw.text((W + 2 + W // 2 - 30, 5), "GT (green)", fill=(0, 255, 0), font=font)
    draw.text((2 * W + 4 + W // 2 - 35, 5), "Pred (red)", fill=(255, 100, 100), font=font)
    
    img.save(save_path)


# ==================== Main Evaluation Function ====================

@torch.no_grad()
def evaluate():
    """Main evaluation function"""
    print("\n" + "=" * 70)
    print("RCBANet Evaluation - IRSTD-1K Dataset")
    print("=" * 70)
    
    # Check device
    device = DEVICE
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    mask_dir = Path(MASK_DIR)
    comparison_dir = Path(COMPARISON_DIR)
    mask_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directories created:")
    print(f"  Masks: {mask_dir}")
    print(f"  Comparisons: {comparison_dir}")
    
    # Create model
    print("\nCreating model...")
    cfg = RCBANetConfig(
        C1=C1, C2=C2, C3=C3,
        cau_ch_s1=CAU_CH_S1, cau_ch_s2=CAU_CH_S2, cau_ch_s3=CAU_CH_S3,
        cau_kernels=CAU_KERNELS, cau_strides=CAU_STRIDES,
        base_ch_s1=BASE_CH_S1, base_ch_s2=BASE_CH_S2, base_ch_s3=BASE_CH_S3,
        mid_ch_s1=MID_CH_S1, mid_ch_s2=MID_CH_S2, mid_ch_s3=MID_CH_S3,
        prox_depth=PROX_DEPTH, grad_depth=GRAD_DEPTH,
        use_gradient_checkpointing=False,
        deep_supervision=False
    )
    
    model = RCBANet(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint_info = load_checkpoint(model, CHECKPOINT_PATH, device=device)
    print(f"Checkpoint from epoch: {checkpoint_info['epoch']}")
    if 'metrics' in checkpoint_info and checkpoint_info['metrics']:
        print(f"Training metrics: {checkpoint_info['metrics']}")
    
    # Set model to eval mode
    model.eval()
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    data_path = Path(DATA_ROOT) / DATASET_NAME
    test_root = data_path / 'test'
    
    if not test_root.exists():
        print(f"\n❌ Error: Test dataset path does not exist: {test_root}")
        print(f"Please modify DATA_ROOT or DATASET_NAME in evaluate.py")
        return
    
    test_dataset = ISTDDataset(
        root=str(test_root),
        split='test',
        image_size=IMAGE_SIZE,
        use_augmentation=False
    )
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize metrics
    eval_miou = mIoU()
    eval_pd_fa = PD_FA()
    
    # Evaluate
    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)
    
    image_names = test_dataset.image_names
    
    for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        # Forward pass
        outputs = model(images)
        pred_mask = outputs["mask"]  # [B, 1, H, W]
        
        # Binary prediction
        pred_binary = (pred_mask > THRESHOLD).float()
        
        # Update metrics
        eval_miou.update(pred_binary, masks)
        
        for i in range(pred_mask.shape[0]):
            size = (masks.shape[2], masks.shape[3])
            eval_pd_fa.update(
                (pred_mask[i, 0, :, :] > THRESHOLD),
                masks[i, 0, :, :],
                size
            )
        
        # Save masks and comparison images
        for i in range(pred_mask.shape[0]):
            sample_idx = idx * BATCH_SIZE + i
            if sample_idx < len(image_names):
                img_name = Path(image_names[sample_idx]).stem
            else:
                img_name = f"sample_{sample_idx:04d}"
            
            # Save predicted mask
            mask_path = mask_dir / f"{img_name}_pred.png"
            save_mask(pred_binary[i], mask_path)
            
            # Save comparison image
            comparison_path = comparison_dir / f"{img_name}_compare.png"
            save_comparison(images[i], masks[i], pred_binary[i], comparison_path)
    
    # Get final metrics
    pixAcc, mIoU_val, F1, precision, recall = eval_miou.get()
    PD, FA = eval_pd_fa.get()
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 Performance Metrics:")
    print(f"   ├── Pixel Accuracy:    {pixAcc * 100:.4f}%")
    print(f"   ├── mIoU:              {mIoU_val * 100:.4f}%")
    print(f"   ├── F1 Score:          {F1 * 100:.4f}%")
    print(f"   ├── Precision:         {precision * 100:.4f}%")
    print(f"   ├── Recall:            {recall * 100:.4f}%")
    print(f"   ├── PD (Detection):    {PD * 100:.4f}%")
    print(f"   └── FA (False Alarm):  {FA * 1e6:.4f} × 10⁻⁶")
    
    print(f"\n📁 Output Files:")
    print(f"   ├── Masks saved to:       {mask_dir}")
    print(f"   └── Comparisons saved to: {comparison_dir}")
    print(f"   Total images processed:   {len(test_dataset)}")
    
    # Save metrics to file
    metrics_file = Path(OUTPUT_DIR) / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("RCBANet Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Pixel Accuracy: {pixAcc * 100:.4f}%\n")
        f.write(f"mIoU:           {mIoU_val * 100:.4f}%\n")
        f.write(f"F1 Score:       {F1 * 100:.4f}%\n")
        f.write(f"Precision:      {precision * 100:.4f}%\n")
        f.write(f"Recall:         {recall * 100:.4f}%\n")
        f.write(f"PD:             {PD * 100:.4f}%\n")
        f.write(f"FA:             {FA * 1e6:.4f} × 10^-6\n")
    
    print(f"\n📄 Metrics saved to: {metrics_file}")
    print("\n" + "=" * 70)
    print("✅ Evaluation completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate()

