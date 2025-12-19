#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CGL-RPCANet Training Script - NUAA-SIRST Dataset
Run this file directly to start training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import sys
import numpy as np
import threading
from datetime import datetime
from skimage import measure

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgl_rpcanet.models.rcbanet import RCBANet, RCBANetConfig
from cgl_rpcanet.losses import CombinedLoss, LossWeights
from cgl_rpcanet.data import ISTDDataset, get_dataloader
from cgl_rpcanet.utils.metrics import compute_metrics
from cgl_rpcanet.utils.checkpoint import save_checkpoint, load_checkpoint


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

        return float(pixAcc), float(mIoU_val), float(f1_score)

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


# ==================== Configuration Parameters ====================
# Modify these parameters according to your needs

# Dataset configuration
DATA_ROOT = "/home/gsx/pycahrm_projects/LRCBA-3/datasets"  # Dataset root path
# DATA_ROOT = "/home/gsx/pycahrm_projects/LRCBA-14/dataset"  # Dataset root path
DATASET_NAME = "IRSTD-1k"   # Dataset name: IRSTD-1k, NUDT-SIRST, sirst_aug
TRAIN_SPLIT_FILE = None     # Training split file path (if any)
VAL_SPLIT_FILE = None        # Validation split file path (if any)
IMAGE_SIZE = (256, 256)      # Image size (H, W)
# IMAGE_SIZE = (512, 512)      # Image size (H, W)
CROP_SIZE = (256, 256)       # Random crop size for training (H, W), set to None to disable
USE_AUGMENTATION = True      # Enable IR-specific data augmentation (GSFANet-style)

# Training configuration (optimized)
EPOCHS = 400                # Total training epochs
BATCH_SIZE = 8               # Batch size (reduced from 16 to 8 to save VRAM, can reduce to 4 if OOM)
LEARNING_RATE = 1e-3         # Initial learning rate (OneCycleLR will adjust automatically)
WEIGHT_DECAY = 1e-4          # Weight decay (increased from 1e-5 to 1e-4)
WARMUP_EPOCHS = 5            # Learning rate warmup epochs
GRAD_CLIP_NORM = 1.0         # Gradient clipping threshold

# Model configuration (RCBANet - UNet × RCBA multi-scale architecture)
# Encoder pyramid channels
C1 = 256                      # Full resolution (1/1)
C2 = 256                     # Half resolution (1/2)
C3 = 256                  # Quarter resolution (1/4)

# CAU configuration per scale
CAU_CH_S1 = 128               # CAU channels at scale 1 (1/1) - fine
CAU_CH_S2 = 128               # CAU channels at scale 2 (1/2) - mid
CAU_CH_S3 = 128              # CAU channels at scale 3 (1/4) - coarse
# CAU_KERNELS = [3, 7]         # Multi-scale kernels for CAU
# CAU_STRIDES = [8, 6, 2]      # Per-scale strides: [s3, s2, s1] - coarse stride=2, mid/fine stride=1
CAU_KERNELS = [3, 5]         # Multi-scale kernels for CAU
CAU_STRIDES = [8, 8, 8]      # Per-scale strides: [s3, s2, s1] - coarse stride=2, mid/fine stride=1

# Prox/Grad network configuration per scale
BASE_CH_S1 = 32              # Base channels at scale 1 (full res)
BASE_CH_S2 = 32              # Base channels at scale 2 (1/2 res)
BASE_CH_S3 = 24              # Base channels at scale 3 (1/4 res)
# MID_CH_S1 = 16               # Bottleneck channels at scale 1
# MID_CH_S2 = 16               # Bottleneck channels at scale 2
# MID_CH_S3 = 12               # Bottleneck channels at scale 3
MID_CH_S1 = 24               # Bottleneck channels at scale 1
MID_CH_S2 = 24               # Bottleneck channels at scale 2
MID_CH_S3 = 16               # Bottleneck channels at scale 3
# MID_CH_S1 = 64               # Bottleneck channels at scale 1
# MID_CH_S2 = 32               # Bottleneck channels at scale 2
# MID_CH_S3 = 16               # Bottleneck channels at scale 3
PROX_DEPTH = 2               # Depth of ProxNetB
GRAD_DEPTH = 2               # Depth of GradNet

# Training configuration
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing in CAU
DEEP_SUPERVISION = True      # Enable deep supervision on stage 2 & 3

# Loss weights (GSFANet-style configuration)
ETA_FIDELITY = 0.01          # Reconstruction loss weight (small fidelity anchor for stable iterations)
LAMBDA_DL = 0.0              # Distortionless regularization weight
LAMBDA_MV = 1e-4             # Minimum variance regularization weight (recommended: 1e-4)
DEEP_SUP_WEIGHT = 0.01        # Deep supervision weight for stage_logits (GSFANet-style)

# GSFANet-style loss parameters
IOU_WEIGHT = 1.0             # Weight for SoftIoU term in GSFANet loss

# Checkpoint configuration
SAVE_DIR = "./checkpoints/IRSTD-1k_1218-1"  # Checkpoint save directory
SAVE_FREQ = 10                         # Save frequency (every N epochs)
RESUME_CHECKPOINT = None               # Resume training checkpoint path (None for training from scratch)

# Other configuration
NUM_WORKERS = 4              # Number of data loading threads
DEVICE = "cuda"              # Device: cuda or cpu
SEED = 42                    # Random seed

# ==================================================


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    """Train one epoch"""
    model.train()

    running_loss = 0.0
    running_iou = 0.0
    running_fid = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss_dict = criterion(outputs, masks, images)
        loss = loss_dict["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent gradient explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()  # OneCycleLR updates after each batch

        # Accumulate metrics
        running_loss += loss.item()
        running_iou += loss_dict["loss_iou_main"].item()
        running_fid += loss_dict["loss_fidelity"].item()
        num_batches += 1

        # Update progress bar (show GSFANet-style loss components)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou": f"{loss_dict['loss_iou_main'].item():.4f}",
            "ada": f"{loss_dict.get('loss_ada_focal', torch.tensor(0.0)).item():.4f}",
            "fid": f"{loss_dict['loss_fidelity'].item():.4f}"
        })

    # Calculate average metrics
    avg_metrics = {
        "train_loss": running_loss / num_batches,
        "train_iou_loss": running_iou / num_batches,
        "train_fidelity": running_fid / num_batches
    }

    return avg_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()

    running_loss = 0.0
    num_batches = 0

    # Initialize evaluation metrics
    eval_miou = mIoU()
    eval_pd_fa = PD_FA()

    for images, masks in tqdm(dataloader, desc="Validation"):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        # Forward pass
        outputs = model(images)
        pred_mask = outputs["mask"]  # [B, 1, H, W]

        # Handle size mismatch: resize pred_mask to match masks
        if pred_mask.shape[2:] != masks.shape[2:]:
            pred_mask = torch.nn.functional.interpolate(
                pred_mask, size=masks.shape[2:], mode='bilinear', align_corners=False
            )

        # Calculate loss
        loss_dict = criterion(outputs, masks, images)
        running_loss += loss_dict["loss"].item()

        # Update mIoU metrics
        eval_miou.update((pred_mask > 0.5), masks)

        # Update PD_FA metrics (per sample)
        for i in range(pred_mask.shape[0]):
            size = (masks.shape[2], masks.shape[3])
            eval_pd_fa.update(
                (pred_mask[i, 0, :, :] > 0.5),
                masks[i, 0, :, :],
                size
            )

        num_batches += 1

    # Get final metrics
    pixAcc, mIoU_val, F1 = eval_miou.get()
    PD, FA = eval_pd_fa.get()

    # Calculate average metrics
    avg_metrics = {
        "val_loss": running_loss / num_batches,
        "val_pixAcc": pixAcc,
        "val_mIoU": mIoU_val,
        "val_F1": F1,
        "val_PD": PD,
        "val_FA": FA
    }

    return avg_metrics


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("CGL-RPCANet Training - NUAA-SIRST Dataset")
    print("="*70)

    # Set random seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Check device
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = DATASET_NAME  # Use configured dataset name
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log-{dataset_name}-{timestamp}.txt"

    # Open log file
    f_log = open(log_file, 'w')

    def log_print(message):
        """Print to both console and log file"""
        print(message)
        f_log.write(message + '\n')
        f_log.flush()

    log_print(f"\nLog file: {log_file}")
    log_print(f"Training start time: {timestamp}")

    # Print configuration
    log_print("\nTraining Configuration:")
    log_print(f"  Dataset: {dataset_name}")
    log_print(f"  Data path: {DATA_ROOT}/{DATASET_NAME}")
    log_print(f"  Training set: trainval/")
    log_print(f"  Validation set: test/")
    log_print(f"  Training epochs: {EPOCHS}")
    log_print(f"  Batch size: {BATCH_SIZE}")
    log_print(f"  Learning rate: {LEARNING_RATE}")
    log_print(f"  Image size: {IMAGE_SIZE}")
    log_print(f"  Crop size: {CROP_SIZE} (GSFANet-style, pos_prob=0.5)")
    log_print(f"  Data augmentation: {USE_AUGMENTATION} (GSFANet: flip/transpose/rotate±3°)")
    log_print(f"  Normalization: mean/std (GSFANet-style, dataset-specific)")

    log_print("\nModel Configuration (RCBANet - UNet × RCBA):")
    log_print(f"  Encoder Pyramid: C1={C1}, C2={C2}, C3={C3}")
    log_print(f"  CAU channels: S1={CAU_CH_S1}, S2={CAU_CH_S2}, S3={CAU_CH_S3}")
    log_print(f"  CAU kernels: {CAU_KERNELS}")
    log_print(f"  CAU strides: {CAU_STRIDES} (coarse→mid→fine)")
    log_print(f"  Base channels: S1={BASE_CH_S1}, S2={BASE_CH_S2}, S3={BASE_CH_S3}")
    log_print(f"  Mid channels: S1={MID_CH_S1}, S2={MID_CH_S2}, S3={MID_CH_S3}")
    log_print(f"  Prox depth: {PROX_DEPTH}, Grad depth: {GRAD_DEPTH}")
    log_print(f"  Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    log_print(f"  Deep supervision: {DEEP_SUPERVISION}")
    log_print(f"  Cross-stage fusion: Enabled (last 2 target maps)")

    log_print("\nLoss Weights (GSFANet-style):")
    log_print(f"  Fidelity: {ETA_FIDELITY}")
    log_print(f"  IoU weight: {IOU_WEIGHT}")
    log_print(f"  Deep supervision: {DEEP_SUP_WEIGHT}")
    log_print(f"  MVDR regularization: {LAMBDA_MV}")
    log_print(f"  Loss: AdaFocalLoss + SoftIoULoss")

    # Create model
    log_print("\nCreating model...")
    cfg = RCBANetConfig(
        # Encoder pyramid channels
        C1=C1,
        C2=C2,
        C3=C3,
        # CAU configuration per scale
        cau_ch_s1=CAU_CH_S1,
        cau_ch_s2=CAU_CH_S2,
        cau_ch_s3=CAU_CH_S3,
        cau_kernels=CAU_KERNELS,
        cau_strides=CAU_STRIDES,
        # Prox/Grad network configuration per scale
        base_ch_s1=BASE_CH_S1,
        base_ch_s2=BASE_CH_S2,
        base_ch_s3=BASE_CH_S3,
        mid_ch_s1=MID_CH_S1,
        mid_ch_s2=MID_CH_S2,
        mid_ch_s3=MID_CH_S3,
        prox_depth=PROX_DEPTH,
        grad_depth=GRAD_DEPTH,
        # Training configuration
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        deep_supervision=DEEP_SUPERVISION,
        deep_sup_weight=DEEP_SUP_WEIGHT
    )

    model = RCBANet(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log_print(f"Model parameters: {num_params:,}")

    # Check dataset path
    data_root = Path(DATA_ROOT)
    dataset_path = data_root / DATASET_NAME
    if not dataset_path.exists():
        print(f"\n❌ Error: Dataset path does not exist: {dataset_path}")
        print(f"Please modify DATA_ROOT or DATASET_NAME variable in train.py")
        print(f"\nDataset should have the following directory structure:")
        print(f"  {dataset_path}/")
        print(f"  ├── trainval/")
        print(f"  │   ├── images/")
        print(f"  │   └── masks/")
        print(f"  └── test/")
        print(f"      ├── images/")
        print(f"      └── masks/")
        return

    # Create datasets - training set uses trainval, validation set uses test
    log_print("\nLoading dataset...")
    try:
        # Training set: use trainval directory
        train_root = dataset_path / 'trainval'
        train_dataset = ISTDDataset(
            root=str(train_root),
            split='train',
            split_file=TRAIN_SPLIT_FILE,
            image_size=IMAGE_SIZE,
            use_augmentation=USE_AUGMENTATION,  # Enable GSFANet-style augmentation
            crop_size=CROP_SIZE,                # Enable GSFANet-style random crop (pos_prob=0.5)
            dataset_name=DATASET_NAME           # For mean/std normalization
        )

        # Validation set: use test directory
        val_root = dataset_path / 'test'
        val_dataset = ISTDDataset(
            root=str(val_root),
            split='val',
            split_file=VAL_SPLIT_FILE,
            image_size=IMAGE_SIZE,
            dataset_name=DATASET_NAME           # For mean/std normalization
        )

        log_print(f"Training samples: {len(train_dataset)}")
        log_print(f"Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        print(f"\nTip: Make sure the dataset directory structure is correct:")
        print(f"  {dataset_path}/trainval/images/  <- Training image files")
        print(f"  {dataset_path}/trainval/masks/   <- Training label masks")
        print(f"  {dataset_path}/test/images/      <- Test image files")
        print(f"  {dataset_path}/test/masks/       <- Test label masks")
        return

    # Create data loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # Validation uses PadImg (variable size), so we must use batch_size=1
    val_loader = get_dataloader(
        val_dataset,
        batch_size=1,  # Must be 1 for variable-size validation images (GSFANet-style)
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Create loss function (GSFANet-style AdaFocal + SoftIoU)
    loss_weights = LossWeights(
        eta_fidelity=ETA_FIDELITY,
        lambda_dl=LAMBDA_DL,
        lambda_mv=LAMBDA_MV,
        deep_sup_weight=DEEP_SUP_WEIGHT,
        iou_weight=IOU_WEIGHT
    )
    criterion = CombinedLoss(loss_weights).to(device)

    # Create optimizer (use AdamW instead of Adam)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler (use OneCycleLR instead of CosineAnnealingLR)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=WARMUP_EPOCHS / EPOCHS,  # Warmup phase ratio
        anneal_strategy='cos',
        div_factor=25.0,    # Initial lr = max_lr / div_factor
        final_div_factor=1e4  # Final lr = max_lr / final_div_factor
        # final_div_factor=1e2  # Final lr = max_lr / final_div_factor
    )

    log_print(f"\nOptimizer: AdamW (lr={LEARNING_RATE:.2e}, wd={WEIGHT_DECAY:.2e})")
    log_print(f"Scheduler: OneCycleLR (warmup={WARMUP_EPOCHS} epochs, grad_clip={GRAD_CLIP_NORM})")

    # Resume training (if specified)
    start_epoch = 0
    best_val_iou = 0.0

    if RESUME_CHECKPOINT is not None and Path(RESUME_CHECKPOINT).exists():
        log_print(f"\nResuming training from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint_info = load_checkpoint(model, RESUME_CHECKPOINT, optimizer, device)
        start_epoch = checkpoint_info["epoch"] + 1
        best_val_iou = checkpoint_info["metrics"].get("val_mIoU", 0.0)
        log_print(f"Continuing from epoch {start_epoch}, best mIoU: {best_val_iou*100:.2f}%")

    # Create save directory
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_print(f"\nCheckpoint save path: {save_dir}")

    # Training loop
    log_print("\n" + "="*70)
    log_print("Starting training...")
    log_print("="*70 + "\n")

    for epoch in range(start_epoch, EPOCHS):
        log_print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        log_print("-" * 70)

        # Training
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch+1
        )

        # Validation
        val_metrics = validate(model, val_loader, criterion, device)

        # Get current learning rate (OneCycleLR already updated in each batch)
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        log_print(f"\nTraining - Loss: {train_metrics['train_loss']:.4f}, "
                  f"IoU Loss: {train_metrics['train_iou_loss']:.4f}, "
                  f"Fidelity: {train_metrics['train_fidelity']:.4f}")
        log_print(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                  f"pixAcc: {val_metrics['val_pixAcc']*100:.2f}%, "
                  f"mIoU: {val_metrics['val_mIoU']*100:.2f}%, "
                  f"F1: {val_metrics['val_F1']*100:.2f}%")
        log_print(f"       PD: {val_metrics['val_PD']*100:.2f}%, "
                  f"FA: {val_metrics['val_FA']*1e6:.4f}e-6")
        log_print(f"Learning rate: {current_lr:.2e}")

        # Save checkpoint
        is_best = val_metrics['val_mIoU'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['val_mIoU']
            log_print(f" New best model! mIoU: {best_val_iou*100:.2f}%")

        if (epoch + 1) % SAVE_FREQ == 0 or is_best:
            all_metrics = {**train_metrics, **val_metrics}
            save_checkpoint(
                model,
                optimizer,
                epoch,
                all_metrics,
                save_dir / f"checkpoint_epoch_{epoch+1}.pth",
                is_best=is_best
            )

    log_print("\n" + "="*70)
    log_print(f"Training completed! Best validation mIoU: {best_val_iou*100:.2f}%")
    log_print(f"Best model saved at: {save_dir / 'best_model.pth'}")
    log_print(f"Log file saved at: {log_file}")
    log_print("="*70 + "\n")

    # Close log file
    f_log.close()
    print(f"\n✅ Training completed! Log saved to: {log_file}")


if __name__ == "__main__":
    main()
