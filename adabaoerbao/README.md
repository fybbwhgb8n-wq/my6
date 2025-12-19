# LRCBA-10 English Complete Package

This directory contains all the code files required to run the training script `/train.py` for the CGL-RPCANet (Capon-Guided Lightweight RPCA Unfolding Network) project for infrared small target detection (ISTD).

## Project Overview

**CGL-RPCANet** is a deep learning model that combines:
- RCBA (Robust Component and Background Analysis) unfolding with Capon attention
- UNet-like multi-scale architecture
- Advanced loss functions for small target detection

## File Directory Structure

```
LRCBA-10-English-Complete/
├── train.py                              # Main training script
├── README.md                             # This file
└── cgl_rpcanet/                          # Main package directory
    ├── __init__.py                       # Package initialization
    ├── data.py                           # Dataset loaders and data utilities
    ├── losses.py                         # Loss functions (SoftIoU, Focal-Tversky, etc.)
    ├── models/                           # Model architectures
    │   ├── __init__.py                   # Models package init
    │   ├── rcbanet.py                    # RCBANet main architecture
    │   ├── modules.py                    # Neural network modules (SE, ResidualBottleneck, etc.)
    │   └── capon.py                      # Capon attention modules
    └── utils/                            # Utility functions
        ├── __init__.py                   # Utils package init
        ├── checkpoint.py                 # Checkpoint saving/loading utilities
        └── metrics.py                    # Evaluation metrics (IoU, Dice, etc.)
```

## File Descriptions

### Main Script
- **train.py** (648 lines)
  - Main training script for NUAA-SIRST dataset
  - Configurable training parameters
  - Includes metrics: mIoU, pixAcc, F1, PD, FA
  - OneCycleLR scheduler with gradient clipping
  - Checkpoint saving and logging

### Core Package (cgl_rpcanet/)

#### Dataset & Data Processing
- **data.py** (383 lines)
  - `ISTDDataset`: Generic ISTD dataset loader for image-mask pairs
  - `DummyISTDDataset`: Synthetic dataset for quick testing
  - `apply_ir_augmentations()`: IR-specific augmentations (noise, blur, etc.)
  - `random_crop_with_target()`: Target-centered cropping for hard positive sampling
  - `get_dataloader()`: DataLoader creation with standard settings

#### Loss Functions
- **losses.py** (419 lines)
  - `CombinedLoss`: Main loss combining multiple components
  - `soft_iou_loss()`: Differentiable IoU loss
  - `focal_tversky_loss()`: Focal Tversky loss for recall emphasis
  - `fidelity_loss()`: MSE reconstruction loss
  - `mvdr_regularizers()`: MVDR regularization for Capon attention
  - `orthogonality_loss()`: B⊥T regularization
  - `sparsity_loss()`: Sparsity regularization on target
  - Stage-wise BCE+IoU deep supervision

#### Model Architecture (models/)
- **rcbanet.py** (678 lines)
  - `RCBANet`: Main network architecture
  - `RCBANetConfig`: Configuration dataclass
  - `EncoderPyramid`: 3-level feature pyramid (1/1, 1/2, 1/4)
  - `StageOp`: Single stage of scale-aligned RCBA unfolding
  - `MicroDecoder`: UNet-like upsampling with skip connections
  - `FusionHead`: Cross-stage fusion of target maps

- **modules.py** (306 lines)
  - `SEModule`: Squeeze-and-Excitation for channel attention
  - `ResidualBottleneck`: Lightweight residual block
  - `ProxNetB`: SVD-free proximal surrogate for background estimation
  - `GradNet`: BN-free gradient surrogate for target/noise updates
  - `ReconNet`: Reconstruction network (B+T+N → D)

- **capon.py** (363 lines)
  - `PSFHead`: Predicts steering vector from channel statistics
  - `CaponAttentionUnit`: Channel-wise MVDR attention over spatial tiles
  - `MultiScaleCaponAttention`: Multi-scale Capon with parallel kernels

#### Utilities (utils/)
- **checkpoint.py** (94 lines)
  - `save_checkpoint()`: Save model, optimizer, metrics
  - `load_checkpoint()`: Load checkpoint for resuming training

- **metrics.py** (117 lines)
  - `compute_iou()`: Intersection over Union
  - `compute_dice()`: Dice coefficient (F1 score)
  - `compute_pixel_metrics()`: Precision, recall, F1
  - `compute_metrics()`: All metrics combined

## Dependencies

The project requires the following Python packages:
- PyTorch (torch, torch.nn, torch.optim)
- NumPy
- PIL (Python Imaging Library)
- scikit-image (skimage.measure)
- scipy (for augmentations)
- tqdm (progress bars)
- pathlib (file paths)

## How to Run

1. **Prepare your dataset** following this structure:
   ```
   datasets/
   └── IRSTD-1k/  (or your dataset name)
       ├── trainval/
       │   ├── images/
       │   └── masks/
       └── test/
           ├── images/
           └── masks/
   ```

2. **Configure training parameters** in `train.py`:
   - `DATA_ROOT`: Dataset root path
   - `DATASET_NAME`: Dataset name (IRSTD-1k, NUDT-SIRST, etc.)
   - `BATCH_SIZE`: Batch size (default: 8)
   - `EPOCHS`: Total training epochs (default: 1000)
   - `LEARNING_RATE`: Initial learning rate (default: 1e-4)
   - Model architecture parameters (channels, depths, etc.)
   - Loss weights

3. **Run training**:
   ```bash
   python train.py
   ```

4. **Resume training** (optional):
   - Set `RESUME_CHECKPOINT` to your checkpoint path in `train.py`
   - Re-run `python train.py`

## Model Configuration

The default RCBANet configuration:
- **Encoder Pyramid**: C1=32, C2=48, C3=64 (3-level feature pyramid)
- **CAU Channels**: S1=24, S2=16, S3=16 (per-scale)
- **CAU Kernels**: [3, 5] (multi-scale)
- **Prox/Grad Networks**: Base channels 32/32/24, mid channels 16/16/12
- **Deep Supervision**: Enabled on stages 2 & 3
- **Gradient Checkpointing**: Enabled to save memory

## Loss Configuration

The default loss weights (SCTransNet-style):
- **Fidelity**: 0.05 (reconstruction loss)
- **BCE Weight**: 1.0 (binary cross-entropy)
- **IoU Weight**: 1.0 (soft IoU loss)
- **BCE Deep-Sup Weight**: 0.15 (stage-wise supervision)
- **MVDR Regularization**: 1e-4 (minimum variance)
- **Dynamic pos_weight**: Enabled (max=60.0) for class imbalance

## Output

Training produces:
- **Checkpoints**: Saved in `./checkpoints/IRSTD-1k_1119/` (configurable)
  - Regular checkpoints every N epochs
  - Best model checkpoint (highest mIoU)
- **Logs**: Saved in `./logs/` directory
  - Training metrics per epoch
  - Validation metrics per epoch
  - Best model performance

## Evaluation Metrics

The training script computes:
- **mIoU**: Mean Intersection over Union
- **pixAcc**: Pixel accuracy
- **F1**: F1 score
- **PD**: Probability of Detection
- **FA**: False Alarm rate (per million pixels)

## Translation Notes

All files in this directory have been verified to contain **only English text**. The original Chinese project description files (markdown documentation) have been excluded as they are not required for running the training code.

The code is fully self-contained and ready to use for:
- Training new models from scratch
- Resuming training from checkpoints
- Evaluating model performance
- Experimenting with different configurations

## Contact

This is an English translation of the LRCBA-10 project created on November 19, 2025.

For questions about the original implementation, please refer to the parent directory documentation.

