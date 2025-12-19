"""
Loss functions for CGL-RPCANet training.
GSFANet-style losses: AdaFocalLoss + SoftIoULoss.
Also includes legacy loss components for model-specific regularizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict


# ==================== GSFANet-style Loss Functions ====================

class AdaFocalLoss(nn.Module):
    """
    Adaptive Focal Loss from GSFANet.
    
    Features:
        - Adaptive parameter adjustment based on target area
        - IoU-based focal exponent
        - Area-weighted loss for small target enhancement
    
    Args:
        pred: Predicted logits ∈ ℝ[B, 1, H, W] (pre-sigmoid)
        target: Ground truth mask ∈ ℝ[B, 1, H, W] in {0, 1}
    
    Returns:
        Scalar loss (sum over all pixels)
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Handle size mismatch: resize pred to match target
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算面积自适应权重
        area_weight = self._get_area_weight(target)  # [N,1,1,1]
        smooth = 1

        intersection = pred.sigmoid() * target
        iou = (intersection.sum() + smooth) / (pred.sigmoid().sum() + target.sum() - intersection.sum() + smooth)
        iou = torch.clamp(iou, min=1e-6, max=1 - 1e-6).detach()
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # 自带sigmoid

        target = target.type(torch.long)
        at = target * area_weight + (1 - target) * (1 - area_weight)
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        F_loss = (1 - pt) ** (1 - iou + 1e-6) * BCE_loss

        F_loss = at * F_loss
        return F_loss.sum()

    def _get_area_weight(self, target):
        """Compute area-based weight for small target enhancement."""
        # 小目标增强权重
        area = target.sum(dim=(1, 2, 3))  # [N,1]
        return torch.sigmoid(1 - area / (area.max() + 1)).view(-1, 1, 1, 1)


class SoftIoULoss(nn.Module):
    """
    Soft IoU Loss from GSFANet.
    
    Computes differentiable IoU loss between predicted and target masks.
    
    Args:
        pred: Predicted logits ∈ ℝ[B, 1, H, W] (pre-sigmoid)
        mask: Ground truth mask ∈ ℝ[B, 1, H, W] in {0, 1}
    
    Returns:
        Scalar loss: 1 - IoU
    """
    
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def IOU(self, pred, mask):
        smooth = 1

        intersection = pred * mask
        loss = (intersection.sum() + smooth) / (pred.sum() + mask.sum() - intersection.sum() + smooth)

        loss = 1 - torch.mean(loss)
        return loss

    def forward(self, pred, mask):
        # Handle size mismatch: resize pred to match mask
        if pred.shape[2:] != mask.shape[2:]:
            pred = F.interpolate(pred, size=mask.shape[2:], mode='bilinear', align_corners=False)
        
        pred = torch.sigmoid(pred)
        loss_iou = self.IOU(pred, mask)

        return loss_iou


class GSFANetLoss(nn.Module):
    """
    Combined GSFANet-style loss: AdaFocalLoss + SoftIoULoss.
    
    Total loss = AdaFocal + λ_iou * SoftIoU
    
    Args:
        iou_weight: Weight for SoftIoU loss (default: 1.0)
    """
    
    def __init__(self, iou_weight: float = 1.0):
        super().__init__()
        self.ada_focal = AdaFocalLoss()
        self.soft_iou = SoftIoULoss()
        self.iou_weight = iou_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits ∈ ℝ[B, 1, H, W] (pre-sigmoid)
            target: Ground truth mask ∈ ℝ[B, 1, H, W] in {0, 1}
        
        Returns:
            Dictionary with:
                - loss: Total scalar loss
                - loss_ada_focal: AdaFocal loss component
                - loss_iou: SoftIoU loss component
        """
        L_ada_focal = self.ada_focal(pred, target)
        L_iou = self.soft_iou(pred, target)
        
        total_loss = L_ada_focal + self.iou_weight * L_iou
        
        return {
            "loss": total_loss,
            "loss_ada_focal": L_ada_focal.detach(),
            "loss_iou": L_iou.detach()
        }


# ==================== Legacy Loss Components (for model-specific features) ====================

@dataclass
class LossWeights:
    """Weights for different loss components."""
    eta_fidelity: float = 0.1           # Fidelity loss weight
    lambda_dl: float = 0.0              # Distortionless regularizer (optional)
    lambda_mv: float = 1e-4             # Min-variance regularizer (recommended: 1e-4)
    deep_sup_weight: float = 0.2        # Deep supervision loss weight
    lambda_orth: float = 0.0            # Orthogonality regularizer (B^K ⊥ T^K)
    lambda_sparse: float = 0.0          # Sparsity regularizer on T^K
    mask_mvdr_with_bg: bool = True      # Apply MVDR regularizers only on background pixels
    # GSFANet-style loss parameters
    iou_weight: float = 1.0             # Weight for SoftIoU term
    # Legacy parameters (kept for backward compatibility)
    focal_tversky_weight: float = 0.0   # Deprecated: use AdaFocalLoss instead
    bce_weight: float = 1.0             # Deprecated
    bce_deep_sup_weight: float = 0.3    # Deprecated
    use_pos_weight: bool = False        # Deprecated
    pos_weight_max: float = 200.0       # Deprecated
    tversky_alpha: float = 0.7          # Deprecated
    tversky_beta: float = 0.3           # Deprecated
    tversky_gamma: float = 4/3          # Deprecated


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft (differentiable) IoU loss (legacy function).
    
    Args:
        pred: Predicted mask ∈ ℝ[B, 1, H, W], values in [0, 1]
        target: Ground truth mask ∈ ℝ[B, 1, H, W], values in {0, 1}
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss: 1 - IoU (mean over batch)
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.shape[0], -1)      # [B, H*W]
    target_flat = target.view(target.shape[0], -1)  # [B, H*W]
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)  # [B]
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection  # [B]
    
    # IoU per sample
    iou = (intersection + eps) / (union + eps)  # [B]
    
    # Return 1 - IoU (loss to minimize)
    return 1.0 - iou.mean()


def fidelity_loss(D_k: torch.Tensor, D_0: torch.Tensor) -> torch.Tensor:
    """
    Fidelity loss: MSE between final reconstructed image and input.
    
    Args:
        D_k: Final reconstructed image ∈ ℝ[B, 1, H, W]
        D_0: Input image ∈ ℝ[B, 1, H, W]
    
    Returns:
        Scalar MSE loss
    """
    # Handle size mismatch: resize D_k to match D_0
    if D_k.shape[2:] != D_0.shape[2:]:
        D_k = F.interpolate(D_k, size=D_0.shape[2:], mode='bilinear', align_corners=False)
    return F.mse_loss(D_k, D_0)


def mvdr_regularizers(
    stage_aux: List[Dict[str, torch.Tensor]],
    dl_weight: float,
    mv_weight: float,
    mask_bg: torch.Tensor = None
) -> torch.Tensor:
    """
    Optional MVDR regularizers for interpretability.
    
    Encourages Capon attention to maintain:
        - Distortionless (DL): unit gain on steering direction (y_capon ≈ 1)
        - Min-Variance (MV): minimize output variance
    
    Args:
        stage_aux: List of K auxiliary dicts from stages, each containing:
            - y_capon: ℝ[B, 1, H, W]
            - v_capon: ℝ[B, 1, H, W]
        dl_weight: Weight for distortionless regularizer (lambda_dl)
        mv_weight: Weight for min-variance regularizer (lambda_mv)
        mask_bg: Optional background mask (1-target) ℝ[B, 1, H, W] to apply regularizers
                 only on background pixels (recommended to avoid suppressing targets)
    
    Returns:
        Scalar regularization loss
    """
    if dl_weight == 0.0 and mv_weight == 0.0:
        # Return zero tensor on correct device
        if len(stage_aux) > 0 and "y_capon" in stage_aux[0]:
            device = stage_aux[0]["y_capon"].device
        else:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)
    
    L_dl = 0.0
    L_mv = 0.0
    K = len(stage_aux)
    
    for aux_k in stage_aux:
        y_capon = aux_k["y_capon"]  # [B, 1, H, W]
        v_capon = aux_k["v_capon"]  # [B, 1, H, W]
        
        # Apply background mask if provided (Q4: only regularize background)
        if mask_bg is not None:
            # Upsample Capon outputs to match mask resolution if needed
            if y_capon.shape != mask_bg.shape:
                y_capon = F.interpolate(y_capon, size=mask_bg.shape[2:], mode='bilinear', align_corners=False)
                v_capon = F.interpolate(v_capon, size=mask_bg.shape[2:], mode='bilinear', align_corners=False)
            
            y_capon = y_capon * mask_bg
            v_capon = v_capon * mask_bg
            # Normalize by number of background pixels
            norm_factor = mask_bg.sum() + 1e-6
        else:
            norm_factor = y_capon.numel()
        
        if dl_weight > 0.0:
            # Distortionless: encourage y_capon ≈ 1 (or close to positive constant)
            # We use MSE to a target of 1.0
            L_dl += ((y_capon - 1.0) ** 2).sum() / norm_factor
        
        if mv_weight > 0.0:
            # Min-variance: minimize output variance
            L_mv += v_capon.sum() / norm_factor
    
    # Average over stages
    L_dl = L_dl / K if K > 0 else 0.0
    L_mv = L_mv / K if K > 0 else 0.0
    
    return dl_weight * L_dl + mv_weight * L_mv


def orthogonality_loss(B: torch.Tensor, T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Orthogonality regularizer: encourage B^K and T^K to be decorrelated.
    
    Computes cosine similarity and penalizes it to push towards orthogonality.
    
    Args:
        B: Background estimate ℝ[B, 1, H, W]
        T: Target estimate ℝ[B, 1, H, W]
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss (mean absolute cosine similarity)
    """
    # Flatten spatial dimensions
    B_flat = B.view(B.shape[0], -1)  # [B, H*W]
    T_flat = T.view(T.shape[0], -1)  # [B, H*W]
    
    # Compute cosine similarity per sample
    B_norm = torch.norm(B_flat, p=2, dim=1, keepdim=True) + eps
    T_norm = torch.norm(T_flat, p=2, dim=1, keepdim=True) + eps
    
    cos_sim = (B_flat * T_flat).sum(dim=1) / (B_norm.squeeze() * T_norm.squeeze())
    
    # Penalize absolute cosine similarity (want orthogonal = 0)
    return torch.abs(cos_sim).mean()


def sparsity_loss(T: torch.Tensor, target_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Sparsity regularizer: encourage T^K to be compact/sparse.
    
    Uses L1 norm with optional focal weighting on positive pixels.
    
    Args:
        T: Target estimate ℝ[B, 1, H, W]
        target_mask: Optional ground truth mask ℝ[B, 1, H, W] to weight positives more
    
    Returns:
        Scalar loss (weighted L1 norm)
    """
    if target_mask is not None:
        # Focal sparsity: weight positive pixels more (encourage compactness on targets)
        # Use 2x weight on positive pixels, 1x on background
        weight = 1.0 + target_mask
        return (torch.abs(T) * weight).mean()
    else:
        # Standard L1
        return torch.abs(T).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for CGL-RPCANet training with GSFANet-style losses.
    
    Total loss = AdaFocal + iou_weight * SoftIoU
               + deep_sup_weight * mean_k(AdaFocal_k + iou_weight * SoftIoU_k) [if stage_logits exist]
               + eta_fidelity * L_fid
               + MVDR regularizers (lambda_dl * L_dl + lambda_mv * L_mv)
               + Other regularizers (lambda_orth * L_orth + lambda_sparse * L_sparse)
    
    Args:
        weights: LossWeights dataclass with loss component weights
    """
    
    def __init__(self, weights: LossWeights):
        super().__init__()
        self.weights = weights
        # GSFANet-style losses
        self.ada_focal = AdaFocalLoss()
        self.soft_iou = SoftIoULoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        input_img: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Network output dict with keys:
                - mask: ℝ[B, 1, H, W] (sigmoid applied)
                - mask_logit: ℝ[B, 1, H, W] (optional, pre-sigmoid logits)
                - stage_logits: list of ℝ[B, 1, H, W] (optional, per-stage logits for deep-sup)
                - B: ℝ[B, 1, H, W] (background)
                - T: ℝ[B, 1, H, W] (target)
                - D: ℝ[B, 1, H, W] (reconstruction)
                - aux: list of stage aux dicts
            target: Ground truth mask ℝ[B, 1, H, W]
            input_img: Input image ℝ[B, 1, H, W]
        
        Returns:
            Dictionary with:
                - loss: Total scalar loss
                - loss_ada_focal: AdaFocal loss on main head
                - loss_iou_main: IoU loss on main head
                - loss_fidelity: Fidelity loss component
                - loss_deep_sup: Deep supervision loss
                - loss_mvdr: MVDR regularization component
                - loss_orth: Orthogonality regularization component
                - loss_sparse: Sparsity regularization component
        """
        eps = 1e-6
        device = outputs["mask"].device
        
        # ---------- Get logits ----------
        # Prefer 'mask_logit' (pre-sigmoid). Fallback to logit(p) if only 'mask' exists.
        final_logit = outputs.get("mask_logit", None)
        if final_logit is None and "mask" in outputs:
            p = outputs["mask"].clamp(eps, 1 - eps)
            final_logit = torch.log(p) - torch.log(1 - p)
        
        stage_logits: List[torch.Tensor] = outputs.get("stage_logits", [])
        
        # ---------- Main head: AdaFocal + SoftIoU (GSFANet-style) ----------
        L_ada_focal = self.ada_focal(final_logit, target)
        L_iou_main = self.soft_iou(final_logit, target)
        
        L_main = L_ada_focal + self.weights.iou_weight * L_iou_main
        
        # ---------- Fidelity loss ----------
        L_fid = fidelity_loss(outputs["D"], input_img)
        
        # ---------- Deep supervision over stage_logits (GSFANet-style) ----------
        L_deep_sup = torch.tensor(0.0, device=device)
        if len(stage_logits) > 0 and self.weights.deep_sup_weight > 0:
            terms = []
            for logit_k in stage_logits:
                L_ada_k = self.ada_focal(logit_k, target)
                L_iou_k = self.soft_iou(logit_k, target)
                terms.append(L_ada_k + self.weights.iou_weight * L_iou_k)
            L_deep_sup = self.weights.deep_sup_weight * (torch.stack(terms).mean())
        
        # ---------- MVDR regularizers ----------
        mask_bg = (1.0 - target) if self.weights.mask_mvdr_with_bg else None
        L_mvdr = mvdr_regularizers(
            outputs["aux"],
            self.weights.lambda_dl,
            self.weights.lambda_mv,
            mask_bg=mask_bg
        )
        
        # ---------- Orthogonality regularizer ----------
        L_orth = torch.tensor(0.0, device=device)
        if self.weights.lambda_orth > 0.0:
            L_orth = orthogonality_loss(outputs["B"], outputs["T"])
        
        # ---------- Sparsity regularizer ----------
        L_sparse = torch.tensor(0.0, device=device)
        if self.weights.lambda_sparse > 0.0:
            L_sparse = sparsity_loss(outputs["T"], target_mask=target)
        
        # ---------- Total loss ----------
        total_loss = (
            L_main
            + self.weights.eta_fidelity * L_fid 
            + L_deep_sup
            + L_mvdr
            + self.weights.lambda_orth * L_orth
            + self.weights.lambda_sparse * L_sparse
        )
        
        return {
            "loss": total_loss,
            "loss_ada_focal": L_ada_focal.detach(),
            "loss_iou_main": L_iou_main.detach(),
            "loss_fidelity": L_fid.detach(),
            "loss_deep_sup": L_deep_sup.detach(),
            "loss_mvdr": L_mvdr.detach(),
            "loss_orth": L_orth.detach(),
            "loss_sparse": L_sparse.detach(),
            # Legacy compatibility
            "loss_bce_main": L_ada_focal.detach(),  # Map ada_focal to bce_main for compatibility
            "loss_focal_tversky": torch.tensor(0.0, device=device),
            "loss_bce_deep_sup": L_deep_sup.detach()
        }


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
