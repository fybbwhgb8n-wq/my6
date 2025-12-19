"""
RCBANet: RCBA-guided UNet-like Multi-Scale Network
Fuses deep unfolding view (RCBA updates) with UNet view (multi-scale encoder-decoder).

Architecture:
- EncoderPyramid: 3-level feature pyramid (1/1, 1/2, 1/4)
- 3 Stages: each stage operates at a specific scale with:
  * CAU (Capon Attention Unit) at that scale
  * RCBA updates (B, T, N)
  * Micro-decoder for UNet-like upsampling
- FusionHead: cross-stage fusion of last 2-3 target maps

Reference: GPT design (Nov 2025) - scale-aligned unfolding with UNet expressivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .modules import ProxNetB, GradNet, ReconNet, ResidualBottleneck
from .capon import MultiScaleCaponAttention


@dataclass
class RCBANetConfig:
    """Configuration for RCBANet."""
    # Encoder pyramid channels
    C1: int = 32              # Full resolution (1/1)
    C2: int = 48              # Half resolution (1/2)
    C3: int = 64              # Quarter resolution (1/4)
    
    # CAU configuration per scale
    cau_ch_s1: int = 24       # CAU channels at scale 1 (1/1)
    cau_ch_s2: int = 16       # CAU channels at scale 2 (1/2)
    cau_ch_s3: int = 16       # CAU channels at scale 3 (1/4)
    cau_kernels: List[int] = None  # Multi-scale kernels (default: [3, 5])
    cau_strides: List[int] = None  # Per-scale strides (default: [2, 1, 1] for [s3, s2, s1])
    
    # Prox/Grad network configuration per scale
    base_ch_s1: int = 32      # Base channels at scale 1
    base_ch_s2: int = 32      # Base channels at scale 2
    base_ch_s3: int = 24      # Base channels at scale 3
    mid_ch_s1: int = 16       # Bottleneck channels at scale 1
    mid_ch_s2: int = 16       # Bottleneck channels at scale 2
    mid_ch_s3: int = 12       # Bottleneck channels at scale 3
    prox_depth: int = 2       # Depth of ProxNetB
    grad_depth: int = 2       # Depth of GradNet
    
    # Training configuration
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing in CAU
    deep_supervision: bool = True             # Enable deep supervision on stage 2 & 3
    deep_sup_weight: float = 0.3              # Weight for deep supervision losses
    
    def __post_init__(self):
        if self.cau_kernels is None:
            self.cau_kernels = [3, 5]
        if self.cau_strides is None:
            self.cau_strides = [2, 1, 1]  # [s3, s2, s1]: coarse→mid→fine


class EncoderPyramid(nn.Module):
    """
    3-level feature pyramid encoder.
    
    Outputs:
        F1: [B, C1, H, W]     (1/1 resolution)
        F2: [B, C2, H/2, W/2] (1/2 resolution)
        F3: [B, C3, H/4, W/4] (1/4 resolution)
    
    Args:
        C1, C2, C3: Channel dimensions at each level
    """
    
    def __init__(self, C1: int = 32, C2: int = 48, C3: int = 64):
        super().__init__()
        
        # Level 1 (1/1): Input → C1
        self.E1 = nn.Sequential(
            nn.Conv2d(1, C1 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C1 // 2),
            nn.ReLU(inplace=True),
            ResidualBottleneck(C1 // 2, C1 // 4, C1, use_bn=True, use_se=True),
            ResidualBottleneck(C1, C1 // 2, C1, use_bn=True, use_se=True)
        )
        
        # Level 2 (1/2): C1 → C2 with downsampling
        self.down1 = nn.Conv2d(C1, C2, kernel_size=3, stride=2, padding=1, bias=False)
        self.E2 = nn.Sequential(
            nn.BatchNorm2d(C2),
            nn.ReLU(inplace=True),
            ResidualBottleneck(C2, C2 // 2, C2, use_bn=True, use_se=True),
            ResidualBottleneck(C2, C2 // 2, C2, use_bn=True, use_se=True)
        )
        
        # Level 3 (1/4): C2 → C3 with downsampling
        self.down2 = nn.Conv2d(C2, C3, kernel_size=3, stride=2, padding=1, bias=False)
        self.E3 = nn.Sequential(
            nn.BatchNorm2d(C3),
            nn.ReLU(inplace=True),
            ResidualBottleneck(C3, C3 // 2, C3, use_bn=True, use_se=True),
            ResidualBottleneck(C3, C3 // 2, C3, use_bn=True, use_se=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image [B, 1, H, W]
        
        Returns:
            F1: Feature map at 1/1 [B, C1, H, W]
            F2: Feature map at 1/2 [B, C2, H/2, W/2]
            F3: Feature map at 1/4 [B, C3, H/4, W/4]
        """
        # Level 1 (1/1)
        F1 = self.E1(x)
        
        # Level 2 (1/2)
        x2 = self.down1(F1)
        F2 = self.E2(x2)
        
        # Level 3 (1/4)
        x3 = self.down2(F2)
        F3 = self.E3(x3)
        
        return F1, F2, F3


class MicroDecoder(nn.Module):
    """
    Micro-decoder for UNet-like upsampling with skip connections.
    
    Takes target estimate T^k_s and encoder feature F_s, fuses them,
    then upsamples to the next finer scale.
    
    Args:
        T_ch: Channels of target estimate (always 1)
        F_ch: Channels of encoder feature
        out_ch: Output channels (always 1)
        scale_factor: Upsampling scale factor (default: 2)
    """
    
    def __init__(self, T_ch: int = 1, F_ch: int = 32, out_ch: int = 1, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Fusion: concatenate [T, F] → reduce → refine
        self.reduce = nn.Conv2d(T_ch + F_ch, 16, kernel_size=1, bias=True)
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, kernel_size=1, bias=True)
        )
    
    def forward(self, T: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: Target estimate [B, 1, H, W]
            feat: Encoder feature [B, F_ch, H, W]
        
        Returns:
            T_up: Upsampled and refined target [B, 1, H*scale, W*scale]
        """
        # Concatenate
        x = torch.cat([T, feat], dim=1)  # [B, 1+F_ch, H, W]
        
        # Reduce and refine
        x = self.reduce(x)
        x = self.refine(x)
        
        # Upsample
        if self.scale_factor > 1:
            T_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            T_up = x
        
        return T_up


class StageOp(nn.Module):
    """
    One stage of scale-aligned RCBA unfolding.
    
    Operates at a specific scale (s ∈ {3, 2, 1}) with:
    1. CAU stem + Capon Attention at scale s
    2. Background update (SEBEM)
    3. Target update (SETEM) with target-aware shrinkage
    4. Noise update (SENRM) with variance gating
    5. Reconstruction (SEIRM)
    6. Micro-decoder (UNet-like upsampling to next finer scale)
    
    Args:
        scale: Scale index (1=full, 2=half, 3=quarter)
        cau_ch: CAU feature channels
        cau_kernels: List of kernel sizes for multi-scale CAU
        cau_stride: CAU tile stride
        base_ch: Base channel width for Prox/Grad
        mid_ch: Bottleneck middle channels
        prox_depth: Depth of ProxNetB
        grad_depth: Depth of GradNet
        enc_ch: Encoder feature channels at this scale
        use_gradient_checkpointing: Enable gradient checkpointing in CAU
    """
    
    def __init__(
        self,
        scale: int,
        cau_ch: int,
        cau_kernels: List[int],
        cau_stride: int,
        base_ch: int,
        mid_ch: int,
        prox_depth: int,
        grad_depth: int,
        enc_ch: int,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.scale = scale
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # CAU stem: 1 → cau_ch (residual-driven)
        self.cau_stem = nn.Sequential(
            nn.Conv2d(1, cau_ch // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(cau_ch // 2, cau_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Encoder-to-CAU projection (to fuse encoder features with residual)
        self.enc2cau = nn.Conv2d(enc_ch, cau_ch, kernel_size=1, bias=False)
        
        # Multi-scale Capon attention
        self.cau = MultiScaleCaponAttention(
            channels=cau_ch,
            kernels=cau_kernels,
            stride=cau_stride,
            shrink_alpha=0.2,
            diag_delta=3e-3,
            use_psf_head=True
        )
        
        # Background proximal net (SEBEM)
        self.proxB = ProxNetB(in_ch=1, base_ch=base_ch, mid_ch=mid_ch, depth=prox_depth)
        
        # Target gradient net (SETEM) - with dilation
        self.gradT = GradNet(in_ch=1, base_ch=base_ch, mid_ch=mid_ch, depth=grad_depth, use_dilation=True)
        
        # Noise gradient net (SENRM) - with dilation
        self.gradN = GradNet(in_ch=1, base_ch=base_ch, mid_ch=mid_ch, depth=grad_depth, use_dilation=True)
        
        # Reconstruction net (SEIRM)
        self.recon = ReconNet(in_ch=1, base_ch=base_ch, depth=3)
        
        # Micro-decoder (now also used for finest scale, but without upsampling)
        # All scales use Micro-Decoder to fuse encoder features
        if scale > 1:
            # Coarse/mid scales: upsample by 2×
            self.micro_decoder = MicroDecoder(T_ch=1, F_ch=enc_ch, out_ch=1, scale_factor=2)
        else:
            # Finest scale: no upsampling, but still fuse encoder features
            self.micro_decoder = MicroDecoder(T_ch=1, F_ch=enc_ch, out_ch=1, scale_factor=1)
        
        # Learnable step sizes and gating parameters (raw, will be bounded)
        self.eps_raw = nn.Parameter(torch.tensor(0.0))     # → ε ∈ [0.08, 0.15]
        self.sigma_raw = nn.Parameter(torch.tensor(0.0))   # → σ ∈ [0.10, 0.18]
        self.bT_raw = nn.Parameter(torch.tensor(0.0))      # → β_T ∈ [0.3, 0.5]
        self.bB_raw = nn.Parameter(torch.tensor(0.0))      # → β_B ∈ [0.4, 0.6]
        self.gV_raw = nn.Parameter(torch.tensor(0.0))      # → γ_V ∈ [0.2, 0.4]
        self.kappa_raw = nn.Parameter(torch.tensor(0.0))   # → κ ∈ [4, 8]
    
    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        enc_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            state: (D_prev, B_prev, T_prev, N_prev) at this scale
            enc_feat: Encoder feature F_s at this scale
        
        Returns:
            B_k: Updated background [B, 1, H, W]
            T_k: Updated target [B, 1, H, W]
            N_k: Updated noise [B, 1, H, W]
            D_k: Updated reconstruction [B, 1, H, W]
            T_up: Upsampled target (if micro-decoder exists, else None)
            aux: dict with y_capon, v_capon, sigma_eff
        """
        D_prev, B_prev, T_prev, N_prev = state
        
        # Bounded parameters
        epsilon = F.softplus(self.eps_raw) * 0.07 + 0.08      # → [0.08, 0.15]
        sigma = F.softplus(self.sigma_raw) * 0.08 + 0.10      # → [0.10, 0.18]
        beta_T = torch.sigmoid(self.bT_raw) * 0.2 + 0.3       # → [0.3, 0.5]
        beta_B = torch.sigmoid(self.bB_raw) * 0.2 + 0.4       # → [0.4, 0.6]
        gamma_V = torch.sigmoid(self.gV_raw) * 0.2 + 0.2      # → [0.2, 0.4]
        kappa = F.softplus(self.kappa_raw) * 2.0 + 4.0        # → [4, 8]
        
        # Compute residual for CAU
        X = D_prev - T_prev - N_prev  # [B, 1, H, W]
        
        # CAU feature extraction: use residual only (physics-first)
        # This maintains the RCBA/Capon interpretability by only feeding the residual to CAU
        U_res = self.cau_stem(X)        # Residual-driven features
        U = U_res  # Residual-only (better Capon interpretability)
        
        # Use gradient checkpointing to save memory
        if self.use_gradient_checkpointing and self.training:
            y_capon, v_capon = checkpoint(self.cau, U, use_reentrant=False)
        else:
            y_capon, v_capon = self.cau(U)  # [B, 1, H, W] each
        
        # Stabilize Capon output (per-sample normalization)
        B_size = y_capon.shape[0]
        y_mean = y_capon.view(B_size, -1).mean(dim=1, keepdim=True).view(B_size, 1, 1, 1)
        y_std = y_capon.view(B_size, -1).std(dim=1, keepdim=True, unbiased=False).view(B_size, 1, 1, 1)
        y_capon = (y_capon - y_mean) / (y_std + 1e-6)
        y_capon = torch.clamp(y_capon, -3.0, 3.0)  # Avoid extreme boosts/nulls
        
        # --- Background update (SEBEM): target nulling ---
        b_in = X - beta_B * y_capon
        B_k = self.proxB(b_in)  # [B, 1, H, W]
        
        # --- Target update (SETEM) with target-aware shrinkage ---
        xT = T_prev + (D_prev - B_k - N_prev)
        
        # Target-aware shrinkage: reduce gradient suppression where Capon is high
        gate_T = torch.sigmoid(kappa * y_capon)
        gradT_map = self.gradT(xT) * (1.0 - 0.5 * gate_T)  # Preserve bright spots
        
        T_k = xT - epsilon * gradT_map + beta_T * y_capon
        
        # --- Noise update (SENRM): variance-gated denoising ---
        xN = N_prev + (D_prev - B_k - T_k)
        
        # Normalize variance per sample for gating
        B_size = v_capon.shape[0]
        v_mean = v_capon.view(B_size, -1).mean(dim=1, keepdim=True).view(B_size, 1, 1, 1)
        v_std = v_capon.view(B_size, -1).std(dim=1, keepdim=True).view(B_size, 1, 1, 1)
        v_norm = (v_capon - v_mean) / (v_std + 1e-6)
        
        # Variance-gated sigma: higher variance → stronger denoising
        sigma_eff = sigma * (1.0 + gamma_V * torch.tanh(v_norm))
        
        gradN_map = self.gradN(xN)  # [B, 1, H, W]
        N_k = xN - sigma_eff * gradN_map
        
        # --- Reconstruction (SEIRM) ---
        D_k = self.recon(B_k + T_k + N_k)  # [B, 1, H, W]
        
        # --- Micro-decoder (UNet-like upsampling) ---
        if self.micro_decoder is not None:
            T_up = self.micro_decoder(T_k, enc_feat)
        else:
            T_up = None
        
        # Auxiliary outputs
        # Note: y_capon is NOT detached to allow gradients to flow through CAU via fusion head
        # This makes the Capon-Target fusion head fully learnable. Keep v_capon and sigma_eff detached.
        aux = {
            "y_capon": y_capon,  # Allow gradients through CAU
            "v_capon": v_capon.detach(),  # Keep detached for stability
            "sigma_eff": sigma_eff.detach()
        }
        
        return B_k, T_k, N_k, D_k, T_up, aux


class FusionHead(nn.Module):
    """
    Cross-stage fusion head for combining last 2-3 target maps.
    
    Uses multi-scale convolutions (3×3, 5×5) followed by 1×1 projection.
    
    Args:
        in_ch: Number of input target maps (2 or 3)
        ms_kernels: Multi-scale kernel sizes (default: [3, 5])
    """
    
    def __init__(self, in_ch: int, ms_kernels: List[int] = None):
        super().__init__()
        if ms_kernels is None:
            ms_kernels = [3, 5]
        
        # Add y_capon to the fusion (hence in_ch + 1)
        total_in = in_ch + 1
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(total_in, 8, kernel_size=k, padding=k // 2, bias=True)
            for k in ms_kernels
        ])
        
        # Fusion and projection
        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * len(ms_kernels), 8, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True)
        )
    
    def forward(self, T_list: List[torch.Tensor], y_capon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T_list: List of target estimates (last 2-3 stages) [B, 1, H, W] each
            y_capon: Final stage Capon response [B, 1, H, W]
        
        Returns:
            mask_logit: Fused mask logits [B, 1, H, W]
        """
        # Concatenate target maps and Capon response
        x = torch.cat(T_list + [y_capon], dim=1)  # [B, in_ch+1, H, W]
        
        # Multi-scale convolutions
        feats = [conv(x) for conv in self.convs]
        
        # Concatenate and fuse
        x = torch.cat(feats, dim=1)  # [B, 8*len(ms_kernels), H, W]
        mask_logit = self.fusion(x)  # [B, 1, H, W]
        
        return mask_logit


class RCBANet(nn.Module):
    """
    RCBANet: RCBA-guided UNet-like Multi-Scale Network.
    
    Combines deep unfolding (RCBA updates) with UNet (multi-scale encoder-decoder).
    
    Architecture:
        - EncoderPyramid: 3-level feature pyramid (1/1, 1/2, 1/4)
        - Stage 3 @ 1/4: Coarse-scale RCBA update + micro-decoder → 1/2
        - Stage 2 @ 1/2: Mid-scale RCBA update + micro-decoder → 1/1
        - Stage 1 @ 1/1: Fine-scale RCBA update
        - FusionHead: Cross-stage fusion of last 2-3 target maps
    
    Args:
        cfg: RCBANetConfig with hyperparameters
    
    Forward:
        Input:  x ∈ ℝ[B, 1, H, W]  (single IR frame, normalized to [0,1])
        Output: dict with:
            - mask: ℝ[B, 1, H, W] in [0,1] (target probability)
            - B, T, N, D: ℝ[B, 1, H, W] (final component estimates)
            - aux: list of 3 aux dicts from each stage
            - intermediate_masks: list of intermediate masks for deep supervision
    """
    
    def __init__(self, cfg: RCBANetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Encoder pyramid
        self.encoder = EncoderPyramid(C1=cfg.C1, C2=cfg.C2, C3=cfg.C3)
        
        # Stage 3 @ 1/4 (coarse)
        self.stage3 = StageOp(
            scale=3,
            cau_ch=cfg.cau_ch_s3,
            cau_kernels=[3],  # Coarse scale: single large kernel
            cau_stride=cfg.cau_strides[0],  # Typically 2
            base_ch=cfg.base_ch_s3,
            mid_ch=cfg.mid_ch_s3,
            prox_depth=cfg.prox_depth,
            grad_depth=cfg.grad_depth,
            enc_ch=cfg.C3,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing
        )
        
        # Stage 2 @ 1/2 (mid)
        self.stage2 = StageOp(
            scale=2,
            cau_ch=cfg.cau_ch_s2,
            cau_kernels=cfg.cau_kernels,  # Multi-scale [3, 5]
            cau_stride=cfg.cau_strides[1],  # Typically 1
            base_ch=cfg.base_ch_s2,
            mid_ch=cfg.mid_ch_s2,
            prox_depth=cfg.prox_depth,
            grad_depth=cfg.grad_depth,
            enc_ch=cfg.C2,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing
        )
        
        # Stage 1 @ 1/1 (fine)
        self.stage1 = StageOp(
            scale=1,
            cau_ch=cfg.cau_ch_s1,
            cau_kernels=cfg.cau_kernels,  # Multi-scale [3, 5]
            cau_stride=cfg.cau_strides[2],  # Typically 1
            base_ch=cfg.base_ch_s1,
            mid_ch=cfg.mid_ch_s1,
            prox_depth=cfg.prox_depth,
            grad_depth=cfg.grad_depth,
            enc_ch=cfg.C1,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing
        )
        
        # Cross-stage fusion head (fuses last 2 target maps + final Capon)
        self.fusion_head = FusionHead(in_ch=2, ms_kernels=[3, 5])
        
        # Stage-wise segmentation heads for BCE+IoU deep supervision
        self.stage_heads = nn.ModuleDict({
            's3': nn.Conv2d(1, 1, kernel_size=1, bias=True),
            's2': nn.Conv2d(1, 1, kernel_size=1, bias=True),
            's1': nn.Conv2d(1, 1, kernel_size=1, bias=True),
        })
        
        # FA gate: learnable variance-based false alarm suppression
        self.tau_raw = nn.Parameter(torch.tensor(0.0))  # τ ∈ [0, 0.5]
        
        # Deep supervision: No separate heads needed, we'll use logits directly
        # The loss function will handle converting T2, T3 logits to masks
        self.use_deep_supervision = cfg.deep_supervision
    
    def _downsample_state(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        scale: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Downsample (D, B, T, N) to target scale with anti-aliasing."""
        D, B, T, N = state
        if scale == 1:
            return D, B, T, N
        
        # Use anti-aliasing to preserve tiny targets when downsampling
        size = (D.shape[-2] // scale, D.shape[-1] // scale)
        D_down = F.interpolate(D, size=size, mode='bilinear', align_corners=False, antialias=True)
        B_down = F.interpolate(B, size=size, mode='bilinear', align_corners=False, antialias=True)
        T_down = F.interpolate(T, size=size, mode='bilinear', align_corners=False, antialias=True)
        N_down = F.interpolate(N, size=size, mode='bilinear', align_corners=False, antialias=True)
        
        return D_down, B_down, T_down, N_down
    
    def _upsample_state(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        target_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upsample (D, B, T, N) to target size with anti-aliasing."""
        D, B, T, N = state
        
        D_up = F.interpolate(D, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        B_up = F.interpolate(B, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        T_up = F.interpolate(T, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        N_up = F.interpolate(N, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        
        return D_up, B_up, T_up, N_up
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input IR image [B, 1, H, W], normalized to [0,1]
        
        Returns:
            Dictionary with:
                - mask: Target probability map [B, 1, H, W]
                - B, T, N, D: Final component estimates [B, 1, H, W]
                - aux: List of per-stage auxiliary outputs
                - intermediate_masks: List of intermediate masks for deep supervision (if enabled)
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected single-channel input, got {C} channels"
        
        # Encoder pyramid
        F1, F2, F3 = self.encoder(x)  # 1/1, 1/2, 1/4
        
        # Initialize state at full resolution
        D = x
        B_prev = torch.zeros_like(x)
        T_prev = torch.zeros_like(x)
        N_prev = torch.zeros_like(x)
        
        aux_list = []
        intermediate_masks = []
        stage_logits = []  # Collect per-stage logits for BCE+IoU deep supervision
        T_bank = []  # Collect target estimates for cross-stage fusion
        
        # === Stage 3 @ 1/4 (coarse) ===
        H3, W3 = H // 4, W // 4
        state3 = self._downsample_state((D, B_prev, T_prev, N_prev), scale=4)
        
        B3, T3, N3, D3, T3_up, aux3 = self.stage3(state3, F3)
        aux_list.append(aux3)
        
        # Collect stage 3 logit for BCE+IoU deep supervision
        logit3 = self.stage_heads['s3'](T3)
        logit3_full = F.interpolate(logit3, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
        stage_logits.append(logit3_full)
        
        # Deep supervision at stage 3 (use logits, not Sigmoid)
        if self.use_deep_supervision:
            # Upsample T3 to full res for supervision
            T3_full = F.interpolate(T3, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
            intermediate_masks.append(T3_full)  # Return logits for DS loss
        
        # === Stage 2 @ 1/2 (mid) ===
        H2, W2 = H // 2, W // 2
        
        # Use micro-decoder output T3_up as seed for T, upsample others
        D2_up, B2_up, _, N2_up = self._upsample_state((D3, B3, T3, N3), target_size=(H2, W2))
        # Use T3_up (skip-fused) if available, otherwise fallback to bilinear
        if T3_up is not None:
            T2_seed = T3_up
        else:
            T2_seed = F.interpolate(T3, size=(H2, W2), mode='bilinear', align_corners=False, antialias=True)
        state2_from_3 = (D2_up, B2_up, T2_seed, N2_up)
        
        B2, T2, N2, D2, T2_up, aux2 = self.stage2(state2_from_3, F2)
        aux_list.append(aux2)
        T_bank.append(T2)  # Save for fusion
        
        # Collect stage 2 logit for BCE+IoU deep supervision
        logit2 = self.stage_heads['s2'](T2)
        logit2_full = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
        stage_logits.append(logit2_full)
        
        # Deep supervision at stage 2 (use logits, not Sigmoid)
        if self.use_deep_supervision:
            # Upsample T2 to full res for supervision
            T2_full = F.interpolate(T2, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
            intermediate_masks.append(T2_full)  # Return logits for DS loss
        
        # === Stage 1 @ 1/1 (fine) ===
        # Use micro-decoder output T2_up as seed for T, upsample others
        D1_up, B1_up, _, N1_up = self._upsample_state((D2, B2, T2, N2), target_size=(H, W))
        # Use T2_up (skip-fused) if available, otherwise fallback to bilinear
        if T2_up is not None:
            T1_seed = T2_up
        else:
            T1_seed = F.interpolate(T2, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
        state1_from_2 = (D1_up, B1_up, T1_seed, N1_up)
        
        B1, T1, N1, D1, T1_fused, aux1 = self.stage1(state1_from_2, F1)
        aux_list.append(aux1)
        # Use T1_fused (fused with F1) if available, otherwise use T1
        T_final = T1_fused if T1_fused is not None else T1
        T_bank.append(T_final)  # Save fused target for fusion
        
        # Collect stage 1 logit for BCE+IoU deep supervision
        logit1 = self.stage_heads['s1'](T1)  # Already full-res
        stage_logits.append(logit1)
        
        # === Cross-stage fusion ===
        # Upsample T2 to full res if needed
        if T_bank[0].shape[2:] != (H, W):
            T_bank[0] = F.interpolate(T_bank[0], size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse last 2 target maps with final Capon response
        mask_logit = self.fusion_head(T_bank, aux1['y_capon'])
        
        # === FA gate: variance-based false alarm suppression ===
        v_last = aux1['v_capon']  # MVDR variance from final stage
        if v_last.shape != T1.shape:
            v_last = F.interpolate(v_last, size=T1.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize variance per sample (z-score)
        v_mean = v_last.view(B, -1).mean(1, keepdim=True).view(B, 1, 1, 1)
        v_std = v_last.view(B, -1).std(1, keepdim=True).view(B, 1, 1, 1) + 1e-6
        v_norm = (v_last - v_mean) / v_std
        
        # Apply learnable variance-gated suppression (high variance → suppress FP)
        tau = F.softplus(self.tau_raw) * 0.5  # τ ∈ [0, 0.5]
        mask_logit = mask_logit - tau * torch.tanh(v_norm)  # Suppress clutter-driven false positives
        
        mask = torch.sigmoid(mask_logit)
        
        return {
            "mask": mask,
            "mask_logit": mask_logit,  # Pre-sigmoid logit for BCEWithLogitsLoss
            "stage_logits": stage_logits,  # Per-stage logits for deep supervision
            "B": B1,
            "T": T1,
            "N": N1,
            "D": D1,
            "aux": aux_list,
            "intermediate_masks": intermediate_masks
        }

