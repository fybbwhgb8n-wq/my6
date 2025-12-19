"""
Capon (MVDR) attention components for scene-adaptive feature reweighting.
Implements channel-wise minimum variance distortionless response beamforming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PSFHead(nn.Module):
    """
    Predicts a per-tile steering vector a ∈ ℝ^C from tile-wise channel statistics.
    
    Tiny MLP: Linear(C→H) + ReLU + Linear(H→C) + softplus; L2-normalize.
    
    Args:
        channels: Number of channels (C)
        hidden: Hidden layer size (default: max(8, C//2))
    
    Input:  μ ∈ ℝ[B*L, C]   # Mean of channels within each tile
    Output: a ∈ ℝ[B*L, C]   # Normalized, positive steering vector
    """
    
    def __init__(self, channels: int, hidden: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.hidden = hidden if hidden is not None else max(8, channels // 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, channels),
            nn.Softplus()  # Ensure positive entries
        )
    
    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: Channel means [B*L, C]
        Returns:
            Normalized steering vector [B*L, C]
        """
        a = self.mlp(mu)
        # L2 normalize to unit norm
        a = F.normalize(a, p=2, dim=-1, eps=1e-6)
        return a


class CaponAttentionUnit(nn.Module):
    """
    Channel-wise MVDR (Capon) attention over local spatial tiles.
    
    Computes adaptive beamforming weights from local channel covariance,
    yielding target-passing response and output variance maps.
    
    Args:
        channels: Number of channels in CAU feature space (Ccau)
        kernel_size: Spatial window size for covariance estimation (odd int, e.g., 5)
        stride: Stride for tile processing (e.g., 2)
        shrink_alpha: Covariance shrinkage parameter ∈ [0,1] (default: 0.10)
        diag_delta: Diagonal loading for stability (default: 1e-3)
        use_psf_head: If True, learn steering vector; if False, use a = 1_C normalized
    
    Input:
        U ∈ ℝ[B, C, H, W]  # CAU bottleneck feature map
    
    Output:
        y_up ∈ ℝ[B, 1, H, W]   # Target-passing Capon response (upsampled to H×W)
        v_up ∈ ℝ[B, 1, H, W]   # MVDR output variance (upsampled to H×W)
    
    Steps:
        1) Unfold U into tiles → X ∈ ℝ[B, L, C, M] where M = kernel_size²
        2) Compute empirical covariance R = (1/M) X X^T per tile
        3) Apply shrinkage & diagonal loading: R ← (1-α)R + α(tr(R)/C)I + δI
        4) Compute steering vector a (either via PSFHead or a = 1_C)
        5) Solve MVDR: R w_tilde = a; w = w_tilde / (a^T w_tilde)
        6) Capon response: y = w^T u_p
        7) Output variance: v = w^T R w
        8) Upsample (y, v) to original spatial size
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        shrink_alpha: float = 0.10,
        diag_delta: float = 1e-3,
        use_psf_head: bool = True,
        use_centered_cov: bool = True,
        use_center_locality: bool = True
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert 0.0 <= shrink_alpha <= 1.0, "shrink_alpha must be in [0,1]"
        assert diag_delta > 0, "diag_delta must be positive"
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.shrink_alpha = shrink_alpha
        self.diag_delta = diag_delta
        self.use_psf_head = use_psf_head
        self.use_centered_cov = use_centered_cov  # M2: centered (whitened) covariance
        self.use_center_locality = use_center_locality  # Q3: sharper locality
        
        self.padding = kernel_size // 2
        
        # PSF head for adaptive steering (optional)
        if use_psf_head:
            self.psf_head = PSFHead(channels)
        else:
            # Fixed steering: a = 1_C normalized
            self.register_buffer(
                'fixed_steering',
                torch.ones(1, channels) / (channels ** 0.5)
            )
    
    def forward(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            U: Feature map [B, C, H, W]
        
        Returns:
            y_up: Capon response [B, 1, H, W]
            v_up: Output variance [B, 1, H, W]
        """
        B, C, H, W = U.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"
        
        # 1) Unfold into tiles: [B, C, H, W] → [B, C*M, L]
        # where M = kernel_size² and L is number of tiles
        U_unfold = F.unfold(
            U,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )  # [B, C*M, L]
        
        M = self.kernel_size * self.kernel_size
        L = U_unfold.shape[2]
        
        # Reshape to [B, L, C, M]
        U_unfold = U_unfold.view(B, C, M, L).permute(0, 3, 1, 2)  # [B, L, C, M]
        
        # 2) Compute empirical channel covariance per tile: R = (1/M) X X^T
        # M2: Optional centered covariance (whitening) for sharper MVDR
        if self.use_centered_cov:
            # Center channels in each tile: X_c = X - mean(X)
            X_mean = U_unfold.mean(dim=-1, keepdim=True)  # [B, L, C, 1]
            X_centered = U_unfold - X_mean  # [B, L, C, M]
            R = torch.matmul(X_centered, X_centered.transpose(-2, -1)) / M  # [B, L, C, C]
        else:
            # Uncentered covariance (original)
            R = torch.matmul(U_unfold, U_unfold.transpose(-2, -1)) / M  # [B, L, C, C]
        
        # 3) Shrinkage covariance + diagonal loading
        # Compute trace for shrinkage target
        trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)  # [B, L, 1]
        identity_scale = trace_R / C  # [B, L, 1]
        
        eye_C = torch.eye(C, device=U.device, dtype=U.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, C, C]
        
        R_shrink = (1.0 - self.shrink_alpha) * R + \
                   self.shrink_alpha * identity_scale.unsqueeze(-1) * eye_C + \
                   self.diag_delta * eye_C
        # [B, L, C, C]
        
        # 4) Compute steering vector a
        if self.use_psf_head:
            # Compute mean channel vector per tile
            mu = U_unfold.mean(dim=-1)  # [B, L, C]
            mu_flat = mu.reshape(B * L, C)
            a_flat = self.psf_head(mu_flat)  # [B*L, C]
            a = a_flat.view(B, L, C)  # [B, L, C]
        else:
            # Fixed steering: a = 1_C normalized
            a = self.fixed_steering.unsqueeze(0).expand(B, L, -1)  # [B, L, C]
        
        # 5) Solve MVDR: R w_tilde = a via Cholesky decomposition
        # Cholesky: R = L L^T
        try:
            L = torch.linalg.cholesky(R_shrink)  # [B, L, C, C]
        except RuntimeError:
            # Fallback: add more diagonal loading if Cholesky fails
            R_shrink = R_shrink + 1e-2 * eye_C
            L = torch.linalg.cholesky(R_shrink)
        
        # Solve L y = a
        a_unsqueezed = a.unsqueeze(-1)  # [B, L, C, 1]
        y_tilde = torch.cholesky_solve(a_unsqueezed, L)  # [B, L, C, 1]
        w_tilde = y_tilde.squeeze(-1)  # [B, L, C]
        
        # Normalize: w = w_tilde / (a^T w_tilde)
        a_T_w = torch.sum(a * w_tilde, dim=-1, keepdim=True)  # [B, L, 1]
        a_T_w = torch.clamp(a_T_w, min=1e-6)  # Avoid division by zero
        w = w_tilde / a_T_w  # [B, L, C]
        
        # 6) Capon response: y = w^T u_p
        # Q3: Sharper locality - use central position vector instead of tile mean
        if self.use_center_locality:
            # Extract center position from each tile (more peaked response)
            center_idx = M // 2  # Center position in flattened tile
            u_p = U_unfold[:, :, :, center_idx]  # [B, L, C]
        else:
            # Original: use mean of the unfolded tile
            u_p = U_unfold.mean(dim=-1)  # [B, L, C]
        
        y = torch.sum(w * u_p, dim=-1, keepdim=True)  # [B, L, 1]
        
        # 7) Output variance: v = w^T R w
        w_unsqueezed = w.unsqueeze(-1)  # [B, L, C, 1]
        R_w = torch.matmul(R_shrink, w_unsqueezed)  # [B, L, C, 1]
        v = torch.matmul(w.unsqueeze(-2), R_w).squeeze(-1).squeeze(-1)  # [B, L]
        v = v.unsqueeze(-1)  # [B, L, 1]
        
        # 8) Reshape and upsample to original size
        # Compute output tile grid size
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        y = y.view(B, 1, H_out, W_out)  # [B, 1, H_out, W_out]
        v = v.view(B, 1, H_out, W_out)  # [B, 1, H_out, W_out]
        
        # Bilinear upsample to original size
        y_up = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        v_up = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)
        
        return y_up, v_up


class MultiScaleCaponAttention(nn.Module):
    """
    Multi-scale Capon attention with parallel kernels (k=3,5) for different target sizes.
    
    Combines outputs from two CAUs with different spatial scales using learnable local mixing.
    Instead of global scalar fusion (w*y3 + (1-w)*y5), uses a lightweight conv network
    over [y3, y5, v3, v5] to allow per-pixel scale selection based on local target size.
    
    Architecture:
        - Parallel CAUs with different kernels (e.g., k=3 for tiny, k=5 for larger)
        - Concatenate [y_k1, y_k2, ..., v_k1, v_k2, ...] → 2*num_scales channels
        - 1×1 pointwise conv → 8 channels (cross-scale mixing)
        - 3×3 depthwise conv → 8 channels (spatial context)
        - Separate 1×1 heads → y_fused, v_fused
    
    Benefits:
        - Per-pixel adaptive scale selection (tiny vs. slightly bigger targets)
        - Variance information (v) helps guide fusion decisions
        - Spatial context via depthwise conv
        - Minimal parameters (~200 total)
    
    Args:
        channels: Number of channels (default: 24)
        kernels: List of kernel sizes (default: [3, 5])
        stride: Stride for tile processing (default: 1)
        shrink_alpha: Covariance shrinkage (default: 0.2)
        diag_delta: Diagonal loading (default: 3e-3)
        use_psf_head: Whether to use PSF head (default: True)
    
    Input:  U ∈ ℝ[B, C, H, W]
    Output: (y, v) each ∈ ℝ[B, 1, H, W]
    """
    
    def __init__(
        self,
        channels: int = 24,
        kernels: List[int] = None,
        stride: int = 1,
        shrink_alpha: float = 0.2,
        diag_delta: float = 3e-3,
        use_psf_head: bool = True,
        use_centered_cov: bool = True,
        use_center_locality: bool = True
    ):
        super().__init__()
        if kernels is None:
            kernels = [3, 5]
        
        self.kernels = kernels
        
        # Create CAU for each scale
        self.caus = nn.ModuleList([
            CaponAttentionUnit(
                channels=channels,
                kernel_size=k,
                stride=stride,
                shrink_alpha=shrink_alpha,
                diag_delta=diag_delta,
                use_psf_head=use_psf_head,
                use_centered_cov=use_centered_cov,
                use_center_locality=use_center_locality
            )
            for k in kernels
        ])
        
        # Learnable local mixing: 1×1 conv over [y3, y5, v3, v5] + optional 3×3 depthwise
        # This allows per-pixel scale selection (tiny vs. slightly bigger targets)
        num_scales = len(kernels)
        if num_scales > 1:
            # Input: [y_k1, y_k2, ..., v_k1, v_k2, ...] = 2*num_scales channels
            fusion_in_ch = 2 * num_scales
            
            # 1×1 pointwise to mix scales
            self.fusion_pointwise = nn.Sequential(
                nn.Conv2d(fusion_in_ch, 8, kernel_size=1, bias=True),
                nn.ReLU(inplace=True)
            )
            
            # 3×3 depthwise for spatial context (optional but helps)
            self.fusion_depthwise = nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=True),
                nn.ReLU(inplace=True)
            )
            
            # Output heads: separate for y and v
            self.fusion_head_y = nn.Conv2d(8, 1, kernel_size=1, bias=True)
            self.fusion_head_v = nn.Conv2d(8, 1, kernel_size=1, bias=True)
        else:
            self.fusion_pointwise = None
            self.fusion_depthwise = None
            self.fusion_head_y = None
            self.fusion_head_v = None
    
    def forward(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            U: Feature map [B, C, H, W]
        
        Returns:
            y_fused: Fused Capon response [B, 1, H, W]
            v_fused: Fused variance [B, 1, H, W]
        """
        # Compute Capon attention at each scale
        outputs = [cau(U) for cau in self.caus]
        
        # Unpack
        y_scales = [out[0] for out in outputs]  # List of [B, 1, H, W]
        v_scales = [out[1] for out in outputs]  # List of [B, 1, H, W]
        
        # If only one scale, return it directly
        if len(y_scales) == 1:
            return y_scales[0], v_scales[0]
        
        # Learnable local mixing: concatenate [y_k1, y_k2, ..., v_k1, v_k2, ...]
        # For 2 scales: [y3, y5, v3, v5] → 4 channels
        x_fusion = torch.cat(y_scales + v_scales, dim=1)  # [B, 2*num_scales, H, W]
        
        # 1×1 pointwise mixing
        x = self.fusion_pointwise(x_fusion)  # [B, 8, H, W]
        
        # 3×3 depthwise for spatial context
        x = self.fusion_depthwise(x)  # [B, 8, H, W]
        
        # Separate output heads for y and v
        y_fused = self.fusion_head_y(x)  # [B, 1, H, W]
        v_fused = self.fusion_head_v(x)  # [B, 1, H, W]
        
        return y_fused, v_fused

