"""
Core neural network modules for CGL-RPCANet.
Includes SE blocks, bottleneck residuals, and proximal/gradient/reconstruction networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module for channel reweighting.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for squeeze operation (default: 8)
    
    Input:  x ∈ ℝ[B, C, H, W]
    Output: y ∈ ℝ[B, C, H, W]  (channel-wise recalibrated)
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        mid_ch = max(1, channels // reduction)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Channel-recalibrated tensor [B, C, H, W]
        """
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
        s = self.squeeze(x)
        # Excitation: [B, C, 1, 1] -> [B, C, 1, 1]
        scale = self.excitation(s)
        # Scale: element-wise multiplication
        return x * scale


class ResidualBottleneck(nn.Module):
    """
    Lightweight residual bottleneck block.
    
    Layout: 1x1 conv (in→mid) → 3x3 conv (mid) → 1x1 conv (mid→out) + residual.
    
    Args:
        in_ch: Input channels
        mid_ch: Bottleneck (middle) channels
        out_ch: Output channels
        use_bn: Whether to use BatchNorm (True for background prox; False for gradient nets)
        use_se: Whether to use SE module (default: True)
        dilation: Dilation rate for 3x3 conv (default: 1)
    
    Input:  x ∈ ℝ[B, in_ch, H, W]
    Output: y ∈ ℝ[B, out_ch, H, W]
    """
    
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        use_bn: bool = True,
        use_se: bool = True,
        dilation: int = 1
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # 1x1 compression
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(mid_ch) if use_bn else nn.Identity()
        
        # 3x3 spatial (M5: with optional dilation for larger receptive field)
        padding = dilation  # Keep same spatial size
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=padding, 
                                dilation=dilation, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(mid_ch) if use_bn else nn.Identity()
        
        # 1x1 expansion
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=not use_bn)
        self.bn3 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        
        # SE module (after expansion)
        self.se = SEModule(out_ch) if use_se else nn.Identity()
        
        # Shortcut connection
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, in_ch, H, W]
        Returns:
            Output tensor [B, out_ch, H, W]
        """
        identity = self.shortcut(x)
        
        # Bottleneck path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # SE attention
        out = self.se(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        return out


class ProxNetB(nn.Module):
    """
    SVD-free proximal surrogate for low-rank background estimation.
    
    Uses BatchNorm for stability and SE modules. Returns prox-style output: x + W(x).
    
    Args:
        in_ch: Input channels (default: 1)
        base_ch: Base channel width (default: 16)
        mid_ch: Bottleneck middle channels (default: 8)
        depth: Number of bottleneck blocks (default: 2)
    
    Input:  x ∈ ℝ[B, 1, H, W]
    Output: y ∈ ℝ[B, 1, H, W]  (background estimate)
    """
    
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        mid_ch: int = 8,
        depth: int = 2
    ):
        super().__init__()
        
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        
        # Stack of bottleneck blocks
        blocks = []
        for i in range(depth):
            blocks.append(
                ResidualBottleneck(base_ch, mid_ch, base_ch, use_bn=True, use_se=True)
            )
        self.blocks = nn.Sequential(*blocks)
        
        # Output projection
        self.head = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 1, H, W]
        Returns:
            Background estimate [B, 1, H, W] as x + residual(x)
        """
        feat = self.stem(x)
        feat = self.blocks(feat)
        residual = self.head(feat)
        
        # Proximal-style: input + learned residual
        return x + residual


class GradNet(nn.Module):
    """
    BN-free gradient surrogate for target or noise updates.
    
    Retains Lipschitz-like behavior by avoiding BatchNorm. Uses SE inside bottlenecks.
    M5: Uses dilated convolutions in later blocks for larger receptive field.
    
    Args:
        in_ch: Input channels (default: 1)
        base_ch: Base channel width (default: 16)
        mid_ch: Bottleneck middle channels (default: 8)
        depth: Number of bottleneck blocks (default: 2)
        use_dilation: Whether to use dilation in later blocks (default: True)
    
    Input:  x ∈ ℝ[B, 1, H, W]
    Output: g ∈ ℝ[B, 1, H, W]  (gradient-like map)
    """
    
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        mid_ch: int = 8,
        depth: int = 2,
        use_dilation: bool = True
    ):
        super().__init__()
        
        # Initial projection (no BN)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Stack of bottleneck blocks (no BN, with SE)
        # M5: Use dilation in later blocks for larger receptive field
        blocks = []
        for i in range(depth):
            # Use dilation=2 in the second half of blocks if enabled
            dilation = 2 if (use_dilation and i >= depth // 2) else 1
            blocks.append(
                ResidualBottleneck(base_ch, mid_ch, base_ch, use_bn=False, use_se=True, 
                                    dilation=dilation)
            )
        self.blocks = nn.Sequential(*blocks)
        
        # Output projection
        self.head = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 1, H, W]
        Returns:
            Gradient-like map [B, 1, H, W]
        """
        feat = self.stem(x)
        feat = self.blocks(feat)
        grad = self.head(feat)
        return grad


class ReconNet(nn.Module):
    """
    Reconstruction network mapping B^k + T^k + N^k → D^k.
    
    Args:
        in_ch: Input channels (default: 1)
        base_ch: Base channel width (default: 16)
        depth: Number of convolutional layers (default: 3)
    
    Input:  x ∈ ℝ[B, 1, H, W]  (sum of B + T + N)
    Output: y ∈ ℝ[B, 1, H, W]  (reconstructed image D)
    """
    
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        depth: int = 3
    ):
        super().__init__()
        
        layers = []
        
        # First layer: in_ch → base_ch
        layers.append(nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: base_ch → base_ch
        for i in range(depth - 2):
            layers.append(nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer: base_ch → in_ch (no activation)
        layers.append(nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1, bias=True))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 1, H, W]
        Returns:
            Reconstructed tensor [B, 1, H, W]
        """
        return self.net(x)

