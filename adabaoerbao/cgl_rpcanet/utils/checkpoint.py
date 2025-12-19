"""
Checkpoint saving and loading utilities.
"""

import torch
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # If best, save a copy
    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)
    
    print(f"Checkpoint saved to {save_path}")
    if is_best:
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
    
    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    
    return {
        "epoch": epoch,
        "metrics": metrics
    }

