import torch
import numpy as np
import math

def compute_mse(preds: np.ndarray, trues: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((preds - trues) ** 2))

def compute_mae(preds: np.ndarray, trues: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(preds - trues)))

def compute_batch_attention_entropy(attention_tensor: torch.Tensor, eps: float = 1e-12) -> tuple:
    """
    Computes Shannon attention entropy and normalized attention entropy.
    Args:
        attention_tensor: [B, H, O, L] where sum over L equals 1.
    Returns:
        (mean_entropy, normalized_entropy): floats
    """
    # A: [B, H, O, L]
    L = attention_tensor.shape[-1]
    # H = - sum_l (p_l * log(p_l + eps))
    entropy = -torch.sum(attention_tensor * torch.log(attention_tensor + eps), dim=-1) # [B, H, O]
    mean_entropy = float(entropy.mean().item())
    max_entropy = math.log(L)
    norm_entropy = float(mean_entropy / max_entropy) if max_entropy > 0 else 0.0
    return mean_entropy, norm_entropy
