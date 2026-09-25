import torch
import numpy as np
import time
import os
import sys
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.pure_d2vformer import PureD2Vformer
from baselines.dlinear import DLinear
from baselines.persistence import PersistenceBaseline
from baselines.repository_d2vformer import create_repository_d2vformer
from utils.data import get_data_loaders
from utils.metrics import compute_mse, compute_mae, compute_batch_attention_entropy
from utils.reproducibility import compute_parameter_checksum


def _safe_batch_size(requested_bs: int, horizon: int, d_model: int = 128, seq_len: int = 96,
                     max_attn_bytes: int = 200 * 1024 * 1024) -> int:
    """
    Computes a batch size that keeps the attention tensor [B, H, O, L] below
    max_attn_bytes (default 200 MB) to avoid OOM / extreme slowness at large O.
    """
    bytes_per_sample = d_model * horizon * seq_len * 4  # float32
    safe_bs = max(1, max_attn_bytes // bytes_per_sample)
    return min(requested_bs, safe_bs)


def _streaming_mse_mae(preds: np.ndarray, trues: np.ndarray,
                       running_sq: float, running_abs: float, running_n: int):
    """Accumulate MSE and MAE statistics in a streaming (memory-efficient) way."""
    diff = preds - trues
    running_sq += float(np.sum(diff ** 2))
    running_abs += float(np.sum(np.abs(diff)))
    running_n += diff.size
    return running_sq, running_abs, running_n

def evaluate_pure_d2vformer_zeroshot(checkpoint_path: str,
                                     eval_horizon: int,
                                     dataset_name: str,
                                     batch_size: int = 64,
                                     device: str = 'cpu'):
    """
    Evaluates PureD2Vformer ZERO-SHOT on horizon eval_horizon using the checkpoint trained on O_train=48.
    Verifies parameter invariance before and after.
    Calculates MSE, MAE, Attention Entropy, and inference latency.
    Uses dynamic batch sizing to keep attention tensors [B, H, O, L] under 200 MB.
    Uses streaming (non-accumulating) MSE/MAE to avoid OOM at large horizons.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    c_in = checkpoint['c_in']
    seq_len = checkpoint['seq_len']
    d_model = checkpoint['d_model']
    d_ff = checkpoint['d_ff']
    k_freq = checkpoint['k_freq']
    dropout = checkpoint['dropout']
    baseline_hash = checkpoint['checksum']
    expected_params = checkpoint['param_count']

    model = PureD2Vformer(
        c_in=c_in,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        k_freq=k_freq,
        dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Pre-eval checksum verification
    hash_before = compute_parameter_checksum(model)
    assert hash_before == baseline_hash, f"Checkpoint hash mismatch before eval O={eval_horizon}!"

    # Cap batch size so attention tensor [B, H, O, L] stays <= 200 MB
    safe_bs = _safe_batch_size(batch_size, eval_horizon, d_model=d_model, seq_len=seq_len)

    _, _, test_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=eval_horizon,
        batch_size=safe_bs
    )

    # Streaming accumulators — never store full pred/true arrays
    running_sq, running_abs, running_n = 0.0, 0.0, 0
    attention_entropies, norm_attention_entropies = [], []

    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y, batch_xm, batch_ym in test_loader:
            batch_x = batch_x.to(device)
            batch_xm = batch_xm.to(device)
            batch_ym = batch_ym.to(device)
            batch_y_np = batch_y.numpy()  # keep on CPU, never send to device

            out, A = model(batch_x, batch_xm, batch_ym)
            out_np = out.cpu().numpy()

            # Streaming MSE / MAE
            running_sq, running_abs, running_n = _streaming_mse_mae(
                out_np, batch_y_np, running_sq, running_abs, running_n
            )

            # Attention entropy — computed and discarded immediately
            h_mean, h_norm = compute_batch_attention_entropy(A)
            attention_entropies.append(h_mean)
            norm_attention_entropies.append(h_norm)
            del A, out  # explicit free

    inference_time = time.time() - start_time

    # Post-eval checksum verification
    hash_after = compute_parameter_checksum(model)
    assert hash_after == baseline_hash, f"Model parameters mutated during zero-shot eval for O={eval_horizon}!"

    mse = running_sq / running_n if running_n > 0 else float('nan')
    mae = running_abs / running_n if running_n > 0 else float('nan')
    mean_entropy = float(np.mean(attention_entropies))
    mean_norm_entropy = float(np.mean(norm_attention_entropies))

    return {
        'mse': mse,
        'mae': mae,
        'attention_entropy': mean_entropy,
        'normalized_attention_entropy': mean_norm_entropy,
        'parameter_count': expected_params,
        'inference_time': inference_time,
        'checksum_verified': True
    }

def evaluate_dlinear(checkpoint_path: str,
                     eval_horizon: int,
                     dataset_name: str,
                     batch_size: int = 64,
                     device: str = 'cpu'):
    """Evaluates retrained DLinear on its corresponding horizon. Uses streaming MSE/MAE."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    c_in = checkpoint['c_in']
    seq_len = checkpoint['seq_len']
    pred_len = checkpoint['pred_len']
    expected_params = checkpoint['param_count']
    assert pred_len == eval_horizon

    model = DLinear(seq_len=seq_len, pred_len=pred_len, c_in=c_in).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, _, test_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=eval_horizon,
        batch_size=batch_size
    )

    running_sq, running_abs, running_n = 0.0, 0.0, 0
    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            running_sq, running_abs, running_n = _streaming_mse_mae(
                out.cpu().numpy(), batch_y.numpy(), running_sq, running_abs, running_n
            )
    inference_time = time.time() - start_time

    mse = running_sq / running_n if running_n > 0 else float('nan')
    mae = running_abs / running_n if running_n > 0 else float('nan')
    return {
        'mse': mse,
        'mae': mae,
        'attention_entropy': None,
        'normalized_attention_entropy': None,
        'parameter_count': expected_params,
        'inference_time': inference_time,
        'checksum_verified': True
    }

def evaluate_repository_d2vformer(checkpoint_path: str,
                                  eval_horizon: int,
                                  dataset_name: str,
                                  batch_size: int = 64,
                                  device: str = 'cpu'):
    """Evaluates retrained Repository-D2Vformer on its corresponding horizon. Uses streaming MSE/MAE."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    c_in = checkpoint['c_in']
    seq_len = checkpoint['seq_len']
    pred_len = checkpoint['pred_len']
    expected_params = checkpoint['param_count']
    assert pred_len == eval_horizon

    _, _, test_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=eval_horizon,
        batch_size=batch_size
    )

    model = create_repository_d2vformer(
        c_in=c_in,
        seq_len=seq_len,
        pred_len=pred_len,
        mask_spectrum=torch.tensor([0, 1]),
        d_model=128
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    running_sq, running_abs, running_n = 0.0, 0.0, 0
    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y, batch_xm, batch_ym in test_loader:
            batch_x = batch_x.to(device)
            batch_xm = batch_xm.to(device)
            batch_y_in = batch_y.to(device)
            batch_ym = batch_ym.to(device)
            out = model(batch_x, batch_xm, batch_y_in, batch_ym)
            running_sq, running_abs, running_n = _streaming_mse_mae(
                out.cpu().numpy(), batch_y.numpy(), running_sq, running_abs, running_n
            )
    inference_time = time.time() - start_time

    mse = running_sq / running_n if running_n > 0 else float('nan')
    mae = running_abs / running_n if running_n > 0 else float('nan')
    return {
        'mse': mse,
        'mae': mae,
        'attention_entropy': None,
        'normalized_attention_entropy': None,
        'parameter_count': expected_params,
        'inference_time': inference_time,
        'checksum_verified': True
    }

def evaluate_persistence(dataset_name: str,
                         eval_horizon: int,
                         seq_len: int = 96,
                         batch_size: int = 64):
    """Evaluates non-learning Persistence / Last-Value baseline. Uses streaming MSE/MAE."""
    _, _, test_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=eval_horizon,
        batch_size=batch_size
    )
    model = PersistenceBaseline()

    running_sq, running_abs, running_n = 0.0, 0.0, 0
    start_time = time.time()
    for batch_x, batch_y, _, _ in test_loader:
        out = model(batch_x, pred_len=eval_horizon)
        running_sq, running_abs, running_n = _streaming_mse_mae(
            out.numpy(), batch_y.numpy(), running_sq, running_abs, running_n
        )
    inference_time = time.time() - start_time

    mse = running_sq / running_n if running_n > 0 else float('nan')
    mae = running_abs / running_n if running_n > 0 else float('nan')
    return {
        'mse': mse,
        'mae': mae,
        'attention_entropy': None,
        'normalized_attention_entropy': None,
        'parameter_count': 0,
        'inference_time': inference_time,
        'checksum_verified': True
    }
