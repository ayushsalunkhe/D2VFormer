import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.temperature_d2vformer import TemperaturePureD2Vformer
from models.pure_d2vformer import PureD2Vformer
from baselines.dlinear import DLinear
from baselines.persistence import PersistenceBaseline
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def compute_attention_diagnostics(A: torch.Tensor, eps: float = 1e-12):
    """
    Computes H_norm, N_eff, max attention, variance, and KL divergence from uniform.
    A: [B, H, O, L]
    """
    L = A.shape[-1]
    log_L = math.log(L) if L > 1 else 1.0
    
    # Shannon Entropy
    H_tensor = -torch.sum(A * torch.log(A + eps), dim=-1)
    H_norm = float((H_tensor / log_L).mean().item())
    N_eff = float(torch.exp(H_tensor).mean().item())
    max_A = float(torch.max(A, dim=-1).values.mean().item())
    
    # Variance
    A_var = float(torch.var(A, dim=-1).mean().item())
    
    # KL Divergence from uniform: sum A log(A / (1/L))
    u = 1.0 / L
    kl_tensor = torch.sum(A * torch.log((A + eps) / u), dim=-1)
    kl_div = float(kl_tensor.mean().item())
    
    return {
        'H_norm': H_norm,
        'N_eff': N_eff,
        'N_eff_ratio': N_eff / L,
        'max_attention': max_A,
        'attention_variance': A_var,
        'kl_div_from_uniform': kl_div
    }

def train_phase4_temperature_model(
    dataset_name: str,
    initial_temperature: float = 1.0,
    seq_len: int = 96,
    train_horizon: int = 48,
    d_model: int = 128,
    d_ff: int = 256,
    k_freq: int = 16,
    dropout: float = 0.05,
    lr: float = 1e-3,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 64,
    seed: int = 42,
    device: str = 'cpu',
    checkpoint_dir: str = 'results/checkpoints',
    data_root: str = 'datasets'
):
    set_seed(seed)
    train_loader, val_loader, _, meta = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=train_horizon,
        batch_size=batch_size,
        data_root=data_root
    )
    c_in = meta['num_variables']
    
    model = TemperaturePureD2Vformer(
        c_in=c_in,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        k_freq=k_freq,
        dropout=dropout,
        temperature_mode='fixed',
        initial_temperature=initial_temperature
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    tag = f"tau{initial_temperature}"
    ckpt_filename = f"temp_d2v_{dataset_name}_{tag}_seed{seed}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for bx, by, bxm, bym in train_loader:
            bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
            optimizer.zero_grad()
            out, _ = model(bx, bxm, bym)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            n_train += bx.size(0)
            
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by, bxm, bym in val_loader:
                bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
                out, _ = model(bx, bxm, bym)
                val_loss += criterion(out, by).item() * bx.size(0)
                n_val += bx.size(0)
        val_loss /= max(1, n_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'checksum': compute_parameter_checksum(model),
                'param_count': param_count,
                'c_in': c_in,
                'seq_len': seq_len,
                'd_model': d_model,
                'd_ff': d_ff,
                'k_freq': k_freq,
                'dropout': dropout,
                'temperature_mode': 'fixed',
                'initial_temperature': initial_temperature,
                'final_temperature': initial_temperature,
                'train_horizon': train_horizon,
                'seed': seed,
                'dataset': dataset_name,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    train_time = time.time() - start_time
    return ckpt_path, best_val_loss, best_epoch, train_time

def evaluate_model_on_test(
    ckpt_path: str,
    eval_horizon: int,
    dataset_name: str,
    batch_size: int = 64,
    device: str = 'cpu',
    data_root: str = 'datasets',
    uniform_attention: bool = False
):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TemperaturePureD2Vformer(
        c_in=ckpt['c_in'],
        seq_len=ckpt['seq_len'],
        d_model=ckpt['d_model'],
        d_ff=ckpt['d_ff'],
        k_freq=ckpt['k_freq'],
        dropout=ckpt['dropout'],
        temperature_mode=ckpt['temperature_mode'],
        initial_temperature=ckpt['initial_temperature']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    assert compute_parameter_checksum(model) == ckpt['checksum'], "Checksum failed"
    
    actual_bs = min(batch_size, 32 if eval_horizon >= 336 else batch_size)
    _, _, test_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=ckpt['seq_len'],
        pred_len=eval_horizon,
        batch_size=actual_bs,
        data_root=data_root
    )
    
    sq_err, abs_err, n = 0.0, 0.0, 0
    all_diag = []
    
    with torch.no_grad():
        for bx, by, bxm, bym in test_loader:
            bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
            if uniform_attention:
                B, L, D = bx.shape
                x_norm = model.revin(bx, 'norm')
                T = model.tfe(x_norm)
                A_uni = torch.full((B, model.d_model, eval_horizon, L), 1.0 / L, dtype=bx.dtype, device=device)
                Yt = torch.einsum('bhol,bhl->bho', A_uni, T.transpose(1, 2))
                out = model.revin(model.ffn(Yt.transpose(1, 2)), 'denorm')
                A = A_uni
            else:
                out, A = model(bx, bxm, bym)
                
            diff = (out - by).cpu().numpy()
            sq_err += float(np.sum(diff**2))
            abs_err += float(np.sum(np.abs(diff)))
            n += diff.size
            
            if len(all_diag) < 5:
                all_diag.append(compute_attention_diagnostics(A))
                
    diag_summary = {
        k: float(np.mean([d[k] for d in all_diag])) for k in all_diag[0].keys()
    } if len(all_diag) > 0 else {}
    
    return {
        'mse': sq_err / n,
        'mae': abs_err / n,
        'tau': float(ckpt['final_temperature']),
        'diagnostics': diag_summary
    }

def train_and_eval_dlinear_horizon(
    dataset_name: str,
    pred_len: int,
    seed: int = 42,
    device: str = 'cpu',
    data_root: str = 'datasets'
):
    set_seed(seed)
    train_ldr, val_ldr, test_ldr, meta = get_data_loaders(
        dataset_name=dataset_name, seq_len=96, pred_len=pred_len, batch_size=64, data_root=data_root
    )
    model = DLinear(seq_len=96, pred_len=pred_len, c_in=meta['num_variables'], moving_avg_kernel=25).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    best_val = float('inf')
    best_state = None
    no_imp = 0
    for epoch in range(1, 11):
        model.train()
        for bx, by, _, _ in train_ldr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for bx, by, _, _ in val_ldr:
                bx, by = bx.to(device), by.to(device)
                vl += crit(model(bx), by).item() * bx.size(0)
                nv += bx.size(0)
        vl /= max(1, nv)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 3:
                break
                
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    sq_err, abs_err, n = 0.0, 0.0, 0
    with torch.no_grad():
        for bx, by, _, _ in test_ldr:
            bx, by = bx.to(device), by.to(device)
            diff = (model(bx) - by).cpu().numpy()
            sq_err += float(np.sum(diff**2))
            abs_err += float(np.sum(np.abs(diff)))
            n += diff.size
    return sq_err / n, abs_err / n

def eval_persistence_horizon(
    dataset_name: str,
    pred_len: int,
    device: str = 'cpu',
    data_root: str = 'datasets'
):
    _, _, test_ldr, _ = get_data_loaders(
        dataset_name=dataset_name, seq_len=96, pred_len=pred_len, batch_size=64, data_root=data_root
    )
    p_model = PersistenceBaseline().to(device)
    sq_err, abs_err, n = 0.0, 0.0, 0
    with torch.no_grad():
        for bx, by, _, _ in test_ldr:
            bx, by = bx.to(device), by.to(device)
            pred = p_model(bx, pred_len=pred_len)
            diff = (pred - by).cpu().numpy()
            sq_err += float(np.sum(diff**2))
            abs_err += float(np.sum(np.abs(diff)))
            n += diff.size
    return sq_err / n, abs_err / n
