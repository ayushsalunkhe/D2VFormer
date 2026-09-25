import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.temperature_d2vformer import TemperaturePureD2Vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def compute_batch_attention_metrics(A: torch.Tensor, eps: float = 1e-12):
    L = A.shape[-1]
    log_L = math.log(L) if L > 1 else 1.0
    H_tensor = -torch.sum(A * torch.log(A + eps), dim=-1)
    H_norm = float((H_tensor / log_L).mean().item())
    N_eff = float(torch.exp(H_tensor).mean().item())
    max_A = float(torch.max(A, dim=-1).values.mean().item())
    return H_norm, N_eff, N_eff / L, max_A

def train_temperature_d2vformer(
    dataset_name: str,
    seq_len: int = 96,
    train_horizon: int = 48,
    d_model: int = 128,
    d_ff: int = 256,
    k_freq: int = 16,
    dropout: float = 0.05,
    temperature_mode: str = 'fixed',
    initial_temperature: float = 1.0,
    lr: float = 1e-3,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 64,
    seed: int = 42,
    device: str = 'cpu',
    checkpoint_dir: str = 'results/checkpoints'
):
    set_seed(seed)
    train_loader, val_loader, _, meta = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=train_horizon,
        batch_size=batch_size,
        data_root=os.path.join(PROJECT_ROOT, 'datasets')
    )
    c_in = meta['num_variables']
    
    model = TemperaturePureD2Vformer(
        c_in=c_in,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        k_freq=k_freq,
        dropout=dropout,
        temperature_mode=temperature_mode,
        initial_temperature=initial_temperature
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_tau_learned = float(initial_temperature)
    
    variant_tag = f"tau{initial_temperature}" if temperature_mode == 'fixed' else "learnable"
    ckpt_filename = f"temp_d2v_{dataset_name}_{variant_tag}_seed{seed}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        
        for bx, by, bxm, bym in train_loader:
            bx, by = bx.to(device), by.to(device)
            bxm, bym = bxm.to(device), bym.to(device)
            
            optimizer.zero_grad()
            out, _ = model(bx, bxm, bym)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
            n_train += bx.size(0)
            
        train_loss /= max(1, n_train)
        
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by, bxm, bym in val_loader:
                bx, by = bx.to(device), by.to(device)
                bxm, bym = bxm.to(device), bym.to(device)
                out, _ = model(bx, bxm, bym)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
                n_val += bx.size(0)
                
        val_loss /= max(1, n_val)
        current_tau = float(model.get_temperature().detach().cpu().item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            best_tau_learned = current_tau
            
            checksum = compute_parameter_checksum(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'checksum': checksum,
                'param_count': param_count,
                'c_in': c_in,
                'seq_len': seq_len,
                'd_model': d_model,
                'd_ff': d_ff,
                'k_freq': k_freq,
                'dropout': dropout,
                'temperature_mode': temperature_mode,
                'initial_temperature': initial_temperature,
                'final_temperature': best_tau_learned,
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
    return ckpt_path, train_time, param_count, best_tau_learned

def evaluate_temperature_zeroshot(
    ckpt_path: str,
    eval_horizon: int,
    dataset_name: str,
    batch_size: int = 64,
    device: str = 'cpu'
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
    
    # Audit parameter checksum before evaluation
    initial_hash = ckpt['checksum']
    current_hash = compute_parameter_checksum(model)
    assert current_hash == initial_hash, f"Parameter checksum mismatch before evaluation: {current_hash} vs {initial_hash}"
    
    # Dynamic batch cap for long horizons
    actual_bs = min(batch_size, 32 if eval_horizon >= 336 else batch_size)
    _, _, test_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=ckpt['seq_len'],
        pred_len=eval_horizon,
        batch_size=actual_bs,
        data_root=os.path.join(PROJECT_ROOT, 'datasets')
    )
    
    sq_err, abs_err, total_elements = 0.0, 0.0, 0
    h_norms, n_effs, n_eff_ratios, max_attns = [], [], [], []
    
    start_time = time.time()
    with torch.no_grad():
        for bx, by, bxm, bym in test_loader:
            bx, by = bx.to(device), by.to(device)
            bxm, bym = bxm.to(device), bym.to(device)
            
            out, A = model(bx, bxm, bym)
            diff = (out - by).cpu().numpy()
            sq_err += float(np.sum(diff ** 2))
            abs_err += float(np.sum(np.abs(diff)))
            total_elements += diff.size
            
            hn, ne, ner, ma = compute_batch_attention_metrics(A)
            h_norms.append(hn)
            n_effs.append(ne)
            n_eff_ratios.append(ner)
            max_attns.append(ma)
            
    inf_time = time.time() - start_time
    post_hash = compute_parameter_checksum(model)
    assert post_hash == initial_hash, "Parameter checksum mutated during zero-shot evaluation!"
    
    return {
        'mse': sq_err / total_elements,
        'mae': abs_err / total_elements,
        'H_norm': float(np.mean(h_norms)),
        'N_eff': float(np.mean(n_effs)),
        'N_eff_ratio': float(np.mean(n_eff_ratios)),
        'max_attention': float(np.mean(max_attns)),
        'inference_time': inf_time,
        'tau': float(model.get_temperature().detach().cpu().item()),
        'checksum_verified': True
    }
