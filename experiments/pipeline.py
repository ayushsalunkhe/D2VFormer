"""
TC-D2Vformer Experiment Pipeline
Strict Separation of Concerns:
    1. Training (O_train = 48)
    2. Validation Temperature Selection (Zero Test Lookahead)
    3. Locked Test Evaluation (Evaluated Once across O in [24, 48, 96, 192, 336, 720])
    4. Attention Diagnostics
    5. Result Aggregation & Checksum Verification
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import TCD2Vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def train_candidate_model(
    dataset_name: str,
    tau: float,
    seed: int = 42,
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
    device: str = 'cpu',
    checkpoint_dir: str = 'results/checkpoints',
    data_root: str = 'datasets'
) -> Tuple[str, float, int]:
    """Trains a TC-D2Vformer model with a candidate fixed temperature tau."""
    set_seed(seed)
    train_loader, val_loader, _, meta = get_data_loaders(
        dataset_name=dataset_name, seq_len=seq_len, pred_len=train_horizon,
        batch_size=batch_size, data_root=data_root
    )
    c_in = meta['num_variables']
    
    model = TCD2Vformer(
        c_in=c_in, seq_len=seq_len, d_model=d_model, d_ff=d_ff,
        k_freq=k_freq, dropout=dropout, temperature_mode='fixed',
        initial_temperature=tau
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    ckpt_fn = f"temp_d2v_{dataset_name}_tau{tau}_seed{seed}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_fn)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_epoch = 0
    no_imp = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by, bxm, bym in train_loader:
            bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
            optimizer.zero_grad()
            out, _ = model(bx, bxm, bym)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for bx, by, bxm, bym in val_loader:
                bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
                out, _ = model(bx, bxm, bym)
                vl += criterion(out, by).item() * bx.size(0)
                nv += bx.size(0)
        vl /= max(1, nv)
        
        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            no_imp = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'checksum': compute_parameter_checksum(model),
                'param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'c_in': c_in, 'seq_len': seq_len, 'd_model': d_model,
                'd_ff': d_ff, 'k_freq': k_freq, 'dropout': dropout,
                'temperature_mode': 'fixed', 'initial_temperature': tau,
                'final_temperature': tau, 'train_horizon': train_horizon,
                'seed': seed, 'dataset': dataset_name,
                'best_epoch': best_epoch, 'val_loss': best_val_loss
            }, ckpt_path)
        else:
            no_imp += 1
            if no_imp >= patience:
                break
                
    return ckpt_path, best_val_loss, best_epoch

def select_temperature(
    dataset_name: str,
    seed: int,
    candidate_taus: List[float] = [0.5, 1.0, 2.0, 4.0],
    checkpoint_dir: str = 'results/checkpoints',
    device: str = 'cpu',
    data_root: str = 'datasets'
) -> Dict[str, Any]:
    """
    STRICT ZERO-LOOKAHEAD VALIDATION SELECTION.
    Inspects ONLY validation MSE at O_train = 48.
    Test set is NEVER touched during selection.
    Locks and returns selected configuration.
    """
    records = []
    for tau in candidate_taus:
        ckpt_fn = f"temp_d2v_{dataset_name}_tau{tau}_seed{seed}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_fn)
        
        if not os.path.exists(ckpt_path):
            ckpt_path, val_loss, best_ep = train_candidate_model(
                dataset_name=dataset_name, tau=tau, seed=seed,
                device=device, checkpoint_dir=checkpoint_dir, data_root=data_root
            )
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            val_loss = ckpt['val_loss']
            best_ep = ckpt.get('best_epoch', 1)
            
        records.append({
            'dataset': dataset_name,
            'seed': seed,
            'tau': tau,
            'val_loss': val_loss,
            'best_epoch': best_ep,
            'ckpt_path': ckpt_path
        })
        
    df = pd.DataFrame(records)
    best_row = df.loc[df['val_loss'].idxmin()]
    
    selected_config = {
        'dataset': dataset_name,
        'seed': seed,
        'selected_tau': float(best_row['tau']),
        'val_loss': float(best_row['val_loss']),
        'best_epoch': int(best_row['best_epoch']),
        'ckpt_path': str(best_row['ckpt_path']),
        'candidates_evaluated': candidate_taus,
        'selection_metric': 'min_validation_mse'
    }
    return selected_config

def evaluate_locked_test(
    selected_config: Dict[str, Any],
    eval_horizons: List[int] = [24, 48, 96, 192, 336, 720],
    device: str = 'cpu',
    data_root: str = 'datasets'
) -> List[Dict[str, Any]]:
    """
    Evaluates LOCKED configuration once on the test set.
    """
    ckpt_path = selected_config['ckpt_path']
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = TCD2Vformer(
        c_in=ckpt['c_in'], seq_len=ckpt['seq_len'], d_model=ckpt['d_model'],
        d_ff=ckpt['d_ff'], k_freq=ckpt['k_freq'], dropout=ckpt['dropout'],
        temperature_mode='fixed', initial_temperature=selected_config['selected_tau']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Verify checksum
    assert compute_parameter_checksum(model) == ckpt['checksum'], "Checksum failed!"
    
    results = []
    for O in eval_horizons:
        actual_bs = 32 if O >= 336 else 64
        _, _, test_loader, _ = get_data_loaders(
            dataset_name=selected_config['dataset'], seq_len=ckpt['seq_len'],
            pred_len=O, batch_size=actual_bs, data_root=data_root
        )
        
        sq_err, abs_err, n = 0.0, 0.0, 0
        with torch.no_grad():
            for bx, by, bxm, bym in test_loader:
                bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
                out, _ = model(bx, bxm, bym)
                diff = (out - by).cpu().numpy()
                sq_err += float(np.sum(diff**2))
                abs_err += float(np.sum(np.abs(diff)))
                n += diff.size
                
        results.append({
            'dataset': selected_config['dataset'],
            'seed': selected_config['seed'],
            'lookback': ckpt['seq_len'],
            'training_horizon': ckpt['train_horizon'],
            'selected_tau': selected_config['selected_tau'],
            'test_horizon': O,
            'mse': round(sq_err / n, 5),
            'mae': round(abs_err / n, 5),
            'checksum': ckpt['checksum'],
            'param_count': ckpt['param_count']
        })
    return results
