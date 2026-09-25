import os
import sys
import math
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pure_d2vformer import PureD2Vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed

def forward_with_uniform_attention(model: PureD2Vformer, x_enc: torch.Tensor, y_mark_dec: torch.Tensor):
    """
    Executes PureD2Vformer forward pass, but replaces learned cross-temporal attention A
    with a perfectly uniform attention matrix A_uniform = 1 / L for all query positions.
    NO RETRAINING - uses existing frozen weights.
    """
    B, L, D = x_enc.shape
    O = y_mark_dec.shape[1]
    
    # 1. RevIN norm
    x_norm = model.revin(x_enc, 'norm') # [B, L, D]
    
    # 2. TFE
    T = model.tfe(x_norm) # [B, L, H]
    
    # 3. Create Uniform Attention Matrix A_uniform: [B, H, O, L] with values 1/L
    A_uniform = torch.full((B, model.d_model, O, L), 1.0 / L, dtype=x_enc.dtype, device=x_enc.device)
    
    # 4. Temporal Value Aggregation with Uniform Attention: Yt = A_uniform * T^T
    # T.transpose(1, 2) is [B, H, L]
    # Yt is [B, H, O]
    Yt = torch.einsum('bhol,bhl->bho', A_uniform, T.transpose(1, 2))
    
    # 5. Position-wise FFN and RevIN denorm
    out = model.ffn(Yt.transpose(1, 2)) # [B, O, D]
    y_pred = model.revin(out, 'denorm') # [B, O, D]
    
    return y_pred, A_uniform

def safe_batch_size(req_bs: int, O: int, H: int = 128, L: int = 96, max_mb: int = 64) -> int:
    max_bytes = max_mb * 1024 * 1024
    elem_bytes = H * O * L * 4
    return min(req_bs, max(2, max_bytes // elem_bytes))

def run_uniform_control_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['ETTh1', 'exchange']
    horizons = [24, 48, 96, 192, 336, 720]
    seeds = [42, 43, 44]
    
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, 'uniform_attention_control_results.csv')
    
    records = []
    print("=" * 80)
    print("STARTING DIAGNOSTIC 6: CONTROLLED UNIFORM-ATTENTION INFERENCE EXPERIMENT")
    print("Hypothesis Test: Does learned attention provide utility beyond uniform temporal averaging?")
    print(f"Device: {device} | NO RETRAINING")
    print("=" * 80)
    
    for dataset in datasets:
        for seed in seeds:
            ckpt_path = os.path.join(PROJECT_ROOT, 'results', 'checkpoints', f'pured2vformer_{dataset}_seed{seed}.pt')
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint not found: {ckpt_path}")
                continue
                
            ckpt = torch.load(ckpt_path, map_location=device)
            c_in = ckpt['c_in']
            seq_len = ckpt['seq_len']
            d_model = ckpt['d_model']
            d_ff = ckpt['d_ff']
            k_freq = ckpt['k_freq']
            dropout = ckpt.get('dropout', 0.05)
            
            model = PureD2Vformer(
                c_in=c_in,
                seq_len=seq_len,
                d_model=d_model,
                d_ff=d_ff,
                k_freq=k_freq,
                dropout=dropout
            ).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            
            for O in horizons:
                actual_bs = safe_batch_size(64, O, d_model, seq_len)
                _, _, test_loader, _ = get_data_loaders(
                    dataset_name=dataset,
                    seq_len=seq_len,
                    pred_len=O,
                    batch_size=actual_bs,
                    data_root=os.path.join(PROJECT_ROOT, 'datasets')
                )
                
                sq_err_learned, abs_err_learned = 0.0, 0.0
                sq_err_uniform, abs_err_uniform = 0.0, 0.0
                total_elements = 0
                
                with torch.no_grad():
                    for bx, by, bxm, bym in test_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        bxm = bxm.to(device)
                        bym = bym.to(device)
                        
                        # 1. Original learned attention forward pass
                        out_learned, _ = model(bx, bxm, bym)
                        diff_learned = (out_learned - by).cpu().numpy()
                        sq_err_learned += float(np.sum(diff_learned ** 2))
                        abs_err_learned += float(np.sum(np.abs(diff_learned)))
                        
                        # 2. Uniform attention control forward pass
                        out_uniform, _ = forward_with_uniform_attention(model, bx, bym)
                        diff_uniform = (out_uniform - by).cpu().numpy()
                        sq_err_uniform += float(np.sum(diff_uniform ** 2))
                        abs_err_uniform += float(np.sum(np.abs(diff_uniform)))
                        
                        total_elements += diff_learned.size
                
                mse_learned = sq_err_learned / total_elements
                mae_learned = abs_err_learned / total_elements
                mse_uniform = sq_err_uniform / total_elements
                mae_uniform = abs_err_uniform / total_elements
                
                pct_change_mse = ((mse_uniform - mse_learned) / mse_learned) * 100.0
                
                record = {
                    'dataset': dataset,
                    'seed': seed,
                    'eval_horizon': O,
                    'mse_learned': round(mse_learned, 5),
                    'mae_learned': round(mae_learned, 5),
                    'mse_uniform': round(mse_uniform, 5),
                    'mae_uniform': round(mae_uniform, 5),
                    'pct_change_mse': round(pct_change_mse, 2),
                    'learned_better': (mse_learned < mse_uniform)
                }
                records.append(record)
                
                status = "Learned Better" if mse_learned < mse_uniform else "Uniform Equal/Better"
                print(f"[{dataset} | Seed {seed} | O={O:3d}] Learned MSE: {mse_learned:.4f} vs Uniform MSE: {mse_uniform:.4f} (Δ={pct_change_mse:+.2f}%) [{status}]")

    df_res = pd.DataFrame(records)
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved uniform control results to: {csv_out}")
    
    print("\n" + "=" * 80)
    print("UNIFORM ATTENTION CONTROL SUMMARY (Mean across 3 Seeds)")
    print("=" * 80)
    summary = df_res.groupby(['dataset', 'eval_horizon'])[['mse_learned', 'mse_uniform', 'pct_change_mse']].mean().round(4)
    print(summary.to_string())

if __name__ == '__main__':
    run_uniform_control_experiment()
