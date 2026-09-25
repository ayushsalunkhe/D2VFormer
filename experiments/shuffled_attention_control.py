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

def forward_with_shuffled_attention(model: PureD2Vformer, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, y_mark_dec: torch.Tensor, shuffle_seed: int = 123):
    """
    Executes PureD2Vformer forward pass, but preserves the learned attention distribution values
    while randomly permuting/shuffling the historical time index l in [0, L-1].
    NO RETRAINING.
    """
    B, L, D = x_enc.shape
    O = y_mark_dec.shape[1]
    
    # 1. Standard forward pass to get learned attention A and representations
    x_norm = model.revin(x_enc, 'norm')
    T = model.tfe(x_norm) # [B, L, H]
    
    # Date2Vec components
    v_T = torch.einsum('l,blh->bh', model.w_T, T) + model.b_T
    Omega_S = torch.einsum('kl,blh->bkh', model.W_S, T) + model.B_S
    
    E_lin = (v_T.unsqueeze(2).unsqueeze(3) * x_mark_enc.unsqueeze(1) + model.b_1).unsqueeze(1)
    E_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * x_mark_enc.unsqueeze(1).unsqueeze(2) + model.B_2)
    D_x = torch.cat([E_lin, E_har], dim=1).mean(dim=-1)
    
    F_lin = (v_T.unsqueeze(2).unsqueeze(3) * y_mark_dec.unsqueeze(1) + model.b_3).unsqueeze(1)
    F_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * y_mark_dec.unsqueeze(1).unsqueeze(2) + model.B_4)
    D_y = torch.cat([F_lin, F_har], dim=1).mean(dim=-1)
    
    Dx = D_x.permute(0, 2, 3, 1)
    Dy = D_y.permute(0, 2, 3, 1)
    
    # Raw learned attention A: [B, H, O, L]
    A_learned = torch.softmax(torch.einsum('bhok,bhlk->bhol', Dy, Dx) / math.sqrt(model.k_freq + 1), dim=-1)
    
    # 2. Shuffle historical index l
    # Set generator for reproducibility
    g = torch.Generator(device=x_enc.device)
    g.manual_seed(shuffle_seed)
    perm = torch.randperm(L, generator=g, device=x_enc.device)
    A_shuffled = A_learned[..., perm]
    
    # 3. Value aggregation with shuffled attention
    Yt_shuffled = torch.einsum('bhol,bhl->bho', A_shuffled, T.transpose(1, 2))
    
    # 4. Position-wise FFN and RevIN denorm
    out = model.ffn(Yt_shuffled.transpose(1, 2))
    y_pred = model.revin(out, 'denorm')
    
    return y_pred, A_shuffled

def safe_batch_size(req_bs: int, O: int, H: int = 128, L: int = 96, max_mb: int = 64) -> int:
    max_bytes = max_mb * 1024 * 1024
    elem_bytes = H * O * L * 4
    return min(req_bs, max(2, max_bytes // elem_bytes))

def run_shuffled_control_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['ETTh1', 'exchange']
    horizons = [24, 48, 96, 192, 336, 720]
    seeds = [42, 43, 44]
    
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, 'shuffled_attention_control_results.csv')
    
    records = []
    print("=" * 80)
    print("STARTING DIAGNOSTIC 7: CONTROLLED ATTENTION-SHUFFLING INFERENCE EXPERIMENT")
    print("Hypothesis Test: Does temporal index alignment matter under high entropy?")
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
                sq_err_shuffled, abs_err_shuffled = 0.0, 0.0
                total_elements = 0
                
                with torch.no_grad():
                    for bx, by, bxm, bym in test_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        bxm = bxm.to(device)
                        bym = bym.to(device)
                        
                        # 1. Original learned attention
                        out_learned, _ = model(bx, bxm, bym)
                        diff_learned = (out_learned - by).cpu().numpy()
                        sq_err_learned += float(np.sum(diff_learned ** 2))
                        abs_err_learned += float(np.sum(np.abs(diff_learned)))
                        
                        # 2. Shuffled attention
                        out_shuffled, _ = forward_with_shuffled_attention(model, bx, bxm, bym, shuffle_seed=seed*1000 + O)
                        diff_shuffled = (out_shuffled - by).cpu().numpy()
                        sq_err_shuffled += float(np.sum(diff_shuffled ** 2))
                        abs_err_shuffled += float(np.sum(np.abs(diff_shuffled)))
                        
                        total_elements += diff_learned.size
                
                mse_learned = sq_err_learned / total_elements
                mae_learned = abs_err_learned / total_elements
                mse_shuffled = sq_err_shuffled / total_elements
                mae_shuffled = abs_err_shuffled / total_elements
                
                pct_change_mse = ((mse_shuffled - mse_learned) / mse_learned) * 100.0
                
                record = {
                    'dataset': dataset,
                    'seed': seed,
                    'eval_horizon': O,
                    'mse_learned': round(mse_learned, 5),
                    'mae_learned': round(mae_learned, 5),
                    'mse_shuffled': round(mse_shuffled, 5),
                    'mae_shuffled': round(mae_shuffled, 5),
                    'pct_change_mse': round(pct_change_mse, 2),
                    'alignment_matters': (mse_learned < mse_shuffled)
                }
                records.append(record)
                
                status = "Alignment Critical" if mse_learned < mse_shuffled else "Insensitive"
                print(f"[{dataset} | Seed {seed} | O={O:3d}] Learned MSE: {mse_learned:.4f} vs Shuffled MSE: {mse_shuffled:.4f} (Δ={pct_change_mse:+.2f}%) [{status}]")

    df_res = pd.DataFrame(records)
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved shuffled control results to: {csv_out}")
    
    print("\n" + "=" * 80)
    print("SHUFFLED ATTENTION CONTROL SUMMARY (Mean across 3 Seeds)")
    print("=" * 80)
    summary = df_res.groupby(['dataset', 'eval_horizon'])[['mse_learned', 'mse_shuffled', 'pct_change_mse']].mean().round(4)
    print(summary.to_string())

if __name__ == '__main__':
    run_shuffled_control_experiment()
