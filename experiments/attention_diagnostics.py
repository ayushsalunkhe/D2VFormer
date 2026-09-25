import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pure_d2vformer import PureD2Vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed

def compute_detailed_attention_metrics(A: torch.Tensor, eps: float = 1e-12):
    """
    A: [B, H, O, L]
    Returns per-sample/horizon aggregated metrics
    """
    L = A.shape[-1]
    log_L = math.log(L)
    
    # H along historical dimension L: [B, H, O]
    H_tensor = -torch.sum(A * torch.log(A + eps), dim=-1) # [B, H, O]
    H_norm_tensor = H_tensor / log_L # [B, H, O]
    N_eff_tensor = torch.exp(H_tensor) # [B, H, O]
    
    max_tensor = torch.max(A, dim=-1).values # [B, H, O]
    var_tensor = torch.var(A, dim=-1, unbiased=False) # [B, H, O]
    
    return {
        'H': float(H_tensor.mean().item()),
        'H_norm': float(H_norm_tensor.mean().item()),
        'N_eff': float(N_eff_tensor.mean().item()),
        'N_eff_norm': float((N_eff_tensor / L).mean().item()),
        'max_mean': float(max_tensor.mean().item()),
        'max_median': float(torch.median(max_tensor).item()),
        'var_mean': float(var_tensor.mean().item()),
        'std_mean': float(torch.sqrt(var_tensor).mean().item()),
        'sample_H_norm': H_norm_tensor.mean(dim=(1, 2)).cpu().numpy(), # [B]
        'uniform_benchmark': 1.0 / L
    }

def safe_batch_size(req_bs: int, O: int, H: int = 128, L: int = 96, max_mb: int = 64) -> int:
    max_bytes = max_mb * 1024 * 1024
    elem_bytes = H * O * L * 4
    return min(req_bs, max(2, max_bytes // elem_bytes))

def run_diagnostics():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['ETTh1', 'exchange']
    horizons = [24, 48, 96, 192, 336, 720]
    heatmap_horizons = [24, 48, 192, 720]
    seeds = [42, 43, 44]
    
    diag_dir = os.path.join(PROJECT_ROOT, 'results', 'diagnostics')
    plot_dir = os.path.join(diag_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    metrics_records = []
    correlation_records = []
    
    print("=" * 80)
    print("STARTING SCIENTIFIC DIAGNOSTICS 1, 3, 4, 5")
    print(f"Device: {device}")
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
                
                batch_entropies = []
                batch_neff = []
                batch_max = []
                batch_var = []
                batch_std = []
                sample_h_norms = []
                sample_mses = []
                
                first_batch_A = None
                
                with torch.no_grad():
                    for i, (bx, by, bxm, bym) in enumerate(test_loader):
                        bx = bx.to(device)
                        by = by.to(device)
                        bxm = bxm.to(device)
                        bym = bym.to(device)
                        
                        out, A = model(bx, bxm, bym) # A: [B, H, O, L]
                        
                        m = compute_detailed_attention_metrics(A)
                        batch_entropies.append(m['H_norm'])
                        batch_neff.append(m['N_eff'])
                        batch_max.append(m['max_mean'])
                        batch_var.append(m['var_mean'])
                        batch_std.append(m['std_mean'])
                        
                        # Sample-level MSE for correlation analysis
                        se = torch.mean((out - by) ** 2, dim=(1, 2)).cpu().numpy() # [B]
                        sample_mses.extend(se.tolist())
                        sample_h_norms.extend(m['sample_H_norm'].tolist())
                        
                        if i == 0 and first_batch_A is None:
                            first_batch_A = A[0].detach().cpu() # First sample [H, O, L]
                        del out, A
                
                # Aggregate across test set
                mean_h_norm = float(np.mean(batch_entropies))
                mean_neff = float(np.mean(batch_neff))
                mean_max = float(np.mean(batch_max))
                mean_var = float(np.mean(batch_var))
                mean_std = float(np.mean(batch_std))
                
                # Diagnostic 5: Correlation analysis between attention entropy and forecast MSE
                if len(sample_mses) > 2:
                    p_corr, p_val = stats.pearsonr(sample_h_norms, sample_mses)
                    s_corr, s_val = stats.spearmanr(sample_h_norms, sample_mses)
                else:
                    p_corr, p_val, s_corr, s_val = 0.0, 1.0, 0.0, 1.0
                    
                record = {
                    'dataset': dataset,
                    'seed': seed,
                    'eval_horizon': O,
                    'H_norm': round(mean_h_norm, 5),
                    'N_eff': round(mean_neff, 2),
                    'N_eff_ratio': round(mean_neff / seq_len, 4),
                    'max_attention_mean': round(mean_max, 5),
                    'var_attention': round(mean_var, 7),
                    'std_attention': round(mean_std, 5),
                    'uniform_baseline': round(1.0 / seq_len, 5),
                    'pearson_corr': round(p_corr, 4),
                    'pearson_pval': round(p_val, 6),
                    'spearman_corr': round(s_corr, 4),
                    'spearman_pval': round(s_val, 6)
                }
                metrics_records.append(record)
                
                print(f"[{dataset} | Seed {seed} | O={O:3d}] H_norm: {mean_h_norm:.4f} | N_eff: {mean_neff:.1f}/{seq_len} ({mean_neff/seq_len*100:.1f}%) | Max A: {mean_max:.4f} (Uniform: {1/seq_len:.4f}) | r={p_corr:.3f}")
                
                # Diagnostic 1: Generate Visualizations for representative horizons (Seed 42)
                if seed == 42 and O in heatmap_horizons and first_batch_A is not None:
                    # first_batch_A: [H, O, L] -> average over heads: [O, L]
                    A_mean_heads = first_batch_A.mean(dim=0).numpy() # [O, L]
                    
                    # 1. Heatmap
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(A_mean_heads, aspect='auto', cmap='viridis', origin='lower')
                    ax.set_title(f"{dataset} — PureD2Vformer Cross-Temporal Attention Heatmap (O={O}, L={seq_len})")
                    ax.set_xlabel("Historical Lookback Index (l in [0, 95])")
                    ax.set_ylabel("Future Forecast Step (o in [0, O-1])")
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Attention Weight A[o, l]")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"{dataset}_attention_heatmap_O{O}.png"), dpi=300)
                    plt.close()
                    
                    # 2. Line plot of attention distribution for query positions: early (o=0), middle (o=O//2), late (o=O-1)
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    queries = [0, O // 2, O - 1]
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    for q_idx, c in zip(queries, colors):
                        ax.plot(range(seq_len), A_mean_heads[q_idx, :], label=f"Query Step o={q_idx}", color=c, lw=1.8)
                    ax.axhline(1.0 / seq_len, color='red', linestyle='--', lw=1.5, label=f"Uniform Weight (1/{seq_len} = {1/seq_len:.4f})")
                    ax.set_title(f"{dataset} — Attention Weight vs History Across Query Steps (O={O})")
                    ax.set_xlabel("Historical Time Index l")
                    ax.set_ylabel("Attention Weight A[o, l]")
                    ax.set_ylim(0, max(0.03, float(A_mean_heads.max()) * 1.25))
                    ax.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"{dataset}_attention_profiles_O{O}.png"), dpi=300)
                    plt.close()

    df_metrics = pd.DataFrame(metrics_records)
    csv_out = os.path.join(diag_dir, 'attention_metrics_summary.csv')
    df_metrics.to_csv(csv_out, index=False)
    print(f"\nSaved metrics summary to: {csv_out}")
    
    # Print grouped summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY ACROSS 3 SEEDS (Mean ± Std)")
    print("=" * 80)
    summary = df_metrics.groupby(['dataset', 'eval_horizon'])[['H_norm', 'N_eff', 'N_eff_ratio', 'max_attention_mean', 'std_attention', 'pearson_corr']].agg(['mean', 'std']).round(4)
    print(summary.to_string())

if __name__ == '__main__':
    run_diagnostics()
