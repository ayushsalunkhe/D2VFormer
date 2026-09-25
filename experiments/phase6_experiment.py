"""
Phase 6 Experiment Runner: Temporal-Conditioned Date2Vecformer (TCD2Vformer)
Supports:
    1. Pilot experiment on ETTh1, seed=42, O_train=48, O_eval={48, 192, 720}
    2. Full Phase 6 benchmark across ETTh1, Exchange, ETTh2, ETTm1 with seeds 42, 43, 44
    3. Strict Zero-Lookahead Protocol: Validation selection -> Locked Test Evaluation
    4. Horizon-independent parameter verification & SHA-256 checksums
"""

import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.tcd2vformer import TCD2Vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum, verify_parameter_invariance

def compute_attention_diagnostics(A: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Computes H_norm, N_eff, max_attention, variance, and KL divergence from uniform."""
    # A: [B, H, O, L]
    L = A.shape[-1]
    log_L = math.log(L) if L > 1 else 1.0

    H_tensor = -torch.sum(A * torch.log(A + eps), dim=-1)
    H_norm = float((H_tensor / log_L).mean().item())
    N_eff = float(torch.exp(H_tensor).mean().item())
    max_A = float(torch.max(A, dim=-1).values.mean().item())
    A_var = float(torch.var(A, dim=-1).mean().item())

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

def train_tcd2vformer(
    dataset_name: str,
    temperature_mode: str = 'fixed',
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
    checkpoint_dir: str = 'results/phase6/checkpoints',
    data_root: str = 'datasets',
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None
) -> Tuple[str, float, int, float]:
    """Trains a TCD2Vformer model with early stopping on validation loss."""
    set_seed(seed)
    train_loader, val_loader, _, meta = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=train_horizon,
        batch_size=batch_size,
        data_root=data_root
    )
    c_in = meta['num_variables']

    model = TCD2Vformer(
        c_in=c_in,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        k_freq=k_freq,
        dropout=dropout,
        temperature_mode=temperature_mode,
        initial_temperature=initial_temperature
    ).to(device)

    # Enforce parameter constancy check before training
    model.assert_horizon_independence(device=device)
    param_count = model.get_parameter_count()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tag = f"{temperature_mode}" if temperature_mode != 'fixed' else f"fixed_tau{initial_temperature}"
    ckpt_fn = f"tcd2v_{dataset_name}_{tag}_seed{seed}.pt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, ckpt_fn)

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0
        for b_idx, (bx, by, bxm, bym) in enumerate(train_loader):
            if max_train_batches is not None and b_idx >= max_train_batches:
                break
            bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
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
            for b_idx, (bx, by, bxm, bym) in enumerate(val_loader):
                if max_val_batches is not None and b_idx >= max_val_batches:
                    break
                bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
                out, _ = model(bx, bxm, bym)
                val_loss += criterion(out, by).item() * bx.size(0)
                n_val += bx.size(0)
        val_loss /= max(1, n_val)
        ep_time = time.time() - ep_start

        print(f"      Epoch {epoch:2d}/{epochs:2d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | {ep_time:.1f}s", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
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
                'temperature_mode': temperature_mode,
                'initial_temperature': initial_temperature,
                'train_horizon': train_horizon,
                'seed': seed,
                'dataset': dataset_name,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"      Early stopping triggered at epoch {epoch} (patience={patience})", flush=True)
                break

    train_time = time.time() - start_time
    return ckpt_path, best_val_loss, best_epoch, train_time

def evaluate_locked_test(
    ckpt_path: str,
    eval_horizons: List[int],
    batch_size: int = 64,
    device: str = 'cpu',
    data_root: str = 'datasets',
    max_test_batches: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluates locked checkpoint across multi-horizon test sets."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TCD2Vformer(
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

    # Strict invariant assertion
    assert compute_parameter_checksum(model) == ckpt['checksum'], "Checksum failed during locked test evaluation!"
    model.assert_horizon_independence(eval_horizons=eval_horizons, device=device)

    test_records = []
    diag_records = []

    for O in eval_horizons:
        actual_bs = min(batch_size, 32 if O >= 336 else batch_size)
        _, _, test_loader, _ = get_data_loaders(
            dataset_name=ckpt['dataset'],
            seq_len=ckpt['seq_len'],
            pred_len=O,
            batch_size=actual_bs,
            data_root=data_root
        )

        sq_err, abs_err, n = 0.0, 0.0, 0
        diag_list = []
        tau_vals = []

        with torch.no_grad():
            for b_idx, (bx, by, bxm, bym) in enumerate(test_loader):
                if max_test_batches is not None and b_idx >= max_test_batches:
                    break
                bx, by, bxm, bym = bx.to(device), by.to(device), bxm.to(device), bym.to(device)
                out, A, tau = model(bx, bxm, bym, return_attention=True)
                diff = (out - by).cpu().numpy()
                sq_err += float(np.sum(diff ** 2))
                abs_err += float(np.sum(np.abs(diff)))
                n += diff.size

                diag_list.append(compute_attention_diagnostics(A))
                tau_vals.append(tau.detach().cpu().numpy())

        # Average metrics
        mse = sq_err / max(1, n)
        mae = abs_err / max(1, n)

        avg_h_norm = float(np.mean([d['H_norm'] for d in diag_list]))
        avg_n_eff = float(np.mean([d['N_eff'] for d in diag_list]))
        avg_max_a = float(np.mean([d['max_attention'] for d in diag_list]))

        all_tau = np.concatenate([t.flatten() for t in tau_vals])
        mean_tau = float(np.mean(all_tau))
        std_tau = float(np.std(all_tau))
        min_tau = float(np.min(all_tau))
        max_tau = float(np.max(all_tau))

        test_records.append({
            'dataset': ckpt['dataset'],
            'seed': ckpt['seed'],
            'model_name': f"TCD2Vformer_{ckpt['temperature_mode']}",
            'temperature_mode': ckpt['temperature_mode'],
            'training_horizon': ckpt['train_horizon'],
            'eval_horizon': O,
            'test_mse': round(mse, 5),
            'test_mae': round(mae, 5),
            'mean_tau': round(mean_tau, 4),
            'param_count': ckpt['param_count'],
            'checksum': ckpt['checksum']
        })

        diag_records.append({
            'dataset': ckpt['dataset'],
            'seed': ckpt['seed'],
            'model_name': f"TCD2Vformer_{ckpt['temperature_mode']}",
            'temperature_mode': ckpt['temperature_mode'],
            'eval_horizon': O,
            'H_norm': round(avg_h_norm, 5),
            'N_eff': round(avg_n_eff, 4),
            'max_attention': round(avg_max_a, 5),
            'mean_tau': round(mean_tau, 4),
            'std_tau': round(std_tau, 4),
            'min_tau': round(min_tau, 4),
            'max_tau': round(max_tau, 4)
        })

    return test_records, diag_records

def run_pilot_experiment(
    device: str = 'cpu',
    epochs: int = 2,
    batch_size: int = 64,
    max_train_batches: Optional[int] = 10,
    max_val_batches: Optional[int] = 5,
    max_test_batches: Optional[int] = 5
):
    """Runs pilot experiment on ETTh1, seed=42, O_train=48, O_eval={48, 192, 720}."""
    print("=" * 70, flush=True)
    print("STARTING PHASE 6 PILOT EXPERIMENT", flush=True)
    print(f"Target: ETTh1 | seed=42 | O_train=48 | O_eval=[48, 192, 720] | epochs={epochs}", flush=True)
    print(f"Batches: train={max_train_batches} | val={max_val_batches} | test={max_test_batches}", flush=True)
    print("Modes: ['fixed' (tau=1.0), 'learned_global', 'temporal_context', 'query_conditioned']", flush=True)
    print("=" * 70, flush=True)

    dataset = 'ETTh1'
    seed = 42
    eval_horizons = [48, 192, 720]
    modes = [
        ('fixed', 1.0),
        ('learned_global', 1.0),
        ('temporal_context', 1.0),
        ('query_conditioned', 1.0)
    ]

    all_val = []
    all_test = []
    all_diag = []

    for mode, init_t in modes:
        print(f"\n---> Training mode: {mode} (init_tau={init_t})...", flush=True)
        ckpt_path, val_loss, best_ep, t_time = train_tcd2vformer(
            dataset_name=dataset,
            temperature_mode=mode,
            initial_temperature=init_t,
            seed=seed,
            epochs=epochs,
            patience=3,
            batch_size=batch_size,
            device=device,
            checkpoint_dir='results/phase6/checkpoints',
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches
        )
        print(f"      Best Val Loss: {val_loss:.5f} (Epoch {best_ep}, Time: {t_time:.1f}s)", flush=True)
        print(f"      Checkpoint: {ckpt_path}", flush=True)

        all_val.append({
            'dataset': dataset,
            'seed': seed,
            'temperature_mode': mode,
            'initial_temperature': init_t,
            'best_epoch': best_ep,
            'val_loss': round(val_loss, 5),
            'train_time_sec': round(t_time, 2),
            'ckpt_path': ckpt_path
        })

        print(f"---> Evaluating Locked Test on horizons {eval_horizons}...", flush=True)
        test_recs, diag_recs = evaluate_locked_test(
            ckpt_path=ckpt_path,
            eval_horizons=eval_horizons,
            device=device,
            max_test_batches=max_test_batches
        )
        all_test.extend(test_recs)
        all_diag.extend(diag_recs)
        for tr in test_recs:
            print(f"      O={tr['eval_horizon']:3d} | MSE: {tr['test_mse']:.5f} | MAE: {tr['test_mae']:.5f} | Mean tau: {tr['mean_tau']:.3f}", flush=True)

    # Save pilot results
    os.makedirs('results/phase6', exist_ok=True)
    pd.DataFrame(all_val).to_csv('results/phase6/pilot_validation.csv', index=False)
    pd.DataFrame(all_test).to_csv('results/phase6/pilot_test.csv', index=False)
    pd.DataFrame(all_diag).to_csv('results/phase6/pilot_diagnostics.csv', index=False)

    print("\n" + "=" * 70, flush=True)
    print("PILOT EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)
    print("Artifacts saved:", flush=True)
    print("  - results/phase6/pilot_validation.csv", flush=True)
    print("  - results/phase6/pilot_test.csv", flush=True)
    print("  - results/phase6/pilot_diagnostics.csv", flush=True)
    print("=" * 70, flush=True)

def run_full_experiment(
    device: str = 'cpu',
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    data_root: str = 'datasets',
    checkpoint_dir: str = 'results/phase6/checkpoints',
    results_dir: str = 'results/phase6',
    resume: bool = True
):
    """
    Full Phase 6 benchmark experiment.

    Strict Protocol:
        1. Train once at O_train=48 per (dataset, mode, seed) combination.
        2. Select NO hyperparameters from test set.
        3. Evaluate locked checkpoint zero-shot on O in [24, 48, 96, 192, 336, 720].
        4. Report all results without cherry-picking.

    Hard Stopping Rule:
        This IS the final experiment phase. No Phase 7 unless a specific,
        experimentally justified flaw is discovered here.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Full experimental matrix
    DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'exchange']
    SEEDS = [42, 43, 44]
    MODES = [
        ('fixed', 1.0),           # Ablation A: baseline tau=1.0
        ('learned_global', 1.0),  # Ablation C: global learnable tau
        ('temporal_context', 1.0),# Ablation D: temporal context conditioned
        ('query_conditioned', 1.0) # Ablation E: query-conditioned dynamic tau
    ]
    EVAL_HORIZONS = [24, 48, 96, 192, 336, 720]

    # Resume: load already-completed results if they exist
    val_csv = os.path.join(results_dir, 'validation_results.csv')
    test_csv = os.path.join(results_dir, 'locked_test_results.csv')
    diag_csv = os.path.join(results_dir, 'attention_diagnostics.csv')

    all_val = []
    all_test = []
    all_diag = []

    if resume:
        if os.path.exists(val_csv):
            all_val = pd.read_csv(val_csv).to_dict('records')
            print(f"Resuming: loaded {len(all_val)} existing validation records.", flush=True)
        if os.path.exists(test_csv):
            all_test = pd.read_csv(test_csv).to_dict('records')
            print(f"Resuming: loaded {len(all_test)} existing test records.", flush=True)
        if os.path.exists(diag_csv):
            all_diag = pd.read_csv(diag_csv).to_dict('records')
            print(f"Resuming: loaded {len(all_diag)} existing diagnostic records.", flush=True)

    total = len(DATASETS) * len(SEEDS) * len(MODES)
    completed = 0

    print("=" * 72, flush=True)
    print("PHASE 6 FULL EXPERIMENT — TEMPORAL-CONDITIONED DATE2VECFORMER", flush=True)
    print(f"Matrix: {len(DATASETS)} datasets × {len(SEEDS)} seeds × {len(MODES)} modes = {total} training runs", flush=True)
    print(f"Evaluation horizons (zero-shot): {EVAL_HORIZONS}", flush=True)
    print(f"Hard stopping rule: NO Phase 7 after this experiment.", flush=True)
    print("=" * 72, flush=True)

    for dataset in DATASETS:
        for mode, init_t in MODES:
            for seed in SEEDS:
                completed += 1
                mode_tag = f"{mode}" if mode != 'fixed' else f"fixed_tau{init_t}"
                ckpt_fn = f"tcd2v_{dataset}_{mode_tag}_seed{seed}.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_fn)

                # Skip if already completed (resume logic)
                already_done = any(
                    r.get('dataset') == dataset and
                    r.get('temperature_mode') == mode and
                    r.get('seed') == seed
                    for r in all_val
                )
                if resume and already_done and os.path.exists(ckpt_path):
                    print(f"[{completed:3d}/{total}] SKIP  {dataset:<12} | {mode:<18} | seed={seed} (already completed)", flush=True)
                    continue

                print(f"\n[{completed:3d}/{total}] TRAIN {dataset:<12} | {mode:<18} | seed={seed}", flush=True)

                try:
                    ckpt_path, val_loss, best_ep, t_time = train_tcd2vformer(
                        dataset_name=dataset,
                        temperature_mode=mode,
                        initial_temperature=init_t,
                        seq_len=96,
                        train_horizon=48,
                        d_model=128,
                        d_ff=256,
                        k_freq=16,
                        dropout=0.05,
                        lr=lr,
                        epochs=epochs,
                        patience=patience,
                        batch_size=batch_size,
                        seed=seed,
                        device=device,
                        checkpoint_dir=checkpoint_dir,
                        data_root=data_root
                    )

                    print(f"         Val MSE: {val_loss:.5f} | Best Epoch: {best_ep} | Time: {t_time:.1f}s", flush=True)

                    all_val.append({
                        'dataset': dataset,
                        'seed': seed,
                        'temperature_mode': mode,
                        'initial_temperature': init_t,
                        'best_epoch': best_ep,
                        'val_loss': round(val_loss, 6),
                        'train_time_sec': round(t_time, 2),
                        'ckpt_path': ckpt_path
                    })
                    # Save after each training run
                    pd.DataFrame(all_val).to_csv(val_csv, index=False)

                    print(f"         Evaluating locked test on horizons {EVAL_HORIZONS}...", flush=True)
                    test_recs, diag_recs = evaluate_locked_test(
                        ckpt_path=ckpt_path,
                        eval_horizons=EVAL_HORIZONS,
                        batch_size=batch_size,
                        device=device,
                        data_root=data_root
                    )
                    all_test.extend(test_recs)
                    all_diag.extend(diag_recs)

                    # Save after each evaluation
                    pd.DataFrame(all_test).to_csv(test_csv, index=False)
                    pd.DataFrame(all_diag).to_csv(diag_csv, index=False)

                    for tr in test_recs:
                        print(f"         O={tr['eval_horizon']:3d} | MSE={tr['test_mse']:.5f} | MAE={tr['test_mae']:.5f} | tau={tr['mean_tau']:.3f}", flush=True)

                except Exception as e:
                    print(f"         ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue

    # Final save
    pd.DataFrame(all_val).to_csv(val_csv, index=False)
    pd.DataFrame(all_test).to_csv(test_csv, index=False)
    pd.DataFrame(all_diag).to_csv(diag_csv, index=False)

    print("\n" + "=" * 72, flush=True)
    print("PHASE 6 FULL EXPERIMENT COMPLETED.", flush=True)
    print(f"  Validation records: {len(all_val)}", flush=True)
    print(f"  Test records:       {len(all_test)}", flush=True)
    print(f"  Diagnostic records: {len(all_diag)}", flush=True)
    print(f"  Results saved to:   {results_dir}/", flush=True)
    print("=" * 72, flush=True)
    print("\nHard Stopping Rule: Phase 6 evaluation is COMPLETE and FROZEN.", flush=True)
    print("No further experiment phases unless a specific, experimentally", flush=True)
    print("justified implementation flaw is discovered in these results.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCD2Vformer Phase 6 Experiment')
    parser.add_argument('--mode', type=str, default='pilot', choices=['pilot', 'full'],
                        help='Experiment mode: pilot (fast verify) or full (all datasets/seeds)')
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--seed', type=int, default=42)
    # Pilot-specific arguments
    parser.add_argument('--max_train_batches', type=int, default=10)
    parser.add_argument('--max_val_batches', type=int, default=5)
    parser.add_argument('--max_test_batches', type=int, default=5)
    # Shared training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Max training epochs. Pilot default overridden to 2.')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data_root', type=str, default='datasets')
    parser.add_argument('--no_resume', action='store_true',
                        help='Disable resume: re-run all experiments from scratch')
    args = parser.parse_args()

    if args.mode == 'pilot':
        run_pilot_experiment(
            device=args.device,
            epochs=min(args.epochs, 2),
            batch_size=args.batch_size,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches
        )
    elif args.mode == 'full':
        run_full_experiment(
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            data_root=args.data_root,
            resume=not args.no_resume
        )

