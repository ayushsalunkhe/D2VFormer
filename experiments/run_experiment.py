import argparse
import pandas as pd
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.train_pure_d2vformer import train_pure_d2vformer
from experiments.train_dlinear import train_dlinear
from experiments.train_repository_d2vformer import train_repository_d2vformer
from experiments.evaluate import (
    evaluate_pure_d2vformer_zeroshot,
    evaluate_dlinear,
    evaluate_repository_d2vformer,
    evaluate_persistence
)
try:
    from tests.test_zero_shot import run_zero_shot_assertions
except ImportError:
    # If system 'tests' package overrides local namespace
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests'))
    from test_zero_shot import run_zero_shot_assertions

from utils.reproducibility import set_seed

def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner for D2Vformer Horizon Generalization")
    parser.add_argument('--datasets', nargs='+', default=['ETTh1', 'exchange'], help='Datasets to evaluate')
    parser.add_argument('--models', nargs='+', default=['pure_d2vformer', 'dlinear', 'repo_d2vformer', 'persistence'], help='Models to evaluate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44], help='Random seeds')
    parser.add_argument('--train_horizon', type=int, default=48, help='Training horizon for PureD2Vformer')
    parser.add_argument('--eval_horizons', nargs='+', type=int, default=[24, 48, 96, 192, 336, 720], help='Evaluation horizons')
    parser.add_argument('--epochs', type=int, default=8, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=96, help='Input lookback length')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, or cuda)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    import torch
    if args.device in ['cuda', 'auto']:
        try:
            if torch.cuda.is_available():
                args.device = 'cuda'
                print(f"✓ Hardware Acceleration: GPU ({torch.cuda.get_device_name(0)})")
            else:
                print("ℹ️ Hardware Acceleration: CPU (CUDA not active)")
                args.device = 'cpu'
        except Exception as e:
            print(f"ℹ️ CUDA check notice ({e}). Falling back to CPU.")
            args.device = 'cpu'
    else:
        print(f"ℹ️ Hardware Acceleration: {args.device.upper()}")

    raw_dir = os.path.join(args.output_dir, 'raw')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, 'results.csv')
    json_path = os.path.join(raw_dir, 'results.json')

    # Load existing results if resuming
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        all_results = results_df.to_dict('records')
    else:
        all_results = []

    total_runs = len(args.datasets) * len(args.seeds)
    print("=" * 80)
    print("STARTING D2VFORMER HORIZON GENERALIZATION EXPERIMENTAL PROTOCOL")
    print(f"Datasets: {args.datasets} | Models: {args.models} | Seeds: {args.seeds}")
    print(f"PureD2Vformer Train Horizon: {args.train_horizon} | Eval Horizons: {args.eval_horizons}")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"\n>>> Running Dataset: {dataset} | Random Seed: {seed} <<<")
            set_seed(seed)

            # -------------------------------------------------------------
            # 1. PureD2Vformer (Trained ONCE at O_train = 48, then zero-shot evaluated)
            # -------------------------------------------------------------
            if 'pure_d2vformer' in args.models:
                print(f"\n--- Training PureD2Vformer (single training run at O_train={args.train_horizon}) ---")
                ckpt_path, train_time, param_count = train_pure_d2vformer(
                    dataset_name=dataset,
                    seq_len=args.seq_len,
                    train_horizon=args.train_horizon,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=seed,
                    device=args.device,
                    checkpoint_dir=ckpt_dir
                )

                # Pre-evaluation zero-shot assertion audit
                import torch
                from models.pure_d2vformer import PureD2Vformer
                test_ckpt = torch.load(ckpt_path, map_location='cpu')
                v_model = PureD2Vformer(
                    c_in=test_ckpt['c_in'],
                    seq_len=args.seq_len,
                    d_model=test_ckpt['d_model'],
                    d_ff=test_ckpt['d_ff'],
                    k_freq=test_ckpt['k_freq'],
                    dropout=test_ckpt['dropout']
                )
                v_model.load_state_dict(test_ckpt['model_state_dict'])
                run_zero_shot_assertions(
                    v_model,
                    test_ckpt['checksum'],
                    expected_params=param_count,
                    c_in=test_ckpt['c_in'],
                    seq_len=args.seq_len,
                    device=args.device
                )

                # Zero-shot evaluation across all horizons
                for O in args.eval_horizons:
                    eval_metrics = evaluate_pure_d2vformer_zeroshot(
                        checkpoint_path=ckpt_path,
                        eval_horizon=O,
                        dataset_name=dataset,
                        batch_size=args.batch_size,
                        device=args.device
                    )
                    row = {
                        'dataset': dataset,
                        'model': 'PureD2Vformer',
                        'regime': 'Zero-Shot (trained on 48)',
                        'seed': seed,
                        'train_horizon': args.train_horizon,
                        'eval_horizon': O,
                        'mse': round(eval_metrics['mse'], 5),
                        'mae': round(eval_metrics['mae'], 5),
                        'attention_entropy': round(eval_metrics['attention_entropy'], 4),
                        'normalized_attention_entropy': round(eval_metrics['normalized_attention_entropy'], 4),
                        'parameter_count': param_count,
                        'training_time_sec': round(train_time, 2),
                        'inference_time_sec': round(eval_metrics['inference_time'], 3),
                        'checksum_verified': eval_metrics['checksum_verified']
                    }
                    all_results.append(row)
                    print(f"[PureD2Vformer Zero-Shot | {dataset} | Seed {seed} | O={O:3d}] MSE: {row['mse']:.4f} | MAE: {row['mae']:.4f} | Norm Entropy: {row['normalized_attention_entropy']:.4f}")

            # -------------------------------------------------------------
            # 2. DLinear (Horizon-Specific Retraining)
            # -------------------------------------------------------------
            if 'dlinear' in args.models:
                print(f"\n--- Training & Evaluating DLinear (retrained separately per horizon) ---")
                for O in args.eval_horizons:
                    ckpt_path, train_time, param_count = train_dlinear(
                        dataset_name=dataset,
                        seq_len=args.seq_len,
                        pred_len=O,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        seed=seed,
                        device=args.device,
                        checkpoint_dir=ckpt_dir
                    )
                    eval_metrics = evaluate_dlinear(
                        checkpoint_path=ckpt_path,
                        eval_horizon=O,
                        dataset_name=dataset,
                        batch_size=args.batch_size,
                        device=args.device
                    )
                    row = {
                        'dataset': dataset,
                        'model': 'DLinear',
                        'regime': 'Horizon-Specific Retrained',
                        'seed': seed,
                        'train_horizon': O,
                        'eval_horizon': O,
                        'mse': round(eval_metrics['mse'], 5),
                        'mae': round(eval_metrics['mae'], 5),
                        'attention_entropy': None,
                        'normalized_attention_entropy': None,
                        'parameter_count': param_count,
                        'training_time_sec': round(train_time, 2),
                        'inference_time_sec': round(eval_metrics['inference_time'], 3),
                        'checksum_verified': eval_metrics['checksum_verified']
                    }
                    all_results.append(row)
                    print(f"[DLinear Retrained | {dataset} | Seed {seed} | O={O:3d}] MSE: {row['mse']:.4f} | MAE: {row['mae']:.4f} | Params: {param_count}")

            # -------------------------------------------------------------
            # 3. Repository-D2Vformer (Horizon-Specific Retraining)
            # -------------------------------------------------------------
            if 'repo_d2vformer' in args.models:
                print(f"\n--- Training & Evaluating Repository-D2Vformer (retrained separately per horizon) ---")
                for O in args.eval_horizons:
                    ckpt_path, train_time, param_count = train_repository_d2vformer(
                        dataset_name=dataset,
                        seq_len=args.seq_len,
                        pred_len=O,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        seed=seed,
                        device=args.device,
                        checkpoint_dir=ckpt_dir
                    )
                    eval_metrics = evaluate_repository_d2vformer(
                        checkpoint_path=ckpt_path,
                        eval_horizon=O,
                        dataset_name=dataset,
                        batch_size=args.batch_size,
                        device=args.device
                    )
                    row = {
                        'dataset': dataset,
                        'model': 'Repository-D2Vformer',
                        'regime': 'Horizon-Specific Retrained',
                        'seed': seed,
                        'train_horizon': O,
                        'eval_horizon': O,
                        'mse': round(eval_metrics['mse'], 5),
                        'mae': round(eval_metrics['mae'], 5),
                        'attention_entropy': None,
                        'normalized_attention_entropy': None,
                        'parameter_count': param_count,
                        'training_time_sec': round(train_time, 2),
                        'inference_time_sec': round(eval_metrics['inference_time'], 3),
                        'checksum_verified': eval_metrics['checksum_verified']
                    }
                    all_results.append(row)
                    print(f"[Repository-D2Vformer | {dataset} | Seed {seed} | O={O:3d}] MSE: {row['mse']:.4f} | MAE: {row['mae']:.4f} | Params: {param_count}")

            # -------------------------------------------------------------
            # 4. Persistence (Last-Value Non-learning Baseline)
            # -------------------------------------------------------------
            if 'persistence' in args.models:
                print(f"\n--- Evaluating Persistence Baseline ---")
                for O in args.eval_horizons:
                    eval_metrics = evaluate_persistence(
                        dataset_name=dataset,
                        eval_horizon=O,
                        seq_len=args.seq_len,
                        batch_size=args.batch_size
                    )
                    row = {
                        'dataset': dataset,
                        'model': 'Persistence',
                        'regime': 'Non-Learning Control',
                        'seed': seed,
                        'train_horizon': None,
                        'eval_horizon': O,
                        'mse': round(eval_metrics['mse'], 5),
                        'mae': round(eval_metrics['mae'], 5),
                        'attention_entropy': None,
                        'normalized_attention_entropy': None,
                        'parameter_count': 0,
                        'training_time_sec': 0.0,
                        'inference_time_sec': round(eval_metrics['inference_time'], 3),
                        'checksum_verified': True
                    }
                    all_results.append(row)
                    print(f"[Persistence | {dataset} | Seed {seed} | O={O:3d}] MSE: {row['mse']:.4f} | MAE: {row['mae']:.4f}")

            # Save intermediate results to avoid loss of data
            pd.DataFrame(all_results).to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT EXECUTION COMPLETED! Total records saved: {len(all_results)}")
    print(f"Results saved to: {csv_path} and {json_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
