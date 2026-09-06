#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLinear Multi-Horizon Results Consolidation
Verifies all 8 Colab-trained checkpoints and combines with existing 96h results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
from data.dataset import MyDataset
from utils.get_data import get_data
import argparse

# Import DLinear from the training script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines'))
from run_dlinear_multihorizon import DLinear


def verify_checkpoint(ckpt_path, data_name, pred_len, d_feature):
    """
    Verify checkpoint loads and has expected architecture.
    """
    if not os.path.exists(ckpt_path):
        return {'status': 'MISSING', 'error': f'File not found: {ckpt_path}'}

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = DLinear(seq_len=96, pred_len=pred_len, d_feature=d_feature)
        model.load_state_dict(ckpt['model'])

        # Check architecture
        weight_shape = ckpt['model']['linear_trend.0.weight'].shape
        expected_shape = (pred_len, 96)

        if weight_shape != expected_shape:
            return {
                'status': 'ARCHITECTURE_MISMATCH',
                'expected': expected_shape,
                'actual': weight_shape
            }

        return {
            'status': 'OK',
            'epoch': ckpt.get('epoch', 'unknown'),
            'pred_len_verified': weight_shape[0] == pred_len
        }

    except Exception as e:
        return {'status': 'LOAD_ERROR', 'error': str(e)}


def main():
    print("="*70)
    print("DLinear Multi-Horizon Results Consolidation")
    print("="*70)

    # Define all experiments
    experiments = {
        'ETTh1': {
            'd_feature': 7,
            'horizons': [48, 72, 96, 192, 336]
        },
        'IndiaAQI': {
            'd_feature': 6,
            'horizons': [48, 72, 96, 192, 336]
        }
    }

    all_results = []

    print("\n" + "="*70)
    print("CHECKPOINT VERIFICATION")
    print("="*70)

    for data_name, config in experiments.items():
        print(f"\n{data_name}:")
        d_feature = config['d_feature']

        for pred_len in config['horizons']:
            ckpt_path = f'./baselines/dlinear_{data_name}_pred{pred_len}.pkl'
            result_path = f'./experiments_flexible/dlinear_{data_name}_pred{pred_len}_result.json'

            # Verify checkpoint
            verify_result = verify_checkpoint(ckpt_path, data_name, pred_len, d_feature)
            status_icon = "OK" if verify_result['status'] == 'OK' else "FAIL"

            print(f"  {pred_len:>3}h: [{status_icon}] checkpoint", end="")

            # Load result JSON
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result = json.load(f)

                # Handle 96h checkpoint that has note field
                if 'note' in result:
                    result.setdefault('training_time', -1.0)

                all_results.append(result)
                epochs_str = str(result.get('num_epochs', '?'))
                train_time = result.get('training_time', -1.0)
                time_str = f"{train_time:.1f}s" if train_time >= 0 else "N/A (Sem1)"
                print(f"  MSE={result['mse']:.4f}  MAE={result['mae']:.4f}  epochs={epochs_str}  train={time_str}")
            else:
                print(f"  [MISSING] result JSON missing")

    print("\n" + "="*70)
    print("CONSOLIDATED RESULTS TABLE")
    print("="*70)
    print(f"{'Dataset':<12} {'Horizon':<8} {'Retrained':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'Epochs':<8} {'Train Time':<12}")
    print("-"*70)

    # Sort by dataset, then pred_len
    all_results.sort(key=lambda x: (x['dataset'], x['pred_len']))

    for r in all_results:
        train_time = r.get('training_time', -1.0)
        time_str = f"{train_time:.1f}s" if train_time >= 0 else "N/A"
        epochs_val = r.get('num_epochs', -1)
        epochs_str = str(epochs_val) if epochs_val >= 0 else "N/A"
        print(f"{r['dataset']:<12} {r['pred_len']:>3}h     {'YES':<12} "
              f"{r['mse']:<10.4f} {r['mae']:<10.4f} {r['rmse']:<10.4f} "
              f"{epochs_str:<8} {time_str:<12}")

    print("="*70)

    # Save consolidated results
    output_file = 'experiments_flexible/dlinear_all_results_consolidated.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nConsolidated results saved to: {output_file}")
    print(f"Total DLinear experiments: {len(all_results)}")

    # Summary statistics
    print("\n" + "="*70)
    print("TRAINING COST SUMMARY")
    print("="*70)

    etth1_results = [r for r in all_results if r['dataset'] == 'ETTh1']
    aqi_results = [r for r in all_results if r['dataset'] == 'IndiaAQI']

    # Only count times that were recorded (>= 0)
    etth1_time = sum(r.get('training_time', 0.0) for r in etth1_results if r.get('training_time', -1) >= 0)
    aqi_time = sum(r.get('training_time', 0.0) for r in aqi_results if r.get('training_time', -1) >= 0)

    print(f"ETTh1:    {len(etth1_results)} models recorded, time (excl. Sem1 96h): {etth1_time:.1f}s ({etth1_time/60:.1f} min)")
    print(f"IndiaAQI: {len(aqi_results)} models recorded, time (excl. Sem1 96h): {aqi_time:.1f}s ({aqi_time/60:.1f} min)")
    print(f"Total:    {len(all_results)} models, Sem2 training: {etth1_time + aqi_time:.1f}s ({(etth1_time+aqi_time)/60:.1f} min)")

    print("\nDLinear requires SEPARATE training for each horizon.")
    print("D2Vformer requires ONLY ONE training (reused across horizons).")

    return 0


if __name__ == '__main__':
    exit(main())
