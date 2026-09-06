#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flexible Forecasting Checkpoint Test
Tests the EXISTING trained checkpoint with variable pred_len.

NO RETRAINING — uses experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl as-is.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import argparse
from data.dataset import MyDataset
from utils.get_data import get_data

# Import the flexible variant
from model.D2Vformer_simple_flexible import D2Vformer_simple_flexible


def create_flexible_dataset(data, seq_len, label_len, pred_len):
    """
    Create a dataset that provides samples with the specified pred_len.

    Note: This is compatible with standard MyDataset, just with variable pred_len.
    """
    return MyDataset(data, seq_len=seq_len, label_len=label_len, pred_len=pred_len)


def test_checkpoint_with_pred_len(model, testset, pred_len, mean, scale, d_feature=7, max_batches=50):
    """
    Run inference on testset with specified pred_len.

    Args:
        model: Trained D2Vformer_simple_flexible
        testset: MyDataset with the target pred_len
        pred_len: Prediction horizon to test
        mean, scale: For denormalization
        d_feature: Number of features
        max_batches: Limit batches to avoid long runtime (use None for full test)

    Returns:
        dict with mse, mae, rmse
    """
    model.eval()

    preds = []
    trues = []

    batch_size = 32
    num_samples = min(len(testset), max_batches * batch_size) if max_batches else len(testset)

    print(f"  Evaluating on {num_samples} test samples...")

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_samples = [testset[j] for j in range(i, end_idx)]

            # Stack batch
            batch_x = torch.stack([torch.tensor(s[0]) for s in batch_samples]).float()
            batch_y = torch.stack([torch.tensor(s[1]) for s in batch_samples]).float()
            batch_x_mark = torch.stack([torch.tensor(s[2]) for s in batch_samples]).float()
            batch_y_mark = torch.stack([torch.tensor(s[3]) for s in batch_samples]).float()

            # Forward pass with dynamic pred_len
            pred = model(batch_x, batch_x_mark, batch_y, batch_y_mark, mode='test', pred_len=pred_len)

            # Extract predictions and ground truth for this pred_len
            preds.append(pred.cpu().numpy())
            trues.append(batch_y.cpu().numpy()[:, -pred_len:, :])

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Compute metrics (normalized space)
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(mse)

    # Denormalize for interpretability
    preds_denorm = preds * scale + mean
    trues_denorm = trues * scale + mean

    mse_denorm = np.mean((preds_denorm - trues_denorm) ** 2)
    mae_denorm = np.mean(np.abs(preds_denorm - trues_denorm))
    rmse_denorm = np.sqrt(mse_denorm)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mse_denorm': float(mse_denorm),
        'mae_denorm': float(mae_denorm),
        'rmse_denorm': float(rmse_denorm),
        'num_samples': len(preds)
    }


def main():
    print("="*70)
    print("D2Vformer Flexible Forecasting — Checkpoint Test")
    print("="*70)
    print("\nTesting EXISTING checkpoint with variable pred_len")
    print("Checkpoint: experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl")
    print("Training pred_len: 96h")
    print("Testing horizons: 48h, 72h, 96h, 192h, 336h")
    print("\nNO RETRAINING — using trained weights as-is.")
    print("="*70)

    # Configuration (match the training config from exp11)
    dataset = 'ETTh1'
    data_path = './datasets/ETT-small/ETTh1.csv'
    mark_path = './datasets/ETT-small/china.csv'
    checkpoint_path = './experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl'

    seq_len = 96
    label_len = 48
    training_pred_len = 96  # What the model was trained on
    d_feature = 7

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Make sure you're running from D2Vformer/D2Vformer/ directory")
        return 1

    # Load data
    print(f"\nLoading {dataset} data...")
    args_dummy = argparse.Namespace(d_mark=27, mark_index=[0,1,2,3])
    _, _, test, mean, scale, _ = get_data(data_path, mark_path, args=args_dummy)
    print(f"Test set size: {len(test[0])} samples")

    # Create model config
    model_args = argparse.Namespace(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=training_pred_len,  # Training horizon
        d_feature=d_feature,
        c_out=d_feature,
        d_model=512,
        d_ff=1024,
        T2V_outmodel=64,  # exp11 used T2V=64
        mark_index=[0, 1, 2, 3],
        d_mark=27,
        dropout=0.1,
        patch_len=16,
        stride=8,
        n_heads=3,
        save_path='./test_output',
        output_path='./test_output'
    )

    # Load model
    print(f"\nLoading checkpoint...")
    model = D2Vformer_simple_flexible(model_args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Checkpoint loaded successfully")

    # Test multiple horizons
    test_horizons = [48, 72, 96, 192, 336]
    results = []

    print(f"\n{'='*70}")
    print("TESTING FLEXIBLE FORECASTING")
    print(f"{'='*70}\n")

    for pred_len in test_horizons:
        print(f"Testing pred_len={pred_len}h...")

        # Create test dataset with this pred_len
        testset = create_flexible_dataset(test, seq_len, label_len, pred_len)

        # Run evaluation (limit to 50 batches for speed)
        result = test_checkpoint_with_pred_len(
            model, testset, pred_len, mean, scale, d_feature, max_batches=50
        )

        result['pred_len'] = pred_len
        result['retrained'] = False
        results.append(result)

        print(f"  MSE: {result['mse']:.4f}, MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
        print()

    # Summary table
    print(f"{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Horizon':<10} {'Retrained':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10}")
    print(f"{'-'*70}")

    for r in results:
        horizon = f"{r['pred_len']}h"
        retrained = "NO"
        print(f"{horizon:<10} {retrained:<12} {r['mse']:<10.4f} {r['mae']:<10.4f} {r['rmse']:<10.4f}")

    print(f"{'='*70}")

    # Analysis
    print("\nANALYSIS:")
    print(f"- Training horizon: {training_pred_len}h")
    print(f"- All horizons tested with SAME checkpoint (no retraining)")

    mse_96 = [r['mse'] for r in results if r['pred_len'] == 96][0]
    mse_192 = [r['mse'] for r in results if r['pred_len'] == 192][0]
    mse_336 = [r['mse'] for r in results if r['pred_len'] == 336][0]

    print(f"\nMSE at training horizon (96h): {mse_96:.4f}")
    print(f"MSE at 192h (2x training): {mse_192:.4f} ({((mse_192/mse_96-1)*100):+.1f}%)")
    print(f"MSE at 336h (3.5x training): {mse_336:.4f} ({((mse_336/mse_96-1)*100):+.1f}%)")

    if mse_192 > mse_96 * 1.5:
        print("\n[WARNING] Significant accuracy degradation at longer horizons.")
        print("This is expected — model trained on 96h, extrapolating to 192h/336h.")
    else:
        print("\n[NOTE] Model generalizes reasonably to longer horizons.")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nNext step: Compare against DLinear (which requires retraining per horizon)")

    # Save results
    import json
    output_file = 'experiments_flexible/etth1_flexible_results.json'
    os.makedirs('experiments_flexible', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == '__main__':
    exit(main())
