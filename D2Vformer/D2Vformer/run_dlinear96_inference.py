#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLinear 96h Inference Script
Loads the existing 96h DLinear checkpoints and evaluates them to generate result JSONs.
NO TRAINING — inference only.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines'))

import torch
import numpy as np
import json
import argparse

from baselines.run_dlinear_multihorizon import DLinear
from utils.get_data import get_data
from data.dataset import MyDataset
from torch.utils.data import DataLoader

CONFIGS = {
    'ETTh1': {
        'd_feature': 7,
        'data_path': './datasets/ETT-small/ETTh1.csv',
        'mark_path': './datasets/ETT-small/china.csv',
    },
    'IndiaAQI': {
        'd_feature': 6,
        'data_path': './datasets/india_aqi/delhi_aqi.csv',
        'mark_path': './datasets/india_aqi/delhi_mark.csv',
    }
}


def eval_96h_checkpoint(data_name, pred_len=96, seq_len=96, batch_size=64):
    config = CONFIGS[data_name]
    d_feature = config['d_feature']

    ckpt_path = f'./baselines/dlinear_{data_name}_pred{pred_len}.pkl'
    result_path = f'./experiments_flexible/dlinear_{data_name}_pred{pred_len}_result.json'

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"DLinear {data_name} pred{pred_len}h — INFERENCE ONLY")
    print(f"{'='*60}")
    print(f"Loading checkpoint: {ckpt_path}")

    # Load data
    dummy_args = argparse.Namespace(d_mark=27, mark_index=[0,1,2,3])
    train, valid, test, mean, scale, _ = get_data(
        config['data_path'], config['mark_path'], args=dummy_args)

    testset = MyDataset(test, seq_len=seq_len, label_len=0, pred_len=pred_len)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"Test set size: {len(testset)} samples")

    # Load model
    model = DLinear(seq_len=seq_len, pred_len=pred_len, d_feature=d_feature)
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Verify architecture
    weight_shape = ckpt['model']['linear_trend.0.weight'].shape
    expected = (pred_len, seq_len)
    print(f"Checkpoint weight shape: {weight_shape} (expected {expected})")
    assert weight_shape == expected, f"Architecture mismatch: {weight_shape} != {expected}"

    model.load_state_dict(ckpt['model'])
    model.eval()

    # Try to get training metadata
    saved_epoch = ckpt.get('epoch', 'unknown')
    print(f"Checkpoint epoch: {saved_epoch}")

    # Run inference
    preds, trues = [], []
    t_start = time.time()
    with torch.no_grad():
        for bx, by, bxm, bym in testloader:
            preds.append(model(bx.float()).numpy())
            trues.append(by[:, -pred_len:, :].numpy())
    inference_time = time.time() - t_start

    p = np.concatenate(preds)
    t = np.concatenate(trues)

    mse  = float(np.mean((p - t) ** 2))
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(mse))

    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Inference time: {inference_time:.2f}s")

    # training_time is unknown for the old 96h checkpoint (it was from Semester 1)
    # We mark it as -1 to indicate "not recorded"
    result = {
        'dataset': data_name,
        'model': 'DLinear',
        'pred_len': pred_len,
        'retrained': True,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'training_time': -1.0,   # Semester 1 checkpoint — time not recorded
        'inference_time': inference_time,
        'num_epochs': saved_epoch if isinstance(saved_epoch, int) else -1,
        'checkpoint': ckpt_path,
        'note': 'Semester 1 checkpoint — training time not recorded'
    }

    os.makedirs('experiments_flexible', exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {result_path}")

    return result


if __name__ == '__main__':
    results = []
    for ds in ['ETTh1', 'IndiaAQI']:
        r = eval_96h_checkpoint(ds, pred_len=96)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print("96h DLinear Evaluation Complete")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['dataset']:12s} 96h: MSE={r['mse']:.4f}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}")
