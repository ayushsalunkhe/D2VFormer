# -*- coding: utf-8 -*-
"""
plot_forecast.py — First reproduction artifact for guide meetings.

Loads the best D2Vformer_s checkpoint, runs it on the ETTh1 test set, and saves
a predicted-vs-actual graph (denormalized, OT = oil temperature channel).

Usage (from D2Vformer/D2Vformer):
    python plot_forecast.py --exp_dir experiments/expN   (defaults to latest exp)
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.D2Vformer_simple import D2Vformer_simple
from utils.get_data import get_data
from data.dataset import MyDataset
from torch.utils.data import DataLoader


def build_args():
    # Mirror of the training configuration (ETTh1, 96 -> 96)
    ns = argparse.Namespace(
        model_name='D2Vformer_s', data_name='ETTh1', seq_len=96, label_len=48,
        pred_len=96, batch_size=64, d_feature=7, c_out=7, features='M',
        d_model=512, d_ff=1024, dropout=0.1, patch_len=16, stride=8, n_heads=3,
        T2V_outmodel=64, mark_index=[0, 1, 2, 3], d_mark=27,
        output_path='./visualizations', save_path='./visualizations',
        loss='normal', quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    return ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default=None, help='experiments/expN dir with the checkpoint')
    parser.add_argument('--sample', type=int, default=100, help='test-set window index to plot')
    parser.add_argument('--out', default='../../results/ETTh1_pred96_forecast.png')
    cli = parser.parse_args()

    args = build_args()

    # Locate checkpoint: newest exp dir containing a best-model file
    if cli.exp_dir is None:
        candidates = sorted(glob.glob('experiments/exp*/D2Vformer_s/ETTh1_best_model.pkl'),
                            key=os.path.getmtime)
        if not candidates:
            sys.exit('No checkpoint found — train first.')
        ckpt_path = candidates[-1]
    else:
        ckpt_path = os.path.join(cli.exp_dir, 'D2Vformer_s', 'ETTh1_best_model.pkl')
    print('Using checkpoint:', ckpt_path)

    # Data (same split logic as training)
    train, valid, test, mean, scale, dim = get_data(
        './datasets/ETT-small/ETTh1.csv', './datasets/ETT-small/china.csv', args=args)
    testset = MyDataset(test, seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
    print(f'Test windows: {len(testset)}')

    # Model
    model = D2Vformer_simple(args)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}")

    # One window
    idx = min(cli.sample, len(testset) - 1)
    x, y, x_mark, y_mark = testset[idx]
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    x_mark = torch.tensor(x_mark, dtype=torch.float32).unsqueeze(0)
    y_mark = torch.tensor(y_mark, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x, x_mark, y_t, y_mark, 'test')  # [1, pred_len, D]

    # Denormalize (channel -1 = OT, oil temperature)
    ch = -1
    m, s = mean[ch], scale[ch]
    hist = x[0, :, ch].numpy() * s + m
    true = y_t[0, -args.pred_len:, ch].numpy() * s + m
    fc = pred[0, :, ch].numpy() * s + m

    mse_norm = float(np.mean((pred[0, :, :].numpy() - y_t[0, -args.pred_len:, :].numpy()) ** 2))

    # Plot
    L, P = args.seq_len, args.pred_len
    t_hist, t_fut = np.arange(L), np.arange(L, L + P)
    plt.figure(figsize=(12, 5))
    plt.plot(t_hist, hist, color='#0f3460', lw=1.6, label='History (96 h input)')
    plt.plot(t_fut, true, color='#2e8b57', lw=1.8, label='Actual future')
    plt.plot(t_fut, fc, color='#e94560', lw=1.8, ls='--', label='D2Vformer forecast')
    plt.axvline(L, color='gray', lw=1, ls=':')
    plt.text(L + 1, plt.ylim()[1], ' forecast starts', va='top', color='gray', fontsize=9)
    plt.title(f'D2Vformer — ETTh1 oil temperature, 96 h ahead (test window #{idx}, norm. MSE {mse_norm:.3f})')
    plt.xlabel('Hours')
    plt.ylabel('Oil temperature (°C)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    plt.savefig(cli.out, dpi=150)
    print('Saved plot to', os.path.abspath(cli.out))


if __name__ == '__main__':
    main()
