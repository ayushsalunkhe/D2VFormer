# -*- coding: utf-8 -*-
"""
plot_india_aqi.py — Forecast graph for Delhi AQI (PM2.5) — India extension dataset.
Loads the best IndiaAQI checkpoint and plots history vs actual vs D2Vformer forecast
for the PM2.5 channel (channel 0).

Usage (from D2Vformer/D2Vformer):
    python plot_india_aqi.py
"""
import argparse, glob, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.D2Vformer_simple import D2Vformer_simple
from utils.get_data import get_data
from data.dataset import MyDataset


def build_args(t2v=36):
    return argparse.Namespace(
        model_name='D2Vformer_s', data_name='IndiaAQI',
        seq_len=96, label_len=48, pred_len=96,
        batch_size=64, d_feature=6, c_out=6, features='M',
        d_model=512, d_ff=1024, dropout=0.1, patch_len=16, stride=8, n_heads=3,
        T2V_outmodel=t2v, mark_index=[0,1,2,3], d_mark=27,
        output_path='./visualizations', save_path='./visualizations',
        loss='normal', quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    )


def main():
    # Find latest IndiaAQI checkpoint
    pkls = sorted(
        glob.glob('experiments/exp*/D2Vformer_s/IndiaAQI_best_model.pkl'),
        key=os.path.getmtime)
    if not pkls:
        sys.exit('No IndiaAQI checkpoint found — train first.')

    ckpt_path = pkls[-1]
    print('Using checkpoint:', ckpt_path)

    # Read T2V from hparam — handle PyTorch-specific YAML tags gracefully
    import yaml, re
    hp_file = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'hparam.yaml')
    try:
        with open(hp_file) as f:
            hp = yaml.safe_load(f)
    except Exception:
        hp = {}
        with open(hp_file) as f:
            for line in f:
                m = re.match(r'^(\w+):\s*(.+)$', line.strip())
                if m:
                    hp[m.group(1)] = m.group(2).strip().strip("'\"")
    t2v = int(hp.get('T2V_outmodel', 36))
    print(f'T2V_outmodel from hparam: {t2v}')

    args = build_args(t2v)
    _, _, test, mean, scale, _ = get_data(
        './datasets/india_aqi/delhi_aqi.csv',
        './datasets/india_aqi/delhi_mark.csv',
        args=args)
    testset = MyDataset(test, seq_len=96, label_len=48, pred_len=96)
    print(f'Test windows: {len(testset)}')

    model = D2Vformer_simple(args)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}")

    # Plot 3 windows: one each from winter (high pollution), monsoon (clean), post-monsoon
    # test set covers roughly 2022 — pick windows accordingly
    n = len(testset)
    windows = [n // 4, n // 2, 3 * n // 4]
    labels  = ['Winter (high pollution)', 'Post-monsoon', 'Summer']
    colors_actual = ['#2e8b57', '#2e8b57', '#2e8b57']

    POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

    # ── Multi-panel: 3 time windows × top row PM2.5, bottom row O3 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('D2Vformer — Delhi AQI Forecast (India Extension Dataset)',
                 fontsize=14, fontweight='bold')

    mse_list = []
    for col, (idx, season) in enumerate(zip(windows, labels)):
        x, y, xm, ym = testset[idx]
        with torch.no_grad():
            pred = model(
                torch.tensor(x).unsqueeze(0).float(),
                torch.tensor(xm).unsqueeze(0).float(),
                torch.tensor(y).unsqueeze(0).float(),
                torch.tensor(ym).unsqueeze(0).float(), 'test')
        p_np = pred[0].numpy()
        y_np = y[-96:]
        mse = float(np.mean((p_np - y_np) ** 2))
        mse_list.append(mse)

        for row, ch in enumerate([0, 5]):   # PM2.5 (0) and O3 (5)
            ax = axes[row, col]
            m, s = mean[ch], scale[ch]
            hist = x[:, ch] * s + m
            true = y[-96:, ch] * s + m
            fc   = p_np[:, ch] * s + m

            ax.plot(range(96), hist, color='#0f3460', lw=1.4, label='History')
            ax.plot(range(96, 192), true, color='#2e8b57', lw=1.8, label='Actual')
            ax.plot(range(96, 192), fc, color='#e94560', lw=1.8, ls='--', label='Forecast')
            ax.axvline(96, color='gray', lw=1, ls=':')
            ax.set_title(f'{season}\n{POLLUTANTS[ch]}', fontsize=10)
            ax.set_xlabel('Hours')
            ax.set_ylabel(f'{POLLUTANTS[ch]} (µg/m³)')
            ax.grid(alpha=0.3)
            if col == 0 and row == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    out = '../../results/IndiaAQI_forecast.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {os.path.abspath(out)}')
    print(f'Mean test MSE across 3 windows: {np.mean(mse_list):.4f}')


if __name__ == '__main__':
    main()
