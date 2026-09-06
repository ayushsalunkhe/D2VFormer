# -*- coding: utf-8 -*-
"""
plot_comparison.py — Side-by-side comparison: DLinear vs D2Vformer_s
Generates a bar chart + forecast overlay for the guide meeting.

Run from D2Vformer/D2Vformer:
    python baselines/plot_comparison.py
"""
import os, sys, argparse, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.get_data import get_data
from data.dataset import MyDataset
from model.D2Vformer_simple import D2Vformer_simple
from baselines.run_dlinear import DLinear

# ─── Results (collected from training runs) ─────────────────────────────────
RESULTS = {
    'ETTh1': {
        'DLinear':    {'mse': 0.4033, 'mae': 0.4358},
        'D2Vformer':  {'mse': 0.6982, 'mae': 0.6193},
    },
    'Delhi AQI': {
        'DLinear':    {'mse': 0.1658, 'mae': 0.2710},
        'D2Vformer':  {'mse': 0.2212, 'mae': 0.3072},
    },
}

def make_d2v_args(data_name, d_feature, t2v):
    return argparse.Namespace(
        model_name='D2Vformer_s', data_name=data_name,
        seq_len=96, label_len=48, pred_len=96,
        batch_size=64, d_feature=d_feature, c_out=d_feature, features='M',
        d_model=512, d_ff=1024, dropout=0.1, patch_len=16, stride=8, n_heads=3,
        T2V_outmodel=t2v, mark_index=[0,1,2,3], d_mark=27,
        output_path='./visualizations', save_path='./visualizations',
        loss='normal', quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    )

def load_d2v(data_name, d_feature):
    # Explicit checkpoint paths for known-good results
    paths = {
        'ETTh1':    'experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl',
        'IndiaAQI': 'experiments/exp16/D2Vformer_s/IndiaAQI_best_model.pkl',
    }
    ckpt_path = paths.get(data_name)
    if not ckpt_path or not os.path.exists(ckpt_path):
        # Fallback to auto-find
        pkls = sorted(
            glob.glob(f'experiments/exp*/D2Vformer_s/{data_name}_best_model.pkl'),
            key=os.path.getmtime)
        assert pkls, f'No D2Vformer checkpoint for {data_name}'
        ckpt_path = pkls[-1]

    # Read T2V directly from checkpoint weights (hparam.yaml can be stale)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    t2v = ckpt['model']['Date2Vec.freq_upsampler_real.weight'].shape[0]
    print(f'Loaded {data_name}: {ckpt_path} (T2V={t2v} from checkpoint weights)')

    args = make_d2v_args(data_name, d_feature, t2v)
    model = D2Vformer_simple(args)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args

def load_dlinear(data_name, d_feature):
    pkl = f'./baselines/dlinear_{data_name}_pred96.pkl'
    assert os.path.exists(pkl), f'DLinear checkpoint missing: {pkl}'
    model = DLinear(seq_len=96, pred_len=96, d_feature=d_feature)
    ckpt  = torch.load(pkl, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def main():
    # ── Figure 1: Bar chart MSE + MAE comparison ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('D2Vformer vs DLinear — Forecasting Accuracy (96h ahead)',
                 fontsize=13, fontweight='bold')

    colors = {'DLinear': '#d98324', 'D2Vformer': '#0f3460'}
    datasets = list(RESULTS.keys())

    for ax_idx, metric in enumerate(['mse', 'mae']):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.32
        for i, model_name in enumerate(['DLinear', 'D2Vformer']):
            vals = [RESULTS[ds][model_name][metric] for ds in datasets]
            bars = ax.bar(x + i * width, vals, width,
                          label=model_name, color=colors[model_name], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Test {metric.upper()} (lower is better)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs('../../results', exist_ok=True)
    plt.savefig('../../results/comparison_bar.png', dpi=150, bbox_inches='tight')
    print('Saved: results/comparison_bar.png')

    # ── Figure 2: Forecast overlay — one test window each, ETTh1 and IndiaAQI
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 9))
    fig2.suptitle('Forecast Comparison: DLinear vs D2Vformer (96h → 96h)',
                  fontsize=13, fontweight='bold')

    dataset_configs = [
        ('ETTh1',    7, './datasets/ETT-small/china.csv',         './datasets/ETT-small/ETTh1.csv',         -1,  'Oil Temp (°C)'),
        ('IndiaAQI', 6, './datasets/india_aqi/delhi_mark.csv',    './datasets/india_aqi/delhi_aqi.csv',      0,  'PM2.5 (µg/m³)'),
    ]

    for row, (data_name, d_feat, mark_path, data_path, ch, ylabel) in enumerate(dataset_configs):
        d2v_model, d2v_args = load_d2v(data_name, d_feat)
        dl_model            = load_dlinear(data_name, d_feat)

        _, _, test, mean, scale, _ = get_data(data_path, mark_path, args=d2v_args)
        testset = MyDataset(test, seq_len=96, label_len=48, pred_len=96)
        idx = len(testset) // 3

        x, y, xm, ym = testset[idx]
        m, s = mean[ch], scale[ch]
        hist = x[:, ch] * s + m
        true = y[-96:, ch] * s + m

        # D2Vformer forecast
        with torch.no_grad():
            d2v_pred = d2v_model(
                torch.tensor(x).unsqueeze(0).float(),
                torch.tensor(xm).unsqueeze(0).float(),
                torch.tensor(y).unsqueeze(0).float(),
                torch.tensor(ym).unsqueeze(0).float(), 'test')
        d2v_fc = d2v_pred[0, :, ch].numpy() * s + m

        # DLinear forecast (no mark needed)
        dl_testset = MyDataset(test, seq_len=96, label_len=0, pred_len=96)
        x2, y2, _, _ = dl_testset[idx]
        with torch.no_grad():
            dl_pred = dl_model(torch.tensor(x2).unsqueeze(0).float())
        dl_fc = dl_pred[0, :, ch].numpy() * s + m

        mse_d2v = float(np.mean((d2v_fc - true)**2))
        mse_dl  = float(np.mean((dl_fc  - true)**2))

        for col, (fc, model_label, color) in enumerate([
            (d2v_fc, 'D2Vformer', '#e94560'),
            (dl_fc,  'DLinear',   '#d98324'),
        ]):
            ax = axes2[row, col]
            ax.plot(range(96),     hist, color='#0f3460', lw=1.5, label='History')
            ax.plot(range(96,192), true, color='#2e8b57', lw=1.8, label='Actual')
            ax.plot(range(96,192), fc,   color=color,     lw=1.8, ls='--', label=model_label)
            ax.axvline(96, color='gray', lw=1, ls=':')
            mse_val = mse_d2v if col == 0 else mse_dl
            ax.set_title(f'{data_name} — {model_label}  (MSE {mse_val:.3f})', fontsize=11)
            ax.set_xlabel('Hours')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/comparison_forecast.png', dpi=150, bbox_inches='tight')
    print('Saved: results/comparison_forecast.png')
    plt.close('all')

    # ── Print summary table ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(f'{"Dataset":<15} {"Model":<14} {"MSE":>8} {"MAE":>8} {"Winner?"}')
    print('='*60)
    for ds in datasets:
        for m in ['DLinear', 'D2Vformer']:
            r = RESULTS[ds][m]
            other = 'D2Vformer' if m == 'DLinear' else 'DLinear'
            winner = '✓ LOWER' if r['mse'] < RESULTS[ds][other]['mse'] else ''
            print(f'{ds:<15} {m:<14} {r["mse"]:>8.4f} {r["mae"]:>8.4f}  {winner}')
        print('-'*60)


if __name__ == '__main__':
    main()
