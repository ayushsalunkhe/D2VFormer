# -*- coding: utf-8 -*-
"""
D2Vformer Forecasting Demo — Streamlit Web App
Interactive UI for time-series forecasting with D2Vformer and DLinear baseline.

Run from the project root:
    streamlit run streamlit_app.py
"""
import streamlit as st
import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add D2Vformer to path
sys.path.insert(0, str(Path(__file__).parent / 'D2Vformer' / 'D2Vformer'))

import argparse

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='D2Vformer Forecast Demo',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─── Imports (with graceful error handling) ───────────────────────────────────
try:
    from model.D2Vformer_simple_flexible import D2Vformer_simple_flexible
    from baselines.run_dlinear import DLinear
    from utils.get_data import get_data
    from data.dataset import MyDataset
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# ─── Dataset Config ──────────────────────────────────────────────────────────
DATASETS = {
    'ETTh1 (Electricity Transformer Temperature)': {
        'key': 'ETTh1',
        'd_feature': 7,
        'mark_path': './D2Vformer/D2Vformer/datasets/ETT-small/china.csv',
        'data_path': './D2Vformer/D2Vformer/datasets/ETT-small/ETTh1.csv',
        'channel_names': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'default_ch': 6,
    },
    'Delhi AQI (India Air Quality)': {
        'key': 'IndiaAQI',
        'd_feature': 6,
        'mark_path': './D2Vformer/D2Vformer/datasets/india_aqi/delhi_mark.csv',
        'data_path': './D2Vformer/D2Vformer/datasets/india_aqi/delhi_aqi.csv',
        'channel_names': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
        'default_ch': 0,
    },
}

# D2Vformer uses ONE checkpoint per dataset regardless of horizon
D2V_CHECKPOINTS = {
    'ETTh1':    './D2Vformer/D2Vformer/experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl',
    'IndiaAQI': './D2Vformer/D2Vformer/experiments/exp16/D2Vformer_s/IndiaAQI_best_model.pkl',
}

# DLinear uses ONE checkpoint PER horizon per dataset
def dlinear_checkpoint(dataset_key, pred_len):
    return f'./D2Vformer/D2Vformer/baselines/dlinear_{dataset_key}_pred{pred_len}.pkl'

HORIZONS = [48, 72, 96, 192, 336]


# ─── Checkpoint Status ───────────────────────────────────────────────────────
def get_checkpoint_status():
    status = {'d2v': {}, 'dlinear': {}}
    for ds_key in ['ETTh1', 'IndiaAQI']:
        ckpt = D2V_CHECKPOINTS[ds_key]
        status['d2v'][ds_key] = os.path.exists(ckpt)
        status['dlinear'][ds_key] = {}
        for h in HORIZONS:
            status['dlinear'][ds_key][h] = os.path.exists(dlinear_checkpoint(ds_key, h))
    return status


# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_d2v_model(dataset_key):
    """Load D2Vformer model — ONE checkpoint, reused for all horizons."""
    ckpt_file = D2V_CHECKPOINTS.get(dataset_key)
    if not ckpt_file or not os.path.exists(ckpt_file):
        return None, f"D2Vformer checkpoint not found: {ckpt_file}"

    cfg = None
    for v in DATASETS.values():
        if v['key'] == dataset_key:
            cfg = v
            break
    if cfg is None:
        return None, f"Unknown dataset key: {dataset_key}"

    d_feat = cfg['d_feature']

    try:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        t2v = ckpt['model']['Date2Vec.freq_upsampler_real.weight'].shape[0]

        args = argparse.Namespace(
            model_name='D2Vformer_s', data_name=dataset_key,
            seq_len=96, label_len=48, pred_len=96,
            batch_size=64, d_feature=d_feat, c_out=d_feat, features='M',
            d_model=512, d_ff=1024, dropout=0.1, patch_len=16, stride=8, n_heads=3,
            T2V_outmodel=t2v, mark_index=[0, 1, 2, 3], d_mark=27,
            output_path='./visualizations', save_path='./visualizations',
            loss='normal', quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        model = D2Vformer_simple_flexible(args)
        model.load_state_dict(ckpt['model'])
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading D2Vformer: {e}"


@st.cache_resource
def load_dlinear_model(dataset_key, pred_len):
    """Load DLinear model for a SPECIFIC horizon."""
    cfg = None
    for v in DATASETS.values():
        if v['key'] == dataset_key:
            cfg = v
            break
    if cfg is None:
        return None, f"Unknown dataset key: {dataset_key}"

    ckpt_path = dlinear_checkpoint(dataset_key, pred_len)
    if not os.path.exists(ckpt_path):
        return None, f"DLinear checkpoint not found for {dataset_key} {pred_len}h"

    try:
        d_feat = cfg['d_feature']
        model = DLinear(seq_len=96, pred_len=pred_len, d_feature=d_feat)
        dl_ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(dl_ckpt['model'])
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading DLinear: {e}"


@st.cache_data
def load_dataset(dataset_key):
    """Load test data split."""
    cfg = None
    for v in DATASETS.values():
        if v['key'] == dataset_key:
            cfg = v
            break
    try:
        dummy_args = argparse.Namespace(d_mark=27, mark_index=[0, 1, 2, 3])
        _, _, test, mean, scale, _ = get_data(cfg['data_path'], cfg['mark_path'], args=dummy_args)
        df = pd.read_csv(cfg['data_path'])
        df['date'] = pd.to_datetime(df['date'])
        return test, mean, scale, cfg, df, None
    except Exception as e:
        return None, None, None, None, None, str(e)


# ─── Inference ───────────────────────────────────────────────────────────────
def forecast_d2v(model, testset, idx, channel, mean, scale, pred_len):
    x, y, xm, ym = testset[idx]
    with torch.no_grad():
        pred = model(
            torch.tensor(x).unsqueeze(0).float(),
            torch.tensor(xm).unsqueeze(0).float(),
            torch.tensor(y).unsqueeze(0).float(),
            torch.tensor(ym).unsqueeze(0).float(),
            mode='test',
            pred_len=pred_len
        )
    fc = pred[0, :, channel].numpy()
    m, s = mean[channel], scale[channel]
    hist = x[:, channel] * s + m
    true = y[-pred_len:, channel] * s + m
    fc_denorm = fc * s + m
    mse  = float(np.mean((fc_denorm - true) ** 2))
    mae  = float(np.mean(np.abs(fc_denorm - true)))
    rmse = float(np.sqrt(mse))
    return hist, true, fc_denorm, mse, mae, rmse


def forecast_dlinear(model, testset, idx, channel, mean, scale, pred_len):
    x, y, _, _ = testset[idx]
    with torch.no_grad():
        pred = model(torch.tensor(x).unsqueeze(0).float())
    fc = pred[0, :, channel].numpy()
    m, s = mean[channel], scale[channel]
    hist = x[:, channel] * s + m
    true = y[-pred_len:, channel] * s + m
    fc_denorm = fc * s + m
    mse  = float(np.mean((fc_denorm - true) ** 2))
    mae  = float(np.mean(np.abs(fc_denorm - true)))
    rmse = float(np.sqrt(mse))
    return hist, true, fc_denorm, mse, mae, rmse


# ─── Load Pre-computed Summary Results ───────────────────────────────────────
@st.cache_data
def load_precomputed_results():
    """Load saved experiment JSON files for aggregate metric display."""
    results = {'d2v': {}, 'dlinear': {}}
    for ds, fname in [('ETTh1', 'etth1_flexible_results.json'),
                      ('IndiaAQI', 'indiaaqi_flexible_results.json')]:
        path = f'./D2Vformer/D2Vformer/experiments_flexible/{fname}'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results['d2v'][ds] = {r['pred_len']: r for r in data}

    for ds in ['ETTh1', 'IndiaAQI']:
        results['dlinear'][ds] = {}
        for h in HORIZONS:
            path = f'./D2Vformer/D2Vformer/experiments_flexible/dlinear_{ds}_pred{h}_result.json'
            if os.path.exists(path):
                with open(path) as f:
                    results['dlinear'][ds][h] = json.load(f)
    return results


# ─── App Layout ──────────────────────────────────────────────────────────────

st.title('📈 D2Vformer Time-Series Forecasting Demo')
st.markdown("""
**BE Major Project** — D2Vformer: Flexible Multi-Horizon Prediction with Date2Vec  
Compare D2Vformer (calendar-aware, **1 checkpoint for all horizons**) vs DLinear (**retrained per horizon**).
""")

if not IMPORTS_OK:
    st.error(f"Import error: {IMPORT_ERROR}")
    st.info("Make sure you're running from the project root with the correct venv activated.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header('Configuration')

    dataset_name = st.selectbox('Dataset', list(DATASETS.keys()), key='dataset_sel')
    dataset_key  = DATASETS[dataset_name]['key']

    pred_len = st.select_slider(
        'Forecast Horizon',
        options=HORIZONS,
        value=96,
        format_func=lambda x: f'{x}h',
        key='horizon_sel'
    )

    st.markdown('---')
    st.markdown('**Models to display:**')
    show_d2v = st.checkbox('D2Vformer', value=True, key='show_d2v')
    show_dl  = st.checkbox('DLinear Baseline', value=True, key='show_dl')

    st.markdown('---')
    st.markdown('**Test Window**')
    st.caption('Browse different time periods in the test split.')
    window_idx = st.number_input('Window Index', min_value=0, max_value=2000,
                                 value=500, step=50, key='window_idx')

    st.markdown('---')
    st.markdown('**System Status**')
    ck_status = get_checkpoint_status()

    d2v_ok = all(ck_status['d2v'].values())
    st.markdown(f"{'OK' if d2v_ok else 'MISSING'} **D2Vformer:** "
                f"{'1 checkpoint reused across 5 horizons (no retraining)' if d2v_ok else 'checkpoint missing'}")

    dl_all = sum(
        1 for ds in ['ETTh1', 'IndiaAQI']
        for h in HORIZONS if ck_status['dlinear'][ds][h]
    )
    st.markdown(f"{'OK' if dl_all == 10 else 'WARN'} **DLinear:** {dl_all}/10 checkpoints")

    dl_this = ck_status['dlinear'][dataset_key][pred_len]
    st.markdown(
        f"{'OK' if dl_this else 'MISSING'} **DLinear {pred_len}h ({dataset_key})**"
    )


# Load data
test, mean, scale, cfg, df_raw, data_err = load_dataset(dataset_key)
if data_err:
    st.error(f"Data loading error: {data_err}")
    st.stop()

channel_name = st.selectbox('Feature to forecast', cfg['channel_names'], key='channel_sel')
channel = cfg['channel_names'].index(channel_name)

testset_d2v = MyDataset(test, seq_len=96, label_len=48, pred_len=pred_len)
testset_dl  = MyDataset(test, seq_len=96, label_len=0,  pred_len=pred_len)

max_window = len(testset_d2v) - 1
if window_idx > max_window:
    st.warning(f'Window {window_idx} out of range. Using max={max_window}.')
    window_idx = max_window

test_start_idx = int(len(df_raw) * 0.8)
win_start      = test_start_idx + window_idx
win_end        = win_start + 96 + pred_len

if win_end <= len(df_raw):
    hist_dates     = df_raw.iloc[win_start:win_start + 96]['date']
    forecast_dates = df_raw.iloc[win_start + 96:win_end]['date']
    start_date     = hist_dates.iloc[0]
    end_date       = forecast_dates.iloc[-1]
    date_range_str = f"{start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}"
    season         = start_date.strftime('%B %Y')
else:
    date_range_str = "Date info unavailable"
    season         = "Unknown"
    start_date     = None

# Load models
d2v_model, d2v_err = (None, None)
dl_model,  dl_err  = (None, None)

if show_d2v:
    with st.spinner(f'Loading D2Vformer ({dataset_key})...'):
        d2v_model, d2v_err = load_d2v_model(dataset_key)
    if d2v_err:
        st.warning(f'D2Vformer: {d2v_err}')

if show_dl:
    with st.spinner(f'Loading DLinear {pred_len}h ({dataset_key})...'):
        dl_model, dl_err = load_dlinear_model(dataset_key, pred_len)
    if dl_err:
        st.warning(f'DLinear: {dl_err}')

# Run forecasts
st.markdown('---')
col_title, col_info = st.columns([3, 2])
with col_title:
    st.subheader(f'Forecast: {dataset_name} — {channel_name} ({pred_len}h horizon)')
with col_info:
    st.caption(f'Period: {date_range_str}')
    st.caption(f'Season: {season}')

results = {}
COLORS = {'D2Vformer': '#e94560', 'DLinear': '#2196F3'}

if show_d2v and d2v_model:
    try:
        hist, true, fc, mse, mae, rmse = forecast_d2v(
            d2v_model, testset_d2v, window_idx, channel, mean, scale, pred_len)
        results['D2Vformer'] = (hist, true, fc, mse, mae, rmse, COLORS['D2Vformer'])
    except Exception as e:
        st.warning(f'D2Vformer inference failed: {e}')

if show_dl and dl_model:
    try:
        hist, true, fc, mse, mae, rmse = forecast_dlinear(
            dl_model, testset_dl, window_idx, channel, mean, scale, pred_len)
        results['DLinear'] = (hist, true, fc, mse, mae, rmse, COLORS['DLinear'])
    except Exception as e:
        st.warning(f'DLinear inference failed: {e}')

if not results:
    st.warning('Select at least one model and ensure checkpoints are available.')
    st.stop()

# Plot
fig, ax = plt.subplots(figsize=(14, 5))

hist_ref = results[list(results.keys())[0]][0]
true_ref = results[list(results.keys())[0]][1]

x_hist = range(96)
x_pred = range(96, 96 + pred_len)

ax.plot(x_hist, hist_ref, color='#0f3460', lw=1.8, label='History (96h)', zorder=5)
ax.plot(x_pred, true_ref, color='#2e8b57', lw=2.2, label='Actual', zorder=5)

for model_name, (_, _, fc, mse, mae, rmse, color) in results.items():
    ckpt_label = '(1 ckpt, reused)' if model_name == 'D2Vformer' else f'(ckpt/{pred_len}h)'
    ax.plot(x_pred, fc, color=color, lw=2, ls='--',
            label=f'{model_name} {ckpt_label}  MSE={mse:.3f}', zorder=4)

ax.axvline(96, color='gray', lw=1.2, ls=':', alpha=0.8)
ax.set_xlabel('Hours from window start', fontsize=11)
ax.set_ylabel(channel_name, fontsize=11)
ax.set_title(f'{dataset_key} — {channel_name} | Window {window_idx} | {season} | {pred_len}h horizon',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)
ax.axvspan(96, 96 + pred_len, alpha=0.04, color='blue')

if start_date:
    ax.text(0.01, 0.98, f'Start: {start_date.strftime("%Y-%m-%d %H:%M")}',
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

plt.tight_layout()
st.pyplot(fig)

# Metrics table
st.markdown('#### Metrics (denormalized, single window)')
metrics_data = []
for model_name, (_, _, _, mse, mae, rmse, _) in results.items():
    ckpt_note = 'Single checkpoint (no retraining)' if model_name == 'D2Vformer' else f'Checkpoint for {pred_len}h'
    metrics_data.append({
        'Model': model_name,
        'MSE': f'{mse:.4f}',
        'MAE': f'{mae:.4f}',
        'RMSE': f'{rmse:.4f}',
        'Checkpoint Strategy': ckpt_note,
    })
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

# Aggregate benchmark results
with st.expander('Full Test-Set Benchmark Results (pre-computed)', expanded=False):
    precomp = load_precomputed_results()
    st.markdown(f'#### {dataset_key} — All Horizons (normalized MSE on full test set)')

    agg_rows = []
    for h in HORIZONS:
        row = {'Horizon': f'{h}h'}
        d2v_r = precomp['d2v'].get(dataset_key, {}).get(h)
        dl_r  = precomp['dlinear'].get(dataset_key, {}).get(h)
        row['D2Vformer MSE'] = f"{d2v_r['mse']:.4f}" if d2v_r else '—'
        row['D2Vformer MAE'] = f"{d2v_r['mae']:.4f}" if d2v_r else '—'
        row['DLinear MSE']   = f"{dl_r['mse']:.4f}"  if dl_r  else '—'
        row['DLinear MAE']   = f"{dl_r['mae']:.4f}"  if dl_r  else '—'
        agg_rows.append(row)

    st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)
    st.caption(
        'D2Vformer: single checkpoint reused across all horizons. '
        'DLinear: separate checkpoint per horizon. '
        'Normalized z-score space.'
    )

# Model info
st.markdown('---')
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **D2Vformer** *(Calendar-Aware Attention)*
    - Uses Date2Vec to encode absolute calendar timestamps
    - **1 trained checkpoint → all 5 horizons (no retraining)**
    - Flexible because timestamps are continuous, not fixed-length
    """)
with col2:
    st.markdown(f"""
    **DLinear** *(Linear Decomposition Baseline)*
    - Decomposes series into trend + seasonal components
    - **Requires separate model per horizon** (5 checkpoints)
    - Simple and fast; competitive on ETTh1
    - Currently loaded: {pred_len}h checkpoint
    """)

st.markdown('---')
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
<b>D2Vformer</b> — IEEE TNNLS 2026 &nbsp;|&nbsp; <b>DLinear</b> — AAAI 2023<br>
BE Major Project — Semester 2: Flexible Multi-Horizon Forecasting | Built with Streamlit
</div>
""", unsafe_allow_html=True)

