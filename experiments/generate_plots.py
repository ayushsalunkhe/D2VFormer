import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def generate_publication_plots(results_csv: str = 'results/raw/results.csv', plots_dir: str = 'results/plots'):
    if not os.path.exists(results_csv):
        print(f"Results CSV not found: {results_csv}")
        return
        
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    
    datasets = df['dataset'].unique()
    
    # Color palette
    colors = {
        'PureD2Vformer': '#1f77b4',       # Blue
        'DLinear': '#2ca02c',             # Green
        'Repository-D2Vformer': '#d62728',# Red
        'Persistence': '#7f7f7f'          # Gray
    }
    markers = {
        'PureD2Vformer': 'o',
        'DLinear': 's',
        'Repository-D2Vformer': '^',
        'Persistence': 'x'
    }
    
    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset]
        
        # 1. MSE vs Forecast Horizon
        plt.figure(figsize=(8, 5))
        for model in sub_df['model'].unique():
            m_df = sub_df[sub_df['model'] == model].groupby('eval_horizon')['mse'].agg(['mean', 'std']).reset_index()
            label = f"{model} (Zero-Shot)" if model == 'PureD2Vformer' else f"{model} (Retrained)" if model != 'Persistence' else model
            plt.plot(m_df['eval_horizon'], m_df['mean'], marker=markers.get(model, 'o'), label=label,
                     color=colors.get(model, '#333333'), linewidth=2, markersize=7)
            if 'std' in m_df and m_df['std'].notna().any():
                plt.fill_between(m_df['eval_horizon'], m_df['mean'] - m_df['std'], m_df['mean'] + m_df['std'],
                                 alpha=0.15, color=colors.get(model, '#333333'))
                
        plt.title(f"{dataset} — Forecasting MSE across Prediction Horizons")
        plt.xlabel("Prediction Horizon (O)")
        plt.ylabel("Test Mean Squared Error (MSE)")
        plt.xticks([24, 48, 96, 192, 336, 720])
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset}_mse_vs_horizon.png"), dpi=300)
        plt.close()
        
        # 2. MAE vs Forecast Horizon
        plt.figure(figsize=(8, 5))
        for model in sub_df['model'].unique():
            m_df = sub_df[sub_df['model'] == model].groupby('eval_horizon')['mae'].agg(['mean', 'std']).reset_index()
            label = f"{model} (Zero-Shot)" if model == 'PureD2Vformer' else f"{model} (Retrained)" if model != 'Persistence' else model
            plt.plot(m_df['eval_horizon'], m_df['mean'], marker=markers.get(model, 'o'), label=label,
                     color=colors.get(model, '#333333'), linewidth=2, markersize=7)
            if 'std' in m_df and m_df['std'].notna().any():
                plt.fill_between(m_df['eval_horizon'], m_df['mean'] - m_df['std'], m_df['mean'] + m_df['std'],
                                 alpha=0.15, color=colors.get(model, '#333333'))
                
        plt.title(f"{dataset} — Forecasting MAE across Prediction Horizons")
        plt.xlabel("Prediction Horizon (O)")
        plt.ylabel("Test Mean Absolute Error (MAE)")
        plt.xticks([24, 48, 96, 192, 336, 720])
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset}_mae_vs_horizon.png"), dpi=300)
        plt.close()
        
        # 3. Attention Entropy vs Horizon for PureD2Vformer
        p_df = sub_df[sub_df['model'] == 'PureD2Vformer'].groupby('eval_horizon')['normalized_attention_entropy'].agg(['mean', 'std']).reset_index()
        if not p_df.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(p_df['eval_horizon'], p_df['mean'], marker='o', color='#9467bd', linewidth=2, markersize=8, label='Normalized Entropy H_norm')
            plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Max (Uniform Attention)')
            plt.axvline(48, color='blue', linestyle=':', alpha=0.7, label='Training Horizon (O_train=48)')
            plt.title(f"{dataset} — PureD2Vformer Cross-Temporal Attention Entropy vs Horizon")
            plt.xlabel("Prediction Horizon (O)")
            plt.ylabel("Normalized Attention Entropy [H(A) / log(L)]")
            plt.xticks([24, 48, 96, 192, 336, 720])
            plt.ylim(0.0, 1.1)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{dataset}_attention_entropy_vs_horizon.png"), dpi=300)
            plt.close()

    # 4. Parameter Count vs Horizon (Global comparison)
    plt.figure(figsize=(8, 5))
    param_df = df[df['dataset'] == datasets[0]].drop_duplicates(subset=['model', 'eval_horizon'])
    for model in param_df['model'].unique():
        if model == 'Persistence':
            continue
        m_df = param_df[param_df['model'] == model]
        label = f"{model} (Parameter-Free Horizon)" if model == 'PureD2Vformer' else f"{model} (Horizon-Dependent)"
        plt.plot(m_df['eval_horizon'], m_df['parameter_count'], marker=markers.get(model, 'o'), label=label,
                 color=colors.get(model, '#333333'), linewidth=2, markersize=7)
                 
    plt.title("Trainable Parameter Count vs Forecast Horizon")
    plt.xlabel("Prediction Horizon (O)")
    plt.ylabel("Trainable Parameter Count")
    plt.xticks([24, 48, 96, 192, 336, 720])
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "parameter_count_vs_horizon.png"), dpi=300)
    plt.close()

    # 5. Cumulative Training Cost Comparison
    plt.figure(figsize=(8, 5))
    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset]
        train_cost = sub_df.groupby(['model', 'seed'])['training_time_sec'].sum().groupby('model').mean().reset_index()
        # Bar chart of training time
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(train_cost['model'], train_cost['training_time_sec'], color=[colors.get(m, '#333333') for m in train_cost['model']])
        plt.title(f"{dataset} — Total Cumulative Training Time across all 6 Horizons")
        plt.ylabel("Total Training Time (Seconds)")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}s", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset}_training_time_comparison.png"), dpi=300)
        plt.close()

    print(f"[SUCCESS] All research plots generated and saved to: {plots_dir}")

if __name__ == '__main__':
    generate_publication_plots()
