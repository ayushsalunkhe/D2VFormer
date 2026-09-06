#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLinear Multi-Horizon Training Script

Trains DLinear separately for each prediction horizon.
This demonstrates the retraining cost compared to D2Vformer's flexible forecasting.

Usage:
    python baselines/run_dlinear_multihorizon.py --data_name ETTh1 --pred_len 48
    python baselines/run_dlinear_multihorizon.py --data_name IndiaAQI --pred_len 192
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.get_data import get_data
from data.dataset import MyDataset
from utils.earlystopping import EarlyStopping
from torch.optim import Adam


# ─── Moving average decomposition ──────────────────────────────────────────
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual    = x - moving_mean
        return residual, moving_mean


# ─── DLinear model ──────────────────────────────────────────────────────────
class DLinear(nn.Module):
    """
    DLinear: decompose into trend + seasonal, then separate Linear per component.
    individual=True: one linear per channel (best variant per the paper).
    """
    def __init__(self, seq_len, pred_len, d_feature, individual=True, kernel_size=25):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.decomp   = SeriesDecomp(kernel_size)
        self.individual = individual

        if individual:
            self.linear_trend    = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(d_feature)])
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(d_feature)])
        else:
            self.linear_trend    = nn.Linear(seq_len, pred_len)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        if self.individual:
            out_t = torch.stack([self.linear_trend[i](trend[:, :, i])
                                 for i in range(trend.size(-1))], dim=-1)
            out_s = torch.stack([self.linear_seasonal[i](seasonal[:, :, i])
                                 for i in range(seasonal.size(-1))], dim=-1)
        else:
            out_t = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            out_s = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        return out_t + out_s


# ─── Training function ──────────────────────────────────────────────────────
def train_and_eval(data_name, d_feature, mark_path, data_path,
                   seq_len=96, pred_len=96, epochs=50, patience=5,
                   lr=0.001, batch_size=64, output_dir='./baselines'):
    """
    Train DLinear for specific pred_len and evaluate.

    Returns:
        dict with mse, mae, training_time, num_epochs, checkpoint_path
    """
    print(f"\n{'='*70}")
    print(f"Training DLinear — {data_name}, pred_len={pred_len}h")
    print(f"{'='*70}")

    import argparse as ap
    dummy_args = ap.Namespace(d_mark=27, mark_index=[0,1,2,3])
    train, valid, test, mean, scale, _ = get_data(data_path, mark_path, args=dummy_args)

    trainset = MyDataset(train, seq_len=seq_len, label_len=0, pred_len=pred_len)
    validset = MyDataset(valid, seq_len=seq_len, label_len=0, pred_len=pred_len)
    testset  = MyDataset(test,  seq_len=seq_len, label_len=0, pred_len=pred_len)

    print(f"Dataset sizes: train={len(trainset)}, val={len(validset)}, test={len(testset)}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  drop_last=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=True)

    model     = DLinear(seq_len=seq_len, pred_len=pred_len, d_feature=d_feature)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f'dlinear_{data_name}_pred{pred_len}.pkl')

    # Dummy scheduler for EarlyStopping
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
    early_stop = EarlyStopping(path=ckpt_path, optimizer=optimizer,
                                scheduler=scheduler, patience=patience)

    # Training loop
    start_time = time.time()
    best_epoch = 1

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = []
        for bx, by, bxm, bym in trainloader:
            optimizer.zero_grad()
            pred = model(bx.float())
            loss = criterion(pred, by[:, -pred_len:, :].float())
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(bx.float()),
                                    by[:, -pred_len:, :].float()).item()
                          for bx, by, bxm, bym in validloader]
        val_loss = np.mean(val_losses)

        print(f'Epoch {ep:3d}  train_loss={np.mean(train_loss):.4f}  val_loss={val_loss:.4f}')

        early_stop(val_loss, model, ep)
        if early_stop.early_stop:
            print(f'Early stopping triggered at epoch {ep}')
            best_epoch = ep - patience
            break
        best_epoch = ep

    training_time = time.time() - start_time

    # Load best checkpoint and evaluate
    print(f"\nLoading best checkpoint from epoch {best_epoch}...")
    saved = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(saved['model'])
    model.eval()

    preds, trues = [], []
    inference_start = time.time()
    with torch.no_grad():
        for bx, by, bxm, bym in testloader:
            preds.append(model(bx.float()).numpy())
            trues.append(by[:, -pred_len:, :].numpy())
    inference_time = time.time() - inference_start

    p = np.concatenate(preds)
    t = np.concatenate(trues)

    # Compute metrics
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(mse))

    print(f"\n{'='*70}")
    print(f"RESULTS — {data_name} pred_len={pred_len}h")
    print(f"{'='*70}")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Training time: {training_time:.2f}s")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*70}\n")

    return {
        'dataset': data_name,
        'model': 'DLinear',
        'pred_len': pred_len,
        'retrained': True,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'training_time': training_time,
        'inference_time': inference_time,
        'num_epochs': best_epoch,
        'checkpoint': ckpt_path
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DLinear for specific prediction horizon')
    parser.add_argument('--data_name', required=True, choices=['ETTh1', 'IndiaAQI'],
                        help='Dataset name')
    parser.add_argument('--pred_len', required=True, type=int,
                        help='Prediction horizon (48, 72, 96, 192, 336)')
    parser.add_argument('--seq_len', default=96, type=int,
                        help='Input sequence length (default: 96)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Max epochs (default: 50)')
    parser.add_argument('--patience', default=5, type=int,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--output_dir', default='./baselines',
                        help='Output directory for checkpoints')

    args = parser.parse_args()

    # Dataset configurations
    configs = {
        'ETTh1': {
            'd_feature': 7,
            'mark_path': './datasets/ETT-small/china.csv',
            'data_path': './datasets/ETT-small/ETTh1.csv'
        },
        'IndiaAQI': {
            'd_feature': 6,
            'mark_path': './datasets/india_aqi/delhi_mark.csv',
            'data_path': './datasets/india_aqi/delhi_aqi.csv'
        }
    }

    config = configs[args.data_name]

    result = train_and_eval(
        data_name=args.data_name,
        d_feature=config['d_feature'],
        mark_path=config['mark_path'],
        data_path=config['data_path'],
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # Save result to JSON
    os.makedirs('experiments_flexible', exist_ok=True)
    result_file = f'experiments_flexible/dlinear_{args.data_name}_pred{args.pred_len}_result.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to: {result_file}")
