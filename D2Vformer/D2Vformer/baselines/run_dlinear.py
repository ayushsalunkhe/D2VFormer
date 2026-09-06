# -*- coding: utf-8 -*-
"""
DLinear baseline (Zeng et al., AAAI 2023 — "Are Transformers Effective for Time Series Forecasting?")
One individual linear model per channel, with optional decomposition.
This is the honest baseline that any new forecasting model must beat.

Usage (from D2Vformer/D2Vformer):
    python baselines/run_dlinear.py --data_name ETTh1 --pred_len 96
    python baselines/run_dlinear.py --data_name IndiaAQI --pred_len 96
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        # x: (B, L, D) — pad both ends to keep length
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
    DLinear: decompose into trend + seasonal, then a separate Linear per component.
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
        # x: (B, L, D)
        seasonal, trend = self.decomp(x)
        if self.individual:
            out_t = torch.stack([self.linear_trend[i](trend[:, :, i])
                                 for i in range(trend.size(-1))], dim=-1)
            out_s = torch.stack([self.linear_seasonal[i](seasonal[:, :, i])
                                 for i in range(seasonal.size(-1))], dim=-1)
        else:
            out_t = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            out_s = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        return out_t + out_s   # (B, pred_len, D)


# ─── Training loop ──────────────────────────────────────────────────────────
def train_and_eval(data_name, d_feature, mark_path, data_path,
                   seq_len=96, pred_len=96, epochs=50, patience=5,
                   lr=0.001, batch_size=64):
    import argparse as ap
    dummy_args = ap.Namespace(d_mark=27, mark_index=[0,1,2,3])
    train, valid, test, mean, scale, _ = get_data(data_path, mark_path, args=dummy_args)

    trainset = MyDataset(train, seq_len=seq_len, label_len=0, pred_len=pred_len)
    validset = MyDataset(valid, seq_len=seq_len, label_len=0, pred_len=pred_len)
    testset  = MyDataset(test,  seq_len=seq_len, label_len=0, pred_len=pred_len)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  drop_last=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=True)

    model     = DLinear(seq_len=seq_len, pred_len=pred_len, d_feature=d_feature)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ckpt_path = f'./baselines/dlinear_{data_name}_pred{pred_len}.pkl'
    os.makedirs('./baselines', exist_ok=True)
    # dummy scheduler so EarlyStopping.save_checkpoint doesn't crash on None
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
    early_stop = EarlyStopping(path=ckpt_path, optimizer=optimizer,
                                scheduler=scheduler, patience=patience)

    print(f'\nTraining DLinear — {data_name}, pred={pred_len}')
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
        print(f'Ep {ep:3d}  train={np.mean(train_loss):.4f}  val={val_loss:.4f}')
        early_stop(val_loss, model, ep)
        if early_stop.early_stop:
            print('Early stopped.')
            break

    # Load best and evaluate test
    saved = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(saved['model'])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in testloader:
            preds.append(model(bx.float()).numpy())
            trues.append(by[:, -pred_len:, :].numpy())
    p = np.concatenate(preds); t = np.concatenate(trues)
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    print(f'\nDLinear {data_name} pred={pred_len}  Test MSE={mse:.4f}  MAE={mae:.4f}')
    return mse, mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='ETTh1')
    parser.add_argument('--pred_len',  default=96, type=int)
    args = parser.parse_args()

    configs = {
        'ETTh1': (7, './datasets/ETT-small/china.csv', './datasets/ETT-small/ETTh1.csv'),
        'IndiaAQI': (6, './datasets/india_aqi/delhi_mark.csv', './datasets/india_aqi/delhi_aqi.csv'),
    }
    if args.data_name not in configs:
        raise ValueError(f'Unknown dataset: {args.data_name}. Choose from {list(configs)}')

    d_feat, mark, data = configs[args.data_name]
    train_and_eval(args.data_name, d_feat, mark, data, pred_len=args.pred_len)
