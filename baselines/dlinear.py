import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """Moving average block to extract trends."""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear baseline (Zeng et al., AAAI 2023).
    Trained separately for each target prediction horizon O.
    """
    def __init__(self, seq_len: int, pred_len: int, c_in: int, moving_avg_kernel: int = 25, individual: bool = False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.individual = individual
        self.decompsition = series_decomp(moving_avg_kernel)
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.c_in)])
            self.Linear_Trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.c_in)])
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, *args, **kwargs):
        # x_enc: [B, L, D]
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1) # [B, D, L]
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype, device=trend_init.device)
            for i in range(self.c_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1) # [B, O, D]
