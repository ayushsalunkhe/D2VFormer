import torch
import torch.nn as nn

class PersistenceBaseline(nn.Module):
    """
    Non-learning Last-Value / Persistence baseline.
    For every future time step in horizon O:
    prediction[t] = last observed value x[L-1].
    """
    def __init__(self):
        super(PersistenceBaseline, self).__init__()
        
    def forward(self, x_enc, pred_len: int, *args, **kwargs):
        # x_enc: [B, L, D]
        # Repeats last observed timestep across horizon pred_len
        last_val = x_enc[:, -1:, :] # [B, 1, D]
        return last_val.repeat(1, pred_len, 1) # [B, pred_len, D]
