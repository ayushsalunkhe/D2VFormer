import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from layers.Revin import RevIN

class TemperaturePureD2Vformer(nn.Module):
    """
    TC-D2Vformer (Temperature-Controlled D2Vformer).
    Extends the mathematically verified PureD2Vformer (arXiv:2409.11024v1, Section 3, Eq. 1-10)
    with parameter-free / horizon-independent temperature scaling of cross-temporal attention:
    
        S = (D_y @ D_x^T) / sqrt(k_freq + 1)
        A = Softmax(S / tau, dim=-1)
        Y_t = A @ T^T
        
    Strict Horizon Independence Guaranteed:
        - Exactly 44,021 trainable parameters for fixed tau, completely independent of O.
        - Supports mode='fixed' (constant tau) or mode='learnable' (scalar tau = softplus(tau_raw) + eps).
    """
    def __init__(
        self,
        c_in: int,
        seq_len: int = 96,
        d_model: int = 128,
        d_ff: int = 256,
        k_freq: int = 16,
        dropout: float = 0.05,
        temperature_mode: str = 'fixed', # 'fixed' or 'learnable'
        initial_temperature: float = 1.0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.d_model = d_model
        self.k_freq = k_freq
        self.temperature_mode = temperature_mode
        
        # 1. RevIN
        self.revin = RevIN(c_in, affine=True, subtract_last=False)
        
        # 2. TFE (Eq. 3)
        self.tfe = nn.Linear(c_in, d_model)
        
        # 3. Date2Vec parameters (Eq. 4-7)
        self.w_T = nn.Parameter(torch.randn(seq_len))
        self.b_T = nn.Parameter(torch.zeros(d_model))
        
        self.W_S = nn.Parameter(torch.randn(k_freq, seq_len))
        self.B_S = nn.Parameter(torch.zeros(k_freq, d_model))
        
        self.b_1 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_2 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        self.b_3 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_4 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        
        # 4. Position-wise FFN (Eq. 10)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, c_in)
        )
        
        # 5. Temperature parameterization
        if temperature_mode == 'fixed':
            self.register_buffer('tau_fixed', torch.tensor(float(initial_temperature)))
        elif temperature_mode == 'learnable':
            # Inverse softplus: softplus(x) = ln(1 + exp(x)) -> x = ln(exp(tau) - 1)
            init_raw = math.log(math.exp(initial_temperature) - 1.0) if initial_temperature > 1e-4 else -5.0
            self.tau_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown temperature_mode: {temperature_mode}. Must be 'fixed' or 'learnable'.")

    def get_temperature(self) -> torch.Tensor:
        """Returns scalar temperature tau in (0, inf)."""
        if self.temperature_mode == 'fixed':
            return self.tau_fixed
        else:
            # Guarantee tau > 1e-4 strictly
            return F.softplus(self.tau_raw) + 1e-4

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, y_mark_dec: torch.Tensor):
        """
        Args:
            x_enc: Historical data [B, L, D]
            x_mark_enc: Historical timestamps [B, L, M]
            y_mark_dec: Future timestamps [B, O, M]
        Returns:
            Y_out: [B, O, D]
            A: [B, H, O, L]
        """
        B, L, D = x_enc.shape
        O = y_mark_dec.shape[1]
        
        # 1. RevIN norm
        x_norm = self.revin(x_enc, 'norm') # [B, L, D]
        
        # 2. TFE
        T = self.tfe(x_norm) # [B, L, H]
        
        # 3. Date2Vec linear & harmonic projection
        v_T = torch.einsum('l,blh->bh', self.w_T, T) + self.b_T # [B, H]
        Omega_S = torch.einsum('kl,blh->bkh', self.W_S, T) + self.B_S # [B, k, H]
        
        # Historical Time-Position Embedding Dx
        E_lin = (v_T.unsqueeze(2).unsqueeze(3) * x_mark_enc.unsqueeze(1) + self.b_1).unsqueeze(1) # [B, 1, H, L, 4]
        E_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * x_mark_enc.unsqueeze(1).unsqueeze(2) + self.B_2) # [B, k, H, L, 4]
        D_x = torch.cat([E_lin, E_har], dim=1).mean(dim=-1) # [B, k+1, H, L]
        
        # Future Time-Position Embedding Dy
        F_lin = (v_T.unsqueeze(2).unsqueeze(3) * y_mark_dec.unsqueeze(1) + self.b_3).unsqueeze(1) # [B, 1, H, O, 4]
        F_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * y_mark_dec.unsqueeze(1).unsqueeze(2) + self.B_4) # [B, k, H, O, 4]
        D_y = torch.cat([F_lin, F_har], dim=1).mean(dim=-1) # [B, k+1, H, O]
        
        # Align for inner product: Dx [B, H, L, k+1], Dy [B, H, O, k+1]
        Dx = D_x.permute(0, 2, 3, 1)
        Dy = D_y.permute(0, 2, 3, 1)
        
        # Cross-temporal similarity matrix S: [B, H, O, L]
        S = torch.einsum('bhok,bhlk->bhol', Dy, Dx) / math.sqrt(self.k_freq + 1)
        
        # Temperature-scaled Attention
        tau = self.get_temperature()
        A = torch.softmax(S / tau, dim=-1) # [B, H, O, L]
        
        # 4. Temporal Value Aggregation: Yt = A * T^T
        Yt = torch.einsum('bhol,bhl->bho', A, T.transpose(1, 2)) # [B, H, O]
        
        # 5. Position-wise FFN and RevIN denorm
        out = self.ffn(Yt.transpose(1, 2)) # [B, O, D]
        Y_out = self.revin(out, 'denorm') # [B, O, D]
        
        return Y_out, A
        
# Canonical naming alias for Phase 5
TCD2Vformer = TemperaturePureD2Vformer
