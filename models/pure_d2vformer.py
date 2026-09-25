import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layers.Revin import RevIN

class PureD2Vformer(nn.Module):
    """
    Exact mathematical implementation of D2Vformer as specified in arXiv:2409.11024v1 (Section 3, Eq. 1-10).
    Completely horizon-independent: zero trainable parameters depend on output horizon O.
    """
    def __init__(self, c_in: int, seq_len: int = 96, d_model: int = 128, d_ff: int = 256, k_freq: int = 16, dropout: float = 0.05):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.d_model = d_model
        self.k_freq = k_freq
        
        # 1. Reversible Instance Normalization (RevIN)
        self.revin = RevIN(c_in, affine=True, subtract_last=False)
        
        # 2. Temporal Feature Extraction (TFE) - Eq. 3
        # Projects features D -> H across time steps L
        self.tfe = nn.Linear(c_in, d_model)
        
        # 3. Date2Vec (D2V) - Eq. 4-7
        # Linear component parameters: w_T in R^{1 x L}, b_T in R^H
        self.w_T = nn.Parameter(torch.randn(seq_len))
        self.b_T = nn.Parameter(torch.zeros(d_model))
        
        # Harmonic components parameters: W_S in R^{k x L}, B_S in R^{k x H}
        self.W_S = nn.Parameter(torch.randn(k_freq, seq_len))
        self.B_S = nn.Parameter(torch.zeros(k_freq, d_model))
        
        # Biases for Kronecker products: b_1, b_3 in R^H; B_2, B_4 in R^{k x H}
        self.b_1 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_2 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        self.b_3 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_4 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        
        # 4. Fusion Block FeedForward - Eq. 10
        # Position-wise FeedForward mapping H -> d_ff -> D (independent of O)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, c_in)
        )
        
    def forward(self, x_enc, x_mark_enc, y_mark_dec):
        """
        Args:
            x_enc: Historical feature sequence [B, L, D]
            x_mark_enc: Historical timestamp markers [B, L, M]
            y_mark_dec: Future target timestamp markers [B, O, M]
        Returns:
            Y_out: Forecast sequence [B, O, D]
            A: Cross-temporal attention weights [B, H, O, L]
        """
        B, L, D = x_enc.shape
        O = y_mark_dec.shape[1]
        
        # Step 1: RevIN normalization [B, L, D]
        x_norm = self.revin(x_enc, 'norm')
        
        # Step 2: Temporal Feature Extraction [B, L, H]
        T = self.tfe(x_norm)
        
        # Step 3: Date2Vec latent frequency and linear components (Eq. 4)
        # v_T: [B, H]
        v_T = torch.einsum('l, blh -> bh', self.w_T, T) + self.b_T
        # Omega_S: [B, k, H]
        Omega_S = torch.einsum('kl, blh -> bkh', self.W_S, T) + self.B_S
        
        # Step 4: Time-Position Embeddings (Eq. 5-7)
        # Input embedding E: [B, k+1, H, L, M] -> D_x_hat: [B, k+1, H, L]
        E_lin = (v_T.unsqueeze(2).unsqueeze(3) * x_mark_enc.unsqueeze(1) + self.b_1).unsqueeze(1)
        E_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * x_mark_enc.unsqueeze(1).unsqueeze(2) + self.B_2)
        E = torch.cat([E_lin, E_har], dim=1)
        D_x_hat = E.mean(dim=-1) # Normalized aggregation over M timestamp dimensions
        
        # Output embedding F: [B, k+1, H, O, M] -> D_y_hat: [B, k+1, H, O]
        F_lin = (v_T.unsqueeze(2).unsqueeze(3) * y_mark_dec.unsqueeze(1) + self.b_3).unsqueeze(1)
        F_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * y_mark_dec.unsqueeze(1).unsqueeze(2) + self.B_4)
        F = torch.cat([F_lin, F_har], dim=1)
        D_y_hat = F.mean(dim=-1)
        
        # Step 5: Fusion Block Cross-Temporal Attention (Eq. 8)
        # Permute to [B, H, L, k+1] and [B, H, O, k+1]
        D_x_tilde = D_x_hat.permute(0, 2, 3, 1)
        D_y_tilde = D_y_hat.permute(0, 2, 3, 1)
        
        scale = 1.0 / math.sqrt(self.k_freq + 1)
        # scores: [B, H, O, L] (Query: future target O, Key: past history L)
        scores = torch.einsum('bhok, bhlk -> bhol', D_y_tilde, D_x_tilde) * scale
        A = torch.softmax(scores, dim=-1) # Normalized across past history L
        
        # Step 6: Temporal Value Aggregation (Eq. 9)
        # T_transposed: [B, H, L] -> Y_tilde: [B, H, O]
        T_transposed = T.transpose(1, 2)
        Y_tilde = torch.einsum('bhol, bhl -> bho', A, T_transposed)
        
        # Step 7: FeedForward Projection H -> D (Eq. 10)
        # Operates along the H dimension independently at each target step o in O
        Y_hat = self.ffn(Y_tilde.transpose(1, 2)) # [B, O, D]
        
        # Step 8: RevIN Denormalization [B, O, D]
        Y_out = self.revin(Y_hat, 'denorm')
        return Y_out, A
