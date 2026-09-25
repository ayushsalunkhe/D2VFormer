# TC-D2Vformer Architecture

> **Document type:** Technical architecture reference  
> **Project:** BE Computer Engineering Major Project 2025–26

---

## 1. Base Architecture: PureD2Vformer

`PureD2Vformer` is our clean reimplementation of D2Vformer (Wang et al., arXiv:2409.11024). It strictly preserves horizon independence: **all parameter tensors are functions of `(d_model, k_freq, C_in)` only, never of the output horizon `O`.**

### 1.1 Input Representation

| Symbol | Shape | Description |
|--------|-------|-------------|
| `X` | `[B, L, C_in]` | Historical input (lookback window) |
| `x_date` | `[B, L, k_freq+1]` | Date2Vec embedding of historical timestamps |
| `y_date` | `[B, O, k_freq+1]` | Date2Vec embedding of forecast timestamps |

- **B** = batch size, **L** = lookback window (96), **C_in** = number of channels (7), **O** = forecast horizon (variable)
- `k_freq` = 16 frequency components in Date2Vec

### 1.2 Date2Vec Embedding (Temporal Keys/Queries)

Date2Vec encodes time features (year, month, day, hour, ...) into a fixed-dimensional embedding:

```
D_x = [t, sin(W·t + b)]   ∈ ℝ^(k_freq+1)
```

where `W ∈ ℝ^(k_freq)` and `b ∈ ℝ^(k_freq)` are **learned parameters** shared across all positions and horizons.

### 1.3 Cross-Temporal Attention

The core mechanism computes similarity between forecast query timestamps and historical key timestamps:

```
S_xy = D_y · D_x^T / sqrt(k_freq + 1)          # [B, O, L]
A    = Softmax(S_xy, dim=-1)                      # [B, O, L]  — row-stochastic
Y_t  = A · X                                      # [B, O, C_in]
```

**Key property:** `A[b, o, :]` is a probability distribution over the L historical positions, representing how much each historical timestep contributes to forecasting output step `o`.

### 1.4 Feed-Forward Layer

```
Y = LayerNorm(Y_t + FFN(Y_t))
FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2
W_1 ∈ ℝ^(C_in × d_ff), W_2 ∈ ℝ^(d_ff × C_in)
```

### 1.5 Horizon Independence Proof

| Parameter tensor | Shape | Depends on O? |
|-----------------|-------|---------------|
| Date2Vec W | `(k_freq, 1)` | ❌ No |
| Date2Vec b | `(k_freq,)` | ❌ No |
| FFN W_1 | `(C_in, d_ff)` | ❌ No |
| FFN W_2 | `(d_ff, C_in)` | ❌ No |
| LayerNorm γ, β | `(C_in,)` | ❌ No |
| **Temperature τ** | `()` | ❌ No |

**Total parameters: 44,021** (C_in=7, d_model=128, d_ff=256, k_freq=16)  
Verified by SHA-256 checksums: `params(O=24) = params(O=720) = 44,021`

---

## 2. TC-D2Vformer: Temperature-Controlled Extension

`TCD2Vformer` extends `PureD2Vformer` with a single scalar temperature parameter τ:

```
A = Softmax( S_xy / τ,  dim=-1 )
```

### 2.1 Effect of Temperature

| τ | Effect on A | H_norm | N_eff |
|---|------------|--------|-------|
| 0.5 | Sharpens attention (amplifies differences) | ↓ decreases | ↓ decreases |
| 1.0 | Original D2Vformer | baseline | baseline |
| 2.0 | Softens attention (reduces differences) | ↑ increases | ↑ increases |
| 4.0 | Near-uniform distribution | ↑↑ increases | ↑↑ near L |

### 2.2 Implementation

```python
class TCD2Vformer(PureD2Vformer):
    def __init__(self, *args, tau=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau  # Fixed scalar — NOT a learned parameter

    def forward(self, x, x_date, y_date):
        D_x = self.date2vec(x_date)   # [B, L, k+1]
        D_y = self.date2vec(y_date)   # [B, O, k+1]
        S   = torch.bmm(D_y, D_x.transpose(1, 2)) / math.sqrt(self.k_freq + 1)
        A   = F.softmax(S / self.tau, dim=-1)   # temperature scaling
        Y_t = torch.bmm(A, x)
        return self.norm(Y_t + self.ffn(Y_t))
```

### 2.3 Parameter Count (Fixed τ mode)

```
Params(TC-D2Vformer, fixed τ) = Params(PureD2Vformer) = 44,021
```

τ is a Python float, not a `nn.Parameter` — it contributes 0 learnable parameters.

---

## 3. Hyperparameter Configuration

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| `C_in` | 7 | ETT datasets: 7 channels |
| `d_model` | 128 | Hidden dimension |
| `d_ff` | 256 | Feed-forward intermediate dim |
| `k_freq` | 16 | Date2Vec frequency components |
| `dropout` | 0.05 | Applied in FFN |
| `L` (lookback) | 96 | Fixed across all experiments |
| `O_train` | 48 | Training output horizon |
| `τ candidates` | {0.5, 1.0, 2.0, 4.0} | Evaluated via validation |
| `lr` | 1e-3 | Adam optimizer |
| `epochs` | 10 | With patience=3 early stopping |
| `batch_size` | 64 | |
| `seeds` | {42, 43, 44} | For variance estimation |

---

## 4. Key Files

| File | Description |
|------|-------------|
| [`models/pure_d2vformer.py`](../models/pure_d2vformer.py) | Clean PureD2Vformer implementation |
| [`models/temperature_d2vformer.py`](../models/temperature_d2vformer.py) | TCD2Vformer with temperature scaling |
| [`models/__init__.py`](../models/__init__.py) | Exports `PureD2Vformer`, `TCD2Vformer` |
| [`tests/test_tcd2vformer.py`](../tests/test_tcd2vformer.py) | Horizon independence unit tests |
