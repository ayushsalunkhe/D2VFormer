# TCD2Vformer: Architectural and Experimental Design (Phase 6)

## 1. Motivation & Context
In Phases 1–5, the D2Vformer research project systematically investigated the mathematical properties and empirical behavior of `PureD2Vformer` (arXiv:2409.11024v1):
1. **Parameter Independence:** A mathematically faithful reconstruction achieved true horizon-independent zero-shot forecasting with exactly 44,021 parameters, whereas the official repository included horizon-dependent linear projections.
2. **Attention Pathology:** Diagnostics across four benchmark datasets revealed severe cross-temporal attention diffuseness ($H_{norm} \approx 0.92 - 0.98$).
3. **Discrete Temperature Control:** Softening or sharpening cross-temporal attention via $A = \text{Softmax}(S/\tau)$ proved that optimal temperature is dataset-dependent:
   - Low-frequency / high-noise datasets (Exchange daily, ETTh1/ETTh2 hourly) favored higher temperatures ($\tau = 2.0$ or $4.0$) to regularize attention over past history.
   - High-frequency datasets (ETTm1 15-minute) favored sharper attention ($\tau = 0.5$) or baseline $\tau = 1.0$, rejecting the universal $\tau = 4$ hypothesis.

However, selecting $\tau$ from a discrete candidate grid $\{0.5, 1.0, 2.0, 4.0\}$ is an empirical hyperparameter search. Phase 6 investigates whether this attention pathology can be addressed via an **intrinsic architectural contribution**:

$$\tau = f(D_x, D_y)$$

where the attention temperature is dynamically conditioned on the Date2Vec temporal representations, while strictly preserving zero-shot horizon generalization and horizon-independent parameter count.

---

## 2. Mathematical Formulation

### 2.1 PureD2Vformer Attention Foundation
Let:
- $x_{enc} \in \mathbb{R}^{B \times L \times D}$ be historical observations.
- $x_{mark\_enc} \in \mathbb{R}^{B \times L \times M}$ be historical calendar markers ($M=4$).
- $y_{mark\_dec} \in \mathbb{R}^{B \times O \times M}$ be future calendar markers ($M=4$).
- $T = \text{Linear}(x_{norm}) \in \mathbb{R}^{B \times L \times H}$ be temporal feature representations.

Date2Vec embeddings are computed via linear and harmonic projections:
- Past embedding: $D_x \in \mathbb{R}^{B \times H \times L \times (k+1)}$
- Future embedding: $D_y \in \mathbb{R}^{B \times H \times O \times (k+1)}$

The scaled similarity matrix is:
$$S_{b, h, o, l} = \frac{1}{\sqrt{k_{freq} + 1}} \sum_{k=1}^{k_{freq}+1} D_y[b, h, o, k] \cdot D_x[b, h, l, k]$$
where $S \in \mathbb{R}^{B \times H \times O \times L}$.

### 2.2 Temperature Mechanisms Formulated for TCD2Vformer

#### Formulation A: Baseline Fixed Temperature
$$\tau = 1.0 \quad \text{or} \quad \tau = \tau^* \in \{0.5, 1.0, 2.0, 4.0\}$$
$$A_{b, h, o, l} = \frac{\exp(S_{b, h, o, l} / \tau)}{\sum_{j=1}^L \exp(S_{b, h, o, j} / \tau)}$$
- Added parameters: $0$.

#### Formulation C: Global Learnable Temperature (`learned_global`)
$$\tau = \text{softplus}(\tau_{raw}) + \epsilon, \quad \epsilon = 10^{-4}$$
where $\tau_{raw} \in \mathbb{R}$ is an unconstrained trainable scalar parameter.
- Optimization: Backpropagated jointly with model parameters using gradient descent:
  $$\frac{\partial \mathcal{L}}{\partial \tau_{raw}} = \frac{\partial \mathcal{L}}{\partial A} \frac{\partial A}{\partial \tau} \sigma(\tau_{raw})$$
- Added parameters: Exactly 1 parameter ($44,022$ total).
- Horizon scaling: Constant with respect to $O$.

#### Formulation D: Temporal-Context Conditioned Temperature (`temporal_context`)
The temporal structure of the historical window $D_x$ carries information about frequency, cycle phase, and seasonality. We aggregate $D_x$ over lookback $L$ and channel dimension $H$:
$$\bar{d}_x = \frac{1}{H \cdot L} \sum_{h=1}^H \sum_{l=1}^L D_x[:, h, l, :] \in \mathbb{R}^{B \times (k+1)}$$
A lightweight non-linear mapping predicts an instance-level temperature:
$$g_{ctx}(\bar{d}_x) = W_2 \cdot \text{GELU}(W_1 \bar{d}_x + b_1) + b_2$$
$$\tau(x) = \tau_{min} + \text{softplus}(g_{ctx}(\bar{d}_x))$$
where:
- $W_1 \in \mathbb{R}^{d_{temp} \times (k+1)}$, $b_1 \in \mathbb{R}^{d_{temp}}$
- $W_2 \in \mathbb{R}^{1 \times d_{temp}}$, $b_2 \in \mathbb{R}^{1}$
- With $k_{freq} = 16$ ($k+1 = 17$), $d_{temp} = 16$, $\tau_{min} = 0.1$:
  - Added parameters: $17 \times 16 + 16 + 16 \times 1 + 1 = 305$ parameters ($44,326$ total).
- Horizon scaling: Constant with respect to $O$.

#### Formulation E: Query-Conditioned Dynamic Temperature (`query_conditioned`)
Different future prediction horizons $o \in \{1, \dots, O\}$ may require varying attention sharpness. We condition temperature on the future Date2Vec query representations:
$$\bar{d}_{y, o} = \frac{1}{H} \sum_{h=1}^H D_y[:, h, o, :] \in \mathbb{R}^{B \times (k+1)}$$
$$g_{qry}(\bar{d}_{y, o}) = W_2 \cdot \text{GELU}(W_1 \bar{d}_{y, o} + b_1) + b_2$$
$$\tau_{b, o} = \tau_{min} + \text{softplus}(g_{qry}(\bar{d}_{y, o})) \in \mathbb{R}^{B \times 1 \times O \times 1}$$
The attention matrix is normalized along $L$:
$$A_{b, h, o, l} = \frac{\exp(S_{b, h, o, l} / \tau_{b, o})}{\sum_{j=1}^L \exp(S_{b, h, o, j} / \tau_{b, o})}$$
- Dimensionality: $\tau \in \mathbb{R}^{B \times 1 \times O \times 1}$, broadcasts over $H$ and $L$.
- Added parameters: Exactly 305 parameters ($44,326$ total).
- Horizon scaling: Constant with respect to $O$ (weights $W_1, W_2, b_1, b_2$ are applied step-wise and do not depend on $O$).

---

## 3. Architecture Diagram Description

```
                     x_enc [B, L, D]
                           │
                         RevIN
                           │
                     x_norm [B, L, D]
                           │
                    TFE Linear(D, H)
                           │
                  T [B, L, H] (Values)
                 /         │          \
                /          │           \
               ▼           ▼            ▼
          (v_T, Omega_S)  (Date2Vec)   (Date2Vec)
                 │         │            │
       x_mark ───┼─────────┘            │
                 ▼                      ▼
           Dx [B, H, L, k+1]      Dy [B, H, O, k+1]
                 │                      │
                 ├──────────────────────┼───────────────────────────┐
                 │                      │                           │
                 ▼                      ▼                           ▼
          Scaled Dot-Product      Query Mean d_y,o           Context Mean d_x
          S = Dy @ Dx^T / sqrt(k+1)    [B, O, k+1]              [B, k+1]
                 │                      │                           │
                 │                MLP_qry (Shared)            MLP_ctx
                 │                      │                           │
                 │                 Dynamic tau_o               Static tau
                 │              [B, 1, O, 1]                    [B, 1, 1, 1]
                 │                      │                           │
                 └──────────────┬───────┴───────────────────────────┘
                                ▼
                       A = Softmax(S / tau, dim=L)
                                │
                                ▼
                    Temporal Value Aggregation
                         Y_tilde = A @ T^T
                            [B, H, O]
                                │
                    FeedForward (H -> d_ff -> D)
                                │
                           RevIN Denorm
                                │
                         Y_out [B, O, D]
```

---

## 4. Parameter-Scaling Analysis

| Component | PureD2Vformer | TCD2Vformer (Fixed) | TCD2Vformer (Learned Global) | TCD2Vformer (Temporal Context) | TCD2Vformer (Query Conditioned) | Dependency on Horizon $O$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RevIN (`affine=True`) | $2 \times D$ | $2 \times D$ | $2 \times D$ | $2 \times D$ | $2 \times D$ | None ($O$-independent) |
| TFE (`c_in -> d_model`) | $D \times H + H$ | $D \times H + H$ | $D \times H + H$ | $D \times H + H$ | $D \times H + H$ | None ($O$-independent) |
| Date2Vec Projections | $L + H + kL + kH + 2(H + kH)$ | Same | Same | Same | Same | None ($O$-independent) |
| Position FFN | $H \times d_{ff} + d_{ff} + d_{ff} \times D + D$ | Same | Same | Same | Same | None ($O$-independent) |
| Temperature Mechanism | 0 | 0 | 1 | 305 | 305 | None ($O$-independent) |
| **Total Parameters (D=7, H=128)** | **44,021** | **44,021** | **44,022** | **44,326** | **44,326** | **$\mathbf{0 \times O}$ (Strictly Invariant)** |

**Automated Proof Requirement:**
For all evaluation horizons $O \in \{24, 48, 96, 192, 336, 720\}$, the parameter count must be:
$$\frac{\partial N_{params}}{\partial O} \equiv 0$$

---

## 5. Computational Complexity
- **Dot-product attention:** $O(B \cdot H \cdot O \cdot L \cdot (k+1))$
- **Query-conditioned temperature:** $O(B \cdot O \cdot (k+1) \cdot d_{temp})$
  Since $d_{temp}=16 \ll L=96$, the temperature MLP adds $< 1.5\%$ additional FLOPs relative to cross-temporal attention.
- **Memory footprint:** Storing $\tau_o \in \mathbb{R}^{B \times 1 \times O \times 1}$ adds negligible memory overhead ($O \times 4$ bytes per sequence).

---

## 6. Expected Risks & Scientific Failure Modes
1. **Gradient Starvation:** If the attention distribution $A$ is diffuse, gradients with respect to $\tau$ may be small ($\frac{\partial \mathcal{L}}{\partial \tau} \propto \text{Var}_A[S]$). When $S$ is nearly uniform, temperature gradients approach zero.
2. **Over-Smoothing:** If the query Date2Vec embeddings $D_y$ lack distinct high-frequency information, $g_{qry}(D_y)$ could collapse to a constant bias, degenerating into Formulation C (learned global scalar).
3. **Generalization Gap at Long Horizons:** If $g_{qry}$ is trained only on $O_{train}=48$ future timestamps, timestamps $o \in (48, 720]$ will receive temporal embeddings outside the training horizon window. Date2Vec harmonic features should theoretically extrapolate, but empirical degradation may occur.
4. **Negative Result Mitigation:** If learned or conditioned temperatures fail to outperform validation-selected fixed temperatures, we will explicitly report the negative result and document the causal pathology.

---

## 7. Exact Experimental Protocol

### 7.1 Data Splits & Normalization
- Split: Strict 60% Train / 20% Validation / 20% Test.
- Normalization: Instance normalization via RevIN; dataset global statistics (mean, std) calculated strictly on the training partition.
- Lookback horizon: $L = 96$.
- Training horizon: $O_{train} = 48$.
- Evaluation horizons (zero-shot): $O_{eval} \in \{24, 48, 96, 192, 336, 720\}$.

### 7.2 Strict Validation-First Protocol
1. Train candidate models strictly on the training set with early stopping based on validation MSE at $O_{train} = 48$.
2. Model checkpoints saved at best validation epoch.
3. Test set is **strictly locked** during training and model selection.
4. Final zero-shot evaluation executed once on the locked test set.

### 7.3 Reproducibility & Determinism
- Seeds: 42, 43, 44.
- Deterministic flags: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- SHA-256 parameter checksums logged before and after evaluation to guarantee parameter invariance.
