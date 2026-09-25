# TCD2Vformer: Phase 6 Experimental Protocol

## 1. Scope & Objective
This protocol establishes the rules, baseline models, candidate models, evaluation metrics, and zero-lookahead validation gates for Phase 6 of the D2Vformer research project.

---

## 2. Experimental Matrix

### 2.1 Benchmark Datasets
1. **ETTh1** (Electricity Transformer Temperature - Hourly 1): $D = 7$ features, $1$-hour sampling interval.
2. **Exchange** (Exchange Rate): $D = 8$ features, $1$-day sampling interval.
3. **ETTh2** (Electricity Transformer Temperature - Hourly 2): $D = 7$ features, $1$-hour sampling interval.
4. **ETTm1** (Electricity Transformer Temperature - 15-Minute 1): $D = 7$ features, $15$-minute sampling interval.

### 2.2 Random Seeds
Strictly fixed across all models and datasets:
$$\text{Seeds} = \{42, 43, 44\}$$

### 2.3 Lookback and Horizons
- **Lookback Window:** $L = 96$ timestamps for all datasets.
- **Training Horizon:** $O_{train} = 48$ timestamps.
  - **Constraint:** Models are trained ONCE at $O_{train}=48$.
- **Evaluation Horizons:** Zero-shot multi-horizon evaluation across:
  $$O \in \{24, 48, 96, 192, 336, 720\}$$
  No retraining or fine-tuning is permitted for evaluation horizons.

---

## 3. Model Configurations & Ablation Modes

All models share the identical backbone hyperparameters:
- $L = 96$
- $d_{model} = 128$
- $d_{ff} = 256$
- $k_{freq} = 16$
- $\text{dropout} = 0.05$
- Reversible Instance Normalization (RevIN) enabled with `affine=True`.

| Model / Ablation | Code Identifier | Description | Parameter Count ($D=7$) |
| :--- | :--- | :--- | :--- |
| **Ablation A** | `pure_d2vformer_tau1` | Fixed baseline $\tau = 1.0$ (PureD2Vformer) | 44,021 |
| **Ablation B** | `fixed_val_selected` | Fixed temperature selected via validation loss from $\{0.5, 1.0, 2.0, 4.0\}$ | 44,021 |
| **Ablation C** | `learned_global` | Global learned temperature $\tau = \text{softplus}(\tau_{raw}) + 10^{-4}$ | 44,022 |
| **Ablation D** | `temporal_context` | Context-conditioned temperature $\tau(x) = \tau_{min} + \text{softplus}(g_{ctx}(\bar{d}_x))$ | 44,326 |
| **Ablation E** | `query_conditioned` | Target-conditioned dynamic temperature $\tau_o = \tau_{min} + \text{softplus}(g_{qry}(\bar{d}_{y, o}))$ | 44,326 |

---

## 4. Strict Validation & Anti-Lookahead Protocol

1. **Partitioning:**
   - 60% Train, 20% Validation, 20% Test chronologically split.
   - Normalization statistics $(\mu, \sigma)$ derived strictly from the training partition.
2. **Early Stopping & Model Checkpointing:**
   - Evaluated on the validation set at the end of each epoch ($O = 48$).
   - Patience: $3$ epochs.
   - Checkpoint saved only when $\mathcal{L}_{val} < \mathcal{L}_{val, best}$.
3. **Locked Test Partition:**
   - Test partition labels are inaccessible during training, hyperparameter tuning, and model selection.
   - Zero-shot evaluation across $O \in \{24, 48, 96, 192, 336, 720\}$ occurs strictly once per locked checkpoint.
4. **Parameter Invariance Checksum:**
   - Deterministic SHA-256 hash computed over all parameter tensors before and after evaluation to verify no in-place mutation or test-set leakage.

---

## 5. Metrics & Diagnostic Outputs

1. **Forecasting Accuracy:**
   - Mean Squared Error (MSE): $\frac{1}{N} \sum (y - \hat{y})^2$
   - Mean Absolute Error (MAE): $\frac{1}{N} \sum |y - \hat{y}|$
2. **Attention Diagnostics:**
   - Shannon Entropy: $H = -\sum_{l=1}^L A_{bhol} \ln(A_{bhol})$
   - Normalized Entropy: $H_{norm} = \frac{H}{\ln(L)}$
   - Effective Attended Timestamps: $N_{eff} = \exp(H)$
   - Maximum Attention Weight: $\max_l A_{bhol}$
   - Attention Variance: $\text{Var}_l(A_{bhol})$
   - KL Divergence from Uniform Distribution: $D_{KL}(A \parallel U)$
3. **Temperature Diagnostics:**
   - Effective Temperature: Mean, standard deviation, minimum, and maximum $\tau$ values per dataset and horizon.

---

## 6. Verification Pipeline
1. **Model Unit Tests:** Verify forward/backward gradient flow, parameter invariance, and horizon-independence.
2. **Pilot Smoke Test:** Single dataset (ETTh1), single seed (42), $O_{train}=48$, $O_{eval} \in \{48, 192, 720\}$.
3. **Full Benchmark Execution:** 4 datasets $\times$ 3 seeds $\times$ 5 ablation modes $\times$ 6 horizons.
4. **Result Verification:** Verify all SHA-256 checksums, constant parameter counts, and compile final tables.
