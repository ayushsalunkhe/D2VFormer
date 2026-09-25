# Phase 4B: Pre-Registered Experimental Protocol

**Project:** D2Vformer Horizon Generalization & Robustness Study  
**Registration Date:** 2026-09-22  
**Status:** Pre-registered before test-set evaluation of unseen datasets.

---

## 1. Experimental Objectives & Hypotheses

### Primary Scientific Question
Does validation-selected temperature scaling generalize to unseen datasets (ETTh2, ETTm1) by improving zero-shot multi-horizon test performance over the default $\tau=1.0$ baseline, without adding any horizon-specific parameters?

### Pre-Registered Hypotheses
1. **$H_1$ (Generalization of Softer Attention):** On power-grid datasets with near-uniform baseline entropy (similar to ETTh1), validation loss will prefer softer attention ($\tau \in \{2.0, 4.0\}$) over sharper attention ($\tau = 0.5$).
2. **$H_2$ (Zero-Lookahead Utility):** PureD2Vformer with validation-selected $\tau$ will achieve lower or equal test MSE than default baseline $\tau=1.0$ on average across horizons.
3. **$H_3$ (Long-Horizon Extrapolation):** The relative advantage of temperature scaling will be more pronounced at extended horizons ($O \in \{336, 720\}$) due to cumulative error reduction.
4. **$H_0$ (Falsification Criteria):** If validation loss selects $\tau=1.0$ or $\tau=0.5$, or if the selected temperature degrades test MSE across the majority of horizons, the hypothesis that softer attention is broadly beneficial across transformer data is falsified.

---

## 2. Models Evaluated

1. **`PureD2Vformer` (Baseline $\tau=1.0$):** Standard mathematical D2Vformer (no temperature scaling).
2. **`PureD2Vformer` (Val-Selected $\tau$):** Temperature $\tau \in \{0.5, 1.0, 2.0, 4.0\}$ selected strictly by minimal validation loss at $O_{\text{train}}=48$.
3. **`Uniform Attention Control`:** Cross-temporal attention replaced with constant $A_{o,l} = 1/L$. Evaluates raw utility of learned temporal non-uniformity.
4. **`DLinear`:** Horizon-specific linear baseline retrained separately for each target horizon $O$.
5. **`Persistence`:** Naive baseline repeating the last observed value $\hat{y}_{t+o} = x_t$.

---

## 3. Strict Training & Selection Protocol

- **Lookback window:** $L = 96$
- **Training forecast horizon:** $O_{\text{train}} = 48$ (fixed for all PureD2Vformer variants)
- **Seeds:** 42, 43, 44 (independent random seeds)
- **Optimizer:** Adam with learning rate $\eta = 10^{-3}$
- **Batch size:** 64
- **Maximum epochs:** 10 with Early Stopping (patience = 3 based on validation MSE)
- **Temperature candidates:** $\tau \in \{0.5, 1.0, 2.0, 4.0\}$

### Hyperparameter Selection Rule (Zero Lookahead)
$$\tau^* = \arg\min_{\tau \in \{0.5, 1.0, 2.0, 4.0\}} \mathcal{L}_{\text{val}}(\tau; O=48)$$
- Test data is **completely untouched** during training and selection.
- Selected $\tau^*$ is locked into `validation_selection.csv` prior to test evaluation.

---

## 4. Test Evaluation Protocol

- **Forecast Horizons:** $O \in \{24, 48, 96, 192, 336, 720\}$ (zero-shot evaluation from a single trained model checkpoint).
- **Batch Size:** 64 (reduced to 32 for $O \ge 336$ if VRAM requires).
- **Metrics Recorded:**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Normalized Attention Entropy ($H_{\text{norm}}$)
  - Effective Attention Sample Size ($N_{\text{eff}}$)
  - Max Attention Weight ($\max A$)
  - Attention Variance ($\text{Var}(A)$)
  - KL Divergence from Uniform ($D_{\text{KL}}(A \parallel U)$)
  - SHA-256 Parameter Checksum verification
