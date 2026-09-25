# TC-D2Vformer Methodology

> **Document type:** Research methodology  
> **Project:** BE Computer Engineering Major Project 2025–26

---

## 1. Research Question

> *Can a single, fixed scalar temperature τ applied to D2Vformer's cross-temporal attention logits improve zero-shot multi-horizon forecasting performance, when τ is selected using only the validation set?*

---

## 2. Background and Motivation

D2Vformer claims horizon-independent forecasting via Date2Vec position embeddings. Our Phase 1 audit revealed:
1. D2Vformer already achieves arbitrary-length forecasting — this is not our contribution.
2. The official repository has a data-split irregularity affecting reproducibility.
3. The cross-temporal attention mechanism has not been diagnostically studied.

**Gap:** The distributional properties of D2Vformer's attention (entropy, selectivity) and their implications for forecasting quality have not been reported.

---

## 3. Research Phases

### Phase 1: Reproduction and Audit
- Reimplemented D2Vformer as `PureD2Vformer` with verified horizon independence.
- SHA-256 parameter checksums confirm parameter count is invariant to `O`.
- Established correct chronological train/val/test splits (no lookahead).

### Phase 2: Attention Diagnostics
- **Diagnostic 1:** Extracted attention matrices from trained checkpoints.
- **Diagnostic 3:** Computed `N_eff = exp(H(A))` — effective number of attended positions.
- **Diagnostic 4:** Measured `Var(A)`, `max(A)`, `std(A)` across horizons.
- **Diagnostic 5:** Pearson/Spearman correlation between `H_norm` and MSE.
- **Diagnostic 6:** Uniform attention ablation (`A ← 1/L · 𝟙`) — no retraining.
- **Diagnostic 7:** Shuffled position ablation (permute `L` dimension of `A`) — no retraining.

**Finding:** H_norm ≈ 0.971 on ETTh1 (near-uniform); ≈ 0.922 on Exchange (more selective).

### Phase 3: Temperature Ablation
- Introduced `TCD2Vformer` with temperature scaling: `A = Softmax(S/τ)`.
- Evaluated τ ∈ {0.5, 1.0, 2.0, 4.0} across 2 datasets × 3 seeds × 6 horizons.
- Reported per-τ validation and test MSE.

### Phase 4: Validation-Selected Evaluation
- Applied the zero-lookahead protocol: τ selected via `min(val_MSE)` at O=48 only.
- Extended to 4 datasets (ETTh1, Exchange, ETTh2, ETTm1) × 3 seeds × 6 horizons.
- Produced locked test-set evaluation.

---

## 4. Validation-Selection Protocol (Zero-Lookahead)

```
For each (dataset, seed):
    For each τ ∈ {0.5, 1.0, 2.0, 4.0}:
        1. Train TCD2Vformer at O_train=48 with fixed τ
        2. Evaluate val_MSE(τ) at O=48

    τ* = argmin_τ  val_MSE(τ)   # ← NO test set used here

    For each O ∈ {24, 48, 96, 192, 336, 720}:
        Evaluate TCD2Vformer(τ=τ*) on locked test set
        Record SHA-256(params) before and after inference
```

**Critical constraint:** The test set is never observed during τ selection. This is a zero-lookahead protocol analogous to hyperparameter selection on a held-out validation split.

---

## 5. Data Splits

All splits are strictly chronological (no shuffling):

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| ETTh1 | First 60% | Next 20% | Last 20% |
| ETTh2 | First 60% | Next 20% | Last 20% |
| ETTm1 | First 60% | Next 20% | Last 20% |
| Exchange | First 60% | Next 20% | Last 20% |

**Note:** ETTh2 and ETTm1 are "unseen" datasets — τ candidates were not originally tuned on them. This provides a test of external generalizability.

---

## 6. Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | `mean((ŷ - y)²)` | Primary metric |
| MAE | `mean(|ŷ - y|)` | Secondary metric |
| Relative improvement | `(MSE_base - MSE_tc) / MSE_base × 100%` | Positive = TC-D2V better |
| H_norm | `-Σ A log A / log(L)` | Normalized attention entropy |
| N_eff | `exp(H(A))` | Effective historical positions |

---

## 7. Claim Scope

**What we claim:**
- Validation-selected temperature control reduces mean test MSE on ETTh1, Exchange, ETTh2, and ETTm1 compared to the τ=1.0 baseline.
- Long-horizon improvements are generally larger than short-horizon improvements.
- Optimal τ is dataset-dependent — τ=4.0 is not universally optimal.

**What we do NOT claim:**
- Statistical significance (n=3 seeds is underpowered for formal testing).
- Universally optimal temperature.
- Improvement over DLinear (a stronger retrained baseline).
- We invented flexible/arbitrary-length forecasting (D2Vformer already provides this).
