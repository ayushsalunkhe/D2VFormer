# Limitations and Future Work

> **Document type:** Limitations and future directions  
> **Project:** BE Computer Engineering Major Project 2025–26

---

## Known Limitations

### L1. Small Seed Count (n=3)
We use only 3 random seeds (42, 43, 44) due to computational constraints. With n=3:
- Formal statistical tests (e.g., Wilcoxon signed-rank) are severely underpowered.
- Seed-level variance is large relative to the reported improvements.
- **We report effect sizes only — we do not claim statistical significance.**

### L2. Small Improvement Magnitudes
The mean relative improvement of +0.81% over the τ=1.0 baseline is modest. For ETTh1 at O=24, the improvement is only +0.38% (0.8258 → 0.8227 MSE). While consistent, these gains may not be practically significant in all applications.

### L3. Dataset-Dependent Temperature
Optimal τ is not universal:
- ETTh1: τ ∈ {2.0, 4.0}
- Exchange: τ ∈ {0.5, 4.0}
- ETTh2: τ ∈ {1.0, 2.0, 4.0}
- ETTm1: τ ∈ {0.5, 2.0}

This means τ must be selected per dataset — it cannot be set once and deployed universally.

### L4. ETTm1 Long-Horizon Degradation
For ETTm1 at horizons O≥96, TC-D2Vformer is worse than the τ=1.0 baseline (−0.12% to −0.88%). The short-horizon gains do not generalize to longer-horizon forecasting on 15-minute data.

### L5. Not Competitive with DLinear
DLinear (a simple decomposition-based linear model retrained per horizon) significantly outperforms all D2Vformer variants on most dataset-horizon combinations. TC-D2Vformer's contribution is within the D2Vformer family.

### L6. Coarse Temperature Grid
We evaluate only 4 temperature values: {0.5, 1.0, 2.0, 4.0}. The optimal continuous τ may lie outside this range or between these values. A finer grid (e.g., {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}) may yield better results.

### L7. Fixed Training Horizon
All models are trained at O_train=48. The effect of training horizon on optimal temperature selection has not been studied. Models trained at O_train=24 or O_train=96 may prefer different τ values.

---

## Future Work

### F1. Learnable Temperature
Implement learnable τ as a single `nn.Parameter` with `τ = softplus(τ_raw) + ε`. This would allow the model to adaptively set τ during training without manual grid search.

### F2. Layer-Wise and Head-Wise Temperature
Apply different τ values per attention head or per layer. This adds minimal parameters but may capture dataset-specific structure more effectively.

### F3. Expanded Temperature Grid
Evaluate τ ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0} to better characterize the τ-MSE landscape.

### F4. Larger Seed Set
Repeat experiments with seeds ∈ {40, 41, 42, 43, 44, 45, 46, 47, 48, 49} (n=10) to enable formal statistical testing.

### F5. Additional Datasets
Evaluate on Weather, ILI, Traffic datasets to assess cross-domain generalizability.

### F6. Attention Regularization
Use H_norm as an auxiliary regularization loss during training to explicitly encourage a target entropy level, rather than scaling post-hoc.
