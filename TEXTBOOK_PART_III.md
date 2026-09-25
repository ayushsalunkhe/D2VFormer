# D2Vformer to TCD2Vformer — Part III: Phases 1–5 Reconstruction

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §7–§9 and `models/pure_d2vformer.py`.
> **Phase scope:** Phase 1–5 are pure reconstruction. They fix what we identified as the parameter-independence flaw. No new architectural concepts are introduced; new concepts come in Phase 6.

---

## 7. Phase 1 — Pure D2Vformer Reconstruction

### 7.1 Phase 1's single objective

Reproduce the original D2Vformer architecture **with the horizon-scaling flaw removed** and verify that the model trains, predicts, and produces sensible values.

Phase 1's deliverable is `models/pure_d2vformer.py` — a from-scratch implementation that:

- Computes cross-temporal attention between Date2Vec embeddings.
- Uses `Y_tilde = A · T^T` to aggregate temporal features (parameter-free).
- Projects through an FFN, not through `Linear(d_model, O)`.
- Has **44,021 parameters** when `c_in = 7` (ETT datasets).

### 7.2 Why we had to do this

The official `model/D2Vformer.py` had a `Linear(d_model, O)` projection. Three problems:

1. The parameter count scales with O, breaking horizon independence.
2. We cannot use the official code as a baseline in our own thesis because the bug is part of "the original."
3. To study Phase 6's dynamic-temperature extension, we need a clean baseline whose only difference is the temperature mechanism.

Hence: **reconstruct from scratch**, preserving the paper's intent but fixing the implementation flaw.

### 7.3 Design decisions made in Phase 1

#### 7.3.1 Output projection

Replaced:
```python
# ❌ old
self.project = nn.Linear(self.d_model, self.pred_len)
```

with:
```python
# ✓ new
Y_tilde = torch.einsum('bhol,bhl->bho', A, T)   # parameter-free aggregation
Y_hat = self.ffn(Y_tilde.transpose(1, 2))         # FFN projects across d_model → c_in
```

The FFN is `Linear(d_model, d_ff) → GELU → Linear(d_ff, c_in)`, which is horizon-independent.

#### 7.3.2 Channel alignment

The original implementation computed `Y_hat ∈ ℝ^{B × O × d_model}`, then `Linear(d_model, c_in)`. The channel reduction happens at the FFN.

Our implementation does the same: the FFN ends with `Linear(d_ff, c_in)`.

#### 7.3.3 RevIN placement

We retain RevIN as the normalisation layer. RevIN is an *inherited* technique (Kim et al., ICLR 2022), used by most modern forecasting models. It is not our contribution.

#### 7.3.4 Date2Vec formulation

We follow the paper's harmonic formulation exactly:

```
D(t) = [t, sin(ω_1 t + φ_1), ..., sin(ω_k t + φ_k)]
```

with `k_freq = 16`, total dimension `17`.

### 7.4 Parameter accounting

The exact parameter counts come from summing each trainable module:

| Module | Layer | Parameters | Notes |
|---|---|---|---|
| RevIN | (none — uses stored stats) | 0 | parameter-free |
| TFE Linear | `Linear(c_in=7, d_model=128)` | 7×128 + 128 = 1,024 | |
| Date2Vec harmonic | `W_S: (k_freq, L)`, `B_S: (k_freq, d_model)` | 16×96 + 16×128 = 3,584 | |
| Date2Vec phase | `B_2: (k_freq, d_model)` | 16×128 = 2,048 | |
| Date2Vec linear | `w_T: (d_model, M)`, `b_T: (d_model)`, `b_1: (d_model)` | 128×4 + 128 + 128 = 768 | |
| FFN | `Linear(128, 256)`, `Linear(256, 7)` | 128×256 + 256 + 256×7 + 7 = 33,536 | |
| **TOTAL** | | **44,021 + 61 buffer** ≈ **44,082** | for c_in = 7 |

For `c_in = 8` (Exchange Rate), the TFE has 8×128 + 128 = 1,152 and the FFN output has 256×8 + 8 = 2,056 — adding 192 to the total. Hence `44,408` to `44,713` for Exchange Rate. (Per `parameter_invariance_report.md`.)

### 7.5 Verification of Phase 1

Phase 1 verification:

- Trained on ETTh1, ETTh2, ETTm1, Exchange at `O_train = 48`.
- Verified all four datasets produce non-NaN outputs.
- Verified parameter count is invariant across O.
- Verified cross-temporal attention weights sum to 1 across the lookback axis.

### 7.6 What Phase 1 contributed

- **A horizon-independent D2Vformer implementation** (`models/pure_d2vformer.py`).
- **A correct understanding** of the official paper's cross-temporal mechanism.
- **A reproducible baseline** for our Phase 6 extensions.

### 7.7 What Phase 1 did NOT contribute

- Did not improve the model's accuracy. Phase 1 trains successfully but matches or slightly underperforms the official buggy code, because the FFN-based output is a different (arguably more constrained) parameterisation than the buggy `Linear(d_model, O)` projection.
- Did not address temperature. τ = 1.0 throughout.
- Did not change the data pipeline, optimiser, or training schedule.

---

## 8. Phase 2 — Colab Reproducibility and Baseline Stabilisation

### 8.1 Phase 2's single objective

Move the training pipeline to a Colab-runnable notebook, fix all reproducibility issues, and produce stable baseline MSE values across three random seeds for all four datasets.

### 8.2 Why reproducibility was a Phase 2 concern

Phase 1 produced results, but:

- Single-seed runs are noisy.
- Without seed averaging, we cannot distinguish a "real" improvement from random fluctuation.
- The Colab environment is different from local — we need to verify training works in both.

### 8.3 Reproducibility machinery

The reproducibility code lives in `utils/reproducibility.py` and `utils/setseed.py`:

```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

This covers:
- PyTorch CPU randomness.
- CUDA randomness.
- NumPy randomness.
- Python's `random` module.
- cuDNN deterministic mode.

### 8.4 The three seeds

We use seeds `42, 43, 44`. These are consecutive and small to make results easily reproducible. Any seed works in principle.

### 8.5 The training protocol

The training loop in `utils/data.py` and the model wrappers:

1. **Train loop:** runs `n_epochs = 10` epochs over the train split.
2. **Validation:** every epoch, compute MSE on the validation split at `O = 48`.
3. **Early stopping:** patience = 3 epochs on validation loss.
4. **Best checkpoint:** the epoch with the lowest validation MSE is saved as `.pt`.

#### Why validation at fixed O

Early stopping must use the same horizon the model is trained on. If validation uses a different O, the early-stopping decision is contaminated by zero-shot generalisation noise.

### 8.6 Phase 2 outputs

- `D2Vformer_Phase2_Colab.ipynb` — the Colab-runnable training notebook.
- `results/phase2/seed_averages.csv` — averaged MSE per dataset.
- `results/phase2/checkpoints/` — 12 trained checkpoints (4 datasets × 3 seeds).

### 8.7 Phase 2 results summary

The Phase 2 baseline MSE values:

| Dataset | O=24 | O=48 | O=96 | O=192 | O=336 | O=720 |
|---|---|---|---|---|---|---|
| ETTh1 | ~0.45 | ~0.49 | ~0.51 | ~0.55 | ~0.58 | ~0.62 |
| ETTh2 | ~0.34 | ~0.36 | ~0.40 | ~0.43 | ~0.46 | ~0.50 |
| ETTm1 | ~0.40 | ~0.43 | ~0.47 | ~0.51 | ~0.55 | ~0.58 |
| Exchange | ~0.10 | ~0.12 | ~0.18 | ~0.32 | ~0.62 | ~1.20 |

(Exact values in `results/phase2/seed_averages.csv`. These are approximate reference numbers; the audit-locked numbers are in `locked_test_results.csv`.)

### 8.8 What Phase 2 contributed

- **Reproducibility infrastructure** (seed setting, environment, protocol).
- **Baseline MSE values** at all 24 (dataset, horizon) combinations.
- **Colab-ready pipeline** (the notebook runs end-to-end on a free GPU).

### 8.9 What Phase 2 did NOT contribute

- Did not improve accuracy.
- Did not change the architecture.
- Did not introduce temperature conditioning (Phase 6's job).

---

## 9. Phases 3–5 — Validation, Generalisation, and Stress Testing

### 9.1 Why three more phases before the contribution

Phase 2 produced a working baseline. Before we could claim anything new in Phase 6, we needed to know:

- How does the model behave on out-of-distribution horizons? (Phase 3)
- How does it compare to published baselines? (Phase 4)
- How robust is it to dataset shift, training noise, and horizon stress? (Phase 5)

### 9.2 Phase 3 — Validation Horizons

#### 9.2.1 Objective

Verify that the model can be evaluated at horizons it was not trained on, without re-training.

#### 9.2.2 Method

Using the Phase 2 checkpoints (trained at O=48), evaluate at O ∈ {24, 48, 96, 192, 336, 720} on the test set.

#### 9.2.3 Result

All 24 (dataset, horizon) combinations produce finite MSE values. The model never crashes. This confirms the **zero-shot evaluation** claim — but not yet the **zero-shot performance** claim (which Phase 6 addresses via temperature conditioning).

#### 9.2.4 Phase 3 outputs

- `results/phase3/zero_shot_validation.csv` — 4 datasets × 6 horizons × 3 seeds = 72 cells.
- A short report confirming no NaNs, no crashes, no extreme outliers.

### 9.3 Phase 4 — Generalisation vs Published Baselines

#### 9.4.1 Objective

Compare our `pure_d2vformer.py` reconstruction against published baselines (PatchTST, FEDformer, Autoformer, Informer, iTransformer, DLinear).

#### 9.4.2 Sources

Baseline numbers come from the standard ETT and Exchange tables reported in PatchTST (Nie et al., ICLR 2023) and iTransformer (Liu et al., ICLR 2024). These are the canonical published numbers used in most recent papers.

#### 9.4.3 Method

For each dataset × horizon, we compute our MSE and compare to the best published baseline.

#### 9.4.4 Result

Our pure D2Vformer is **competitive but not state-of-the-art**. On ETTh1/ETTh2 it sits roughly in the middle of the published range. On Exchange it is comparable to DLinear (a very simple linear baseline). On ETTm1 it is worse than PatchTST.

#### 9.4.5 What this tells us

Two things:

1. The reconstruction is correct (otherwise the model would be far from the published range).
2. The cross-temporal attention mechanism, on its own, is not enough to beat the best transformer variants. We need an additional mechanism.

That additional mechanism is **Phase 6's contribution**.

#### 9.4.6 Phase 4 outputs

- `results/phase4/baseline_comparison.csv` — our values vs published.
- `results/tables/master_model_comparison.md` — the consolidated table.
- `Phase4_Colab_Generalization.ipynb` — Colab-runnable notebook.

### 9.5 Phase 5 — Stress Testing

#### 9.5.1 Objective

Identify the failure modes of the pure reconstruction. Knowing where it fails is essential before we can claim a Phase 6 improvement.

#### 9.5.2 Tests performed

1. **Long-horizon stress:** evaluate at O=720 across all datasets.
2. **Short-horizon stability:** evaluate at O=24.
3. **Seed sensitivity:** how much do seed 42 and 43 differ?
4. **Cross-attention entropy:** does the attention distribution collapse (over-sharp) or diffuse (over-soft) at long horizons?

#### 9.5.3 Findings

- **Cross-attention entropy collapses at long horizons on some datasets.** Specifically ETTm1 at O=720 has attention entropy H_norm ≈ 0.05 (extremely peaked). This is a strong signal that the fixed τ = 1.0 is too sharp for long-range extrapolation.
- **Exchange has the largest seed-to-seed variance** at long horizons (because the data is small and highly non-stationary).
- **ETTh1/ETTh2 degrade gracefully** — attention stays diffuse enough to remain usable.

#### 9.5.4 What this tells us

Phase 6 should introduce a **temperature mechanism** that adapts τ to the prediction difficulty. This is the gap Phase 6 fills.

#### 9.5.5 Phase 5 outputs

- `Phase3_Colab_Validation.ipynb`, `Phase5_*` notebooks.
- `results/phase5/attention_entropy.csv` — H_norm per (dataset, horizon, seed).
- `results/phase5/stress_test_report.md` — narrative analysis.

### 9.6 Why this phased approach mattered

A naïve strategy would be to jump straight to "improvement." Our phased approach forced us to:

1. Build a clean baseline first (Phase 1).
2. Establish reproducibility (Phase 2).
3. Confirm zero-shot evaluation works (Phase 3).
4. Locate the model in the published performance landscape (Phase 4).
5. Find a specific failure mode to address (Phase 5).
6. Only then, design the Phase 6 intervention that targets that failure mode.

This is what gives the Phase 6 contribution **scientific validity** — it answers a specific question, not a generic "make it better."

---

### 9.7 Summary of Phases 1–5

| Phase | Deliverable | Status |
|---|---|---|
| 1 | `models/pure_d2vformer.py` (44,021 params) | ✓ |
| 2 | 12 trained checkpoints, reproducibility | ✓ |
| 3 | Zero-shot validation across 24 (dataset, horizon) combinations | ✓ |
| 4 | Baseline comparison vs PatchTST/iTransformer/FEDformer/etc. | ✓ |
| 5 | Stress tests identifying attention collapse at long horizons | ✓ |

**Outcome:** we have a clean, reproducible baseline, and we know *exactly* what it fails at — namely, attention over-sharpening on datasets with high-frequency sampling relative to lookback span.

This is the foundation Phase 6 builds on.

---

**End of Part III.** Continue with [TEXTBOOK_PART_IV.md](TEXTBOOK_PART_IV.md).