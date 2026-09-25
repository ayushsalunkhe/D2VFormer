# PROJECT_KNOWLEDGE_MAP.md

> **Status:** Study aid. Not a research artifact.
> **Purpose:** Capture everything needed to write the project textbook without re-deriving it. Authoritative sources are cited; discrepancies are flagged with the source of truth. Research artifacts are **frozen** — do not modify.
> **Companion to:** the master context passed earlier (`MASTER CONTEXT` block) and the audit chain in `results/final_audit/`.

---

## 0. How to read this document

This file is organised so the textbook author can:

1. Start with **§1 Identity & Status** (what this project *is* and what is *off-limits*).
2. Use **§2 Source-of-Truth Hierarchy** when any number is questioned.
3. Use **§3 Glossary** for precise definitions (each term has a simple, technical, and code-level reading).
4. Use **§4–§6** as the technical core (architecture → data → experiment).
5. Use **§7 Results** as the empirical record.
6. Use **§8 Discrepancies** before quoting any number.
7. Use **§9 Open Questions** to know what still has to be resolved before the textbook is complete.

---

## 1. Project Identity & Status

### 1.1 What this project is

A **Bachelor of Engineering (Computer Engineering) major project, AY 2025–26**. It audits the base paper **D2Vformer** (Wang et al., *IEEE TNNLS* / arXiv:2409.11024), reconstructs a horizon-independent backbone (**PureD2Vformer**), diagnoses the cross-temporal attention mechanism, validates scalar attention-temperature selection, and proposes an architectural extension (**TCD2Vformer**) where attention temperature is a learned function of Date2Vec phase embeddings. The final contribution has been formally audited and frozen.

### 1.2 What is OFF LIMITS

| Item | Status | Action |
|---|---|---|
| Phases 1–6 + Final Audit | ✅ Complete, **FROZEN** | Do not re-run, modify, or extend |
| Phase 7 | ❌ Does not exist | Do not propose |
| `results/phase6/*.csv`, `*.pt` checkpoints | **Locked** | Read-only; treat as ground truth |
| `results/final_audit/*` | **Locked** | Read-only; treat as audit record |
| Architectural variants beyond the 4 modes | ❌ Out of scope | Do not propose |
| New temperature grids (continuous τ search) | ❌ Replaced by dynamic τ | Do not revisit |
| Statistical significance at p < 0.05 with n=3 | ❌ Statistically underpowered | Report Cohen's d + seed consistency instead |
| Test-set lookahead for selection | ❌ Forbidden by audit | Validation-only selection only |

### 1.3 Three nested contribution levels

1. **Reconstruction (Phase 1):** A genuinely horizon-independent backbone.
2. **Empirical / diagnostic (Phases 2–4):** First attention-entropy study of D2Vformer; counterfactual controls; validation-only τ selection protocol.
3. **Architectural (Phase 6):** Dynamic per-query attention temperature via Date2Vec-conditioned MLP.

**Inherited from base literature:** Date2Vec harmonic embeddings, RevIN, flexible/arbitrary-length forecasting concept, the ETT/Exchange benchmark datasets.

---

## 2. Source-of-Truth Hierarchy

When sources conflict, use this priority order. Lower-priority sources are summaries or interpretations; they may be stale.

| Priority | Source | Why it has authority | Used for |
|---|---|---|---|
| 1 | `results/phase6/checkpoints/*.pt` (48 files) | The trained models themselves | Re-verifiable outputs, parameter counts |
| 2 | `results/phase6/validation_results.csv` (48 rows) | Pre-test decisions | τ selection, mode selection |
| 3 | `results/phase6/locked_test_results.csv` (288 rows) | Frozen test output | Headline test MSE/MAE numbers |
| 4 | `results/phase6/ablation_results.csv` (96 rows) | Per-mode per-horizon aggregates | Comparison of the four modes |
| 5 | `results/phase6/attention_diagnostics.csv` (288 rows) | H_norm and N_eff | Attention entropy claims |
| 6 | `results/final_audit/*.csv` and `*.md` | The audit record | All "verified" or "audited" claims |
| 7 | `models/tcd2vformer.py` | The Phase 6 architecture of record | Code-level architecture truth |
| 8 | `models/pure_d2vformer.py` | The Phase 1 reconstruction | Code-level reconstruction truth |
| 9 | `results/phase6/PHASE6_CONCLUSION.md` | Audit-cleared narrative summary | Headline tables |
| 10 | `results/tables/master_model_comparison.md` | Phase 4 master table (scalar τ) | Pre-Phase 6 baseline numbers |

| Lower trust | Source | Why to be cautious |
|---|---|---|
| ↓ | `README.md` | Title says "TC-D2Vformer", body mixes Phase 4/6 numbers; key results table is Phase 4 scalar-τ; limitations quoted from Phase 5. |
| ↓ | `docs/architecture.md` | **Predates Phase 6** — describes only `fixed` / `learnable` scalar τ, no MLP, no four-mode ablation. Superseded by `models/tcd2vformer.py`. |
| ↓ | `docs/methodology.md` | **Predates Phase 6** — research question is the scalar-τ question, not the dynamic-τ question. |
| ↓ | `docs/results.md` | **Phase 2–4 numbers**, not Phase 6. Use for narrative history. |
| ↓ | `docs/limitations.md` | Future-work list (F1–F6) is now partly obsolete: F1 (learnable τ) and F2 (head-wise τ) were implemented in Phase 6; F3 was replaced by dynamic τ. |
| ↓ | `docs/experimental_protocol.md` | Phase 2–4 protocol; Phase 6 protocol is in `results/phase6/protocol.md`. |
| ↓ | `docs/reproducibility.md` | Phase 5 SHA-256 table; Phase 6 SHA-256 verification is in audit Task 1. |
| ↓ | `RESEARCH_FINDINGS.md` Phase 6 table | Contains the **documented validation-value typo** (see §8). Phase 2/3/4 narratives are sound. |
| ↓ | `results/phase6/analysis.md` | Same validation-value typo as above. |
| ↓ | `model/` directory | Legacy code from the base D2Vformer fork; not part of the audited chain. **Do not use** as architecture of record. |
| ↓ | `models/temperature_d2vformer.py` | Phase 3–4 scalar-τ model; superseded by `models/tcd2vformer.py` (Phase 6) but kept as an alias for backward compatibility. |

**Rule:** when in doubt, quote from the higher-priority source. If a *lower*-priority source disagrees with a higher one, the higher wins — and the lower should be flagged for correction, not edited silently.

---

## 3. Glossary (three levels each)

### 3.1 D2Vformer

- **Level 1 (intuition):** A neural network that predicts future time-series values by asking "which past time steps look like the future I'm trying to predict?" — where "looking like" is measured using a learned representation of time.
- **Level 2 (technical):** A cross-temporal-attention forecasting model that maps historical inputs and historical/future timestamps into a shared Date2Vec space, then aggregates historical values weighted by their temporal similarity to each query step.
- **Level 3 (code):** Implemented in `models/pure_d2vformer.py` (our reconstruction) and the original `model/D2Vformer.py` (legacy base repo).

### 3.2 PureD2Vformer

- **Level 1:** Our cleaned-up version of D2Vformer where the parameter count does not change if you ask it to forecast a different number of steps.
- **Level 2:** A reimplementation that replaces any horizon-dependent linear projection with the parameter-free aggregation `Y = A · T^T`. SHA-256 checksums verified across O ∈ {24, 48, 96, 192, 336, 720}.
- **Level 3:** `models/pure_d2vformer.py`. Parameter count: 44,021 (c_in = 7), 44,408 (c_in = 8). Has `assert_horizon_independence()` (in the Phase 6 file), equivalent functionality in unit tests.

### 3.3 TC-D2Vformer / Temperature-Controlled D2Vformer

- **Level 1:** PureD2Vformer plus a single "sharpness knob" applied to the attention.
- **Level 2:** Cross-temporal attention is scaled by a scalar τ: `A = softmax(S / τ)`. τ is selected per (dataset, seed) using validation loss at O_train = 48 from a small grid {0.5, 1.0, 2.0, 4.0}. Zero learnable parameters are added.
- **Level 3:** `models/temperature_d2vformer.py` (Phase 5–6 scalar-τ model). Note this file is older; for the dynamic-τ extension see TCD2Vformer.

### 3.4 TCD2Vformer / Temporal-Conditioned Date2Vecformer

- **Level 1:** PureD2Vformer where the sharpness knob is automatically set per forecast step, based on what time the model is trying to predict.
- **Level 2:** The attention temperature τ becomes a learned function `τ = f_θ(D_x, D_y)`, computed by a small MLP (Linear(17, 16) → GELU → Linear(16, 1)) whose input is the mean of Date2Vec embeddings. Four temperature modes evaluated: `fixed`, `learned_global`, `temporal_context`, `query_conditioned`. Adds exactly 305 trainable parameters regardless of O.
- **Level 3:** `models/tcd2vformer.py`, class `TCD2Vformer`. Method `compute_temperature(Dx, Dy)` returns a tensor broadcastable to `[B, H, O, L]`. Method `assert_horizon_independence()` programmatically verifies that params(O=24) = params(O=720).

### 3.5 Date2Vec

- **Level 1:** A way to turn "what time is it?" (year, month, day, hour, minute, …) into a vector of numbers that a neural network can use.
- **Level 2:** A learnable harmonic embedding `D = [t, sin(ω_k · t + φ_k), cos(ω_k · t + φ_k)]` parameterised by frequency vector ω and phase φ. The model learns which periodicities matter.
- **Level 3:** Built into `models/pure_d2vformer.py` via `nn.Parameter(torch.randn(seq_len))` (`w_T`, `W_S`) and `nn.Parameter(torch.zeros(...))` (`b_T`, `B_S`). Dimensions: k_freq = 16 harmonic components + 1 linear component = 17 total per (B, H, L) entry. Implemented in `layers/Date2Vec.py` as well for the older interface.

### 3.6 Cross-temporal attention

- **Level 1:** A mechanism that decides, for each future time step, how much weight to put on each past time step when forming its prediction.
- **Level 2:** Given Date2Vec embeddings D_x ∈ ℝ^{B×H×L×d_k} of past times and D_y ∈ ℝ^{B×H×O×d_k} of future times, the similarity is `S = D_y · D_xᵀ / √(k_freq+1) ∈ ℝ^{B×H×O×L}`. Attention `A = softmax(S/τ)` is row-stochastic over L. Output `Y = A · T` where T is the temporal-feature projection of the past inputs.
- **Level 3:** Lines 90–104 of `models/pure_d2vformer.py`, lines 195–225 of `models/tcd2vformer.py`. Tensor: A ∈ [B, d_model, O, L] — note `d_model` is the attention-head-like dimension along which the Date2Vec embedding is replicated.

### 3.7 Temperature τ

- **Level 1:** A single number that decides whether the model focuses sharply on a few past steps (small τ) or spreads attention evenly across all past steps (large τ).
- **Level 2:** A positive scalar dividing the logits inside softmax: `A = exp(S/τ) / Σ exp(S/τ)`. Smaller τ → sharper distribution; larger τ → softer distribution. As τ → ∞, A → uniform 1/L. As τ → 0⁺, A → one-hot at argmax.
- **Level 3:** In `fixed` mode: `register_buffer("tau_fixed", torch.tensor(1.0))`. In `learned_global`: `τ = softplus(τ_raw) + ε`, with `τ_raw` initialised by inverse softplus of 1.0. In dynamic modes: `τ = τ_min + softplus(MLP(·))`, with τ_min = 0.1 by default.

### 3.8 H_norm (Normalised attention entropy)

- **Level 1:** A measure between 0 and 1 of how spread out the attention is. 1.0 means attention is perfectly uniform; 0.0 means attention is a single point.
- **Level 2:** `H_norm = -Σ_l A_{o,l} · ln(A_{o,l}) / ln(L)`. The denominator ln(L) normalises so a uniform distribution gives H_norm = 1.
- **Level 3:** Computed per-row of A in attention diagnostics. Implementation in `results/phase6/attention_diagnostics.csv`.

### 3.9 N_eff (Effective number of attended timestamps)

- **Level 1:** How many past time steps are "really" being attended to, accounting for the fact that some get nearly-zero weight.
- **Level 2:** `N_eff = exp(H(A))` where H is the Shannon entropy of the attention row. If attention were perfectly uniform over L positions, N_eff = L. If attention were one-hot, N_eff = 1.
- **Level 3:** Always reported with H_norm in the attention diagnostics CSV. Relationship: `H_norm = ln(N_eff) / ln(L)` → N_eff = L^{H_norm}.

### 3.10 Zero-shot forecasting

- **Level 1:** Training it to forecast N steps ahead and then, at test time, asking it to forecast M steps ahead (where M can be different from N) without retraining.
- **Level 2:** The model's parameters are not re-fit per horizon O; the same trained model is evaluated at multiple O. In this project, training is at O_train = 48; test horizons are {24, 48, 96, 192, 336, 720}.
- **Level 3:** `O_train = 48` is hard-coded in the training scripts; the trained checkpoint is then evaluated at six horizons. The same `model.forward(x_enc, x_mark_enc, y_mark_dec)` is called with different `y_mark_dec` shapes.

### 3.11 Horizon independence (parameter-level)

- **Level 1:** The number of learnable weights in the model does not change with the forecast horizon O.
- **Level 2:** Mathematically, `∂N_params / ∂O ≡ 0` for all evaluated O ∈ {24, 48, 96, 192, 336, 720}. No linear layer has an output dimension that depends on O.
- **Level 3:** Verified by `assert_horizon_independence()` in `models/tcd2vformer.py` and the unit tests in `tests/test_tcd2vformer.py`. Audit: 96 / 96 checks passed.

### 3.12 RevIN (Reversible Instance Normalisation)

- **Level 1:** A trick to normalise each time-series sample to zero mean and unit variance before the model sees it, then reverse the normalisation on the model's output.
- **Level 2:** Instance normalisation per (batch, channel) on the lookback window; affine γ, β ∈ ℝ^{c_in} are learned. Implemented in `layers/Revin.py`.
- **Level 3:** Used in both `pure_d2vformer.py` and `tcd2vformer.py` (lines `self.revin = RevIN(c_in, affine=True, subtract_last=False)`).

### 3.13 Cohen's d

- **Level 1:** A way to say how big a difference is, in units of "how spread out the data is." A d of 1.0 means the two conditions differ by about one standard deviation.
- **Level 2:** `d = (mean_paired_difference) / (std_paired_difference)`. With n=3 seeds, df = 2; the critical t-value at α = 0.05 is 4.303, so p < 0.05 requires d so large that it is essentially unattainable in this study.
- **Level 3:** Reported in `results/final_audit/statistical_robustness.csv`. ETTh2 O=720: d = 1.67, p = 0.102.

---

## 4. Architecture Reference

### 4.1 Tensor flow (single forward pass)

| Stage | Symbol | Shape | Source |
|---|---|---|---|
| Input series | `x_enc` | [B, L, c_in] | dataset |
| Past timestamps | `x_mark_enc` | [B, L, M] | M=4 (year, month, day, weekday normalised) |
| Future timestamps | `y_mark_dec` | [B, O, M] | O varies at inference |
| Normalised series | `x_norm` | [B, L, c_in] | `RevIN(x_enc, 'norm')` |
| Temporal features | `T` | [B, L, d_model] | `Linear(c_in, d_model)` (TFE, Eq. 3) |
| Date2Vec past | `D_x` | [B, k+1, d_model, L] | Date2Vec on x_mark_enc |
| Date2Vec future | `D_y` | [B, k+1, d_model, O] | Date2Vec on y_mark_dec |
| Similarity | `S` | [B, d_model, O, L] | `(D_y · D_xᵀ) / √(k+1)` |
| Temperature | `τ` | scalar / [B,1] / [B,O,1] / [B,1,O,1] | depends on mode |
| Attention | `A` | [B, d_model, O, L] | `softmax(S/τ, dim=-1)` |
| Aggregated hidden | `Y_tilde` | [B, d_model, O] | `A · Tᵀ` |
| Forecast hidden | `Y_hat` | [B, O, d_model] | transpose for FFN |
| Forecast output | `Y_hat` | [B, O, c_in] | `Linear(d_model, c_in)` |
| De-normalised | `Y_out` | [B, O, c_in] | `RevIN(Y_hat, 'denorm')` |

Tensor shapes in d_model = 128, k_freq = 16, d_ff = 256, c_in ∈ {7, 8}, L = 96, M = 4, O ∈ {24, 48, 96, 192, 336, 720}.

### 4.2 Hyperparameters (canonical)

| Hyperparameter | Value | Where set |
|---|---|---|
| `c_in` | 7 (ETT*) / 8 (Exchange) | dataset-dependent |
| `d_model` | 128 | constructor |
| `d_ff` | 256 | constructor |
| `k_freq` | 16 | constructor |
| `dropout` | 0.05 | constructor |
| `L` | 96 | hard-coded |
| `O_train` | 48 | hard-coded |
| `τ candidates` (Phase 4) | {0.5, 1.0, 2.0, 4.0} | selection protocol |
| `τ_min` | 0.1 | TCD2Vformer default |
| `d_temp` (MLP hidden) | 16 | TCD2Vformer default |
| `lr` | 1e-3 | training script |
| `epochs` | 10 | training script |
| `patience` | 3 | early stopping |
| `batch_size` | 64 | training script |
| `seeds` | {42, 43, 44} | fixed by audit |

### 4.3 The four temperature modes (Phase 6)

| Mode | τ shape | Parameters added | When τ is computed |
|---|---|---|---|
| `fixed` | scalar | 0 | from `tau_fixed` buffer (initially 1.0) |
| `learned_global` | scalar per batch | +1 | `softplus(τ_raw)` from single `nn.Parameter` |
| `temporal_context` | [B, 1, 1, 1] | +305 | `τ_min + softplus(MLP(mean(D_x) over L))` |
| `query_conditioned` | [B, 1, O, 1] | +305 | `τ_min + softplus(MLP(mean(D_y) over H))` per query |

### 4.4 Date2Vec parameter budget

| Parameter | Shape | Total | Horizon-dep? |
|---|---|---|---|
| `w_T` (linear weights over L) | [L] = [96] | 96 | No |
| `b_T` (linear bias) | [d_model] | 128 | No |
| `W_S` (harmonic weights) | [k_freq, L] | 1,536 | No |
| `B_S` (harmonic bias) | [k_freq, d_model] | 2,048 | No |
| `b_1`, `b_3` | [d_model, 1, 1] | 256 | No |
| `B_2`, `B_4` | [k_freq, d_model, 1, 1] | 4,096 | No |
| TFE: `Linear(c_in, d_model)` | [c_in, d_model] + bias | (c_in × d_model) + d_model | No |
| FFN: `Linear(d_model, d_ff)` | + bias | 32,896 | No |
| FFN: `Linear(d_ff, c_in)` | + bias | (d_ff × c_in) + c_in | No |
| RevIN: γ, β | 2 × c_in | 14 (c_in=7) / 16 (c_in=8) | No |
| `temp_mlp[0]` (TCD2Vformer) | (17 × 16) + 16 | 288 | No |
| `temp_mlp[2]` (TCD2Vformer) | (16 × 1) + 1 | 17 | No |

Sum for c_in = 7 (no τ MLP): 96 + 128 + 1536 + 2048 + 128 + 128 + 2048 + 2048 + 7×128 + 128 + 128×256 + 256 + 256×7 + 7 + 14 = 44,021 ✓
Sum for c_in = 7 (with τ MLP): 44,021 + 305 = 44,326 ✓
Sum for c_in = 8: 44,408 (no τ MLP), 44,713 (with τ MLP) ✓

(The bookkeeping is verified by the audit. The exact param breakdown in code can be cross-checked by `sum(p.numel() for p in model.parameters())`.)

---

## 5. Datasets, Splits & Evaluation

### 5.1 Datasets

| Dataset | Source | Sampling | Channels | Samples | Split pattern |
|---|---|---|---|---|---|
| ETTh1 | ETT (Zhou et al., AAAI 2021) | 1 hour | 7 | 17,420 | 60 / 20 / 20 |
| ETTh2 | ETT | 1 hour | 7 | 17,420 | 60 / 20 / 20 |
| ETTm1 | ETT | 15 min | 7 | 69,680 | 60 / 20 / 20 |
| Exchange | LSTNet / open | 1 day | 8 | 7,588 | 60 / 20 / 20 |

All splits are strictly chronological (no shuffling, no leakage). Feature standardisation μ, σ computed only on the training partition. ETTh2 and ETTm1 are "unseen" datasets for the Phase 4 scalar-τ grid was selected on ETTh1 + Exchange; τ candidates {0.5, 1.0, 2.0, 4.0} are pre-registered and not data-dependent.

### 5.2 Physical horizon translation (15-minute ETTm1)

| Forecast horizon O | Real-world span | Real-world span (days) |
|---|---|---|
| 24 | 6 hours | 0.25 |
| 48 | 12 hours | 0.5 |
| 96 | 24 hours | 1.0 |
| 192 | 48 hours | 2.0 |
| 336 | 84 hours | 3.5 |
| 720 | 180 hours | 7.5 |

This matters for the ETTm1 failure analysis: at O = 720, the lookback window (24 h) covers only 1 day, but the forecast spans 7.5 days. The learned τ ≈ 0.18 sharpens attention onto specific 15-minute lags that may not align with the future phase.

### 5.3 Evaluation matrix

| Quantity | Value |
|---|---|
| Datasets | 4 (ETTh1, ETTh2, ETTm1, Exchange) |
| Seeds | 3 (42, 43, 44) |
| Temperature modes | 4 (`fixed`, `learned_global`, `temporal_context`, `query_conditioned`) |
| Trained checkpoints | 4 × 3 × 4 = **48** |
| Horizons | 6 ({24, 48, 96, 192, 336, 720}) |
| Locked evaluations | 48 × 6 = **288** |
| Ablation table rows | 4 × 4 × 6 = **96** |
| Validation rows | 4 × 3 × 4 = **48** |
| Attention-diagnostic rows | 288 |

---

## 6. Experimental Protocol

### 6.1 Training (Phase 6)

- All models trained at O_train = 48.
- Optimiser: Adam, lr = 1e-3.
- Epochs = 10 with early stopping (patience = 3).
- Batch size = 64.
- Loss: MSE on normalised scale; reported MSE is on original scale (after RevIN denorm).
- Seeds set deterministically: `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`.

### 6.2 Selection protocol

For each (dataset, seed), the selected mode is:

```
m* = argmin_{m ∈ {fixed, learned_global, temporal_context, query_conditioned}}  val_MSE(m, O=48)
```

Tie-breaking (validation margins < 1e-4) follows pre-registered order: `temporal_context` → `query_conditioned` → `learned_global` → `fixed`. After selection, the chosen checkpoint is locked (SHA-256 checksum recorded) and evaluated at all six horizons on the test set.

### 6.3 What "zero-lookahead" means here

The test set is never inspected during model selection, τ selection, mode selection, or hyperparameter tuning. Every decision traceable to validation loss at O = 48.

### 6.4 Audit artefacts

- 48 / 48 SHA-256 parameter checksums match across reload.
- 12 / 12 (dataset × seed) selection decisions reconstructed from validation loss only.
- 96 / 96 horizon-independence parameter checks passed.
- 288 / 288 evaluations dimensions verified.

---

## 7. Results Reference

### 7.1 Phase 6 — Master ablation table (test MSE, mean over 3 seeds)

Source: `results/phase6/PHASE6_CONCLUSION.md`, cross-checked against `ablation_results.csv`.

| Dataset | O | Fixed (τ=1) | Learned Global | Temporal Context | Query Cond. | Best | Δ% vs Fixed |
|---|---|---|---|---|---|---|---|
| **ETTh1** | 24 | **0.8302** | 0.8314 | 0.8514 | 0.8444 | fixed | baseline |
| | 48 | 0.8510 | **0.8505** | 0.8702 | 0.8653 | LG | +0.06% |
| | 96 | 0.8779 | **0.8768** | 0.8960 | 0.8943 | LG | +0.12% |
| | 192 | 0.9247 | **0.9226** | 0.9404 | 0.9404 | LG | +0.23% |
| | 336 | 0.9527 | **0.9522** | 0.9684 | 0.9707 | LG | +0.05% |
| | 720 | 1.0944 | **1.0916** | 1.1101 | 1.1117 | LG | +0.25% |
| **ETTh2** | 24 | **0.2268** | 0.2270 | 0.2280 | 0.2280 | fixed | baseline |
| | 48 | **0.2475** | 0.2477 | 0.2482 | 0.2482 | fixed | baseline |
| | 96 | **0.2784** | 0.2787 | 0.2786 | 0.2785 | fixed | baseline |
| | 192 | 0.3105 | 0.3107 | **0.3076** | **0.3076** | TC/QC | +0.93% |
| | 336 | 0.3422 | 0.3424 | **0.3348** | **0.3348** | TC/QC | +2.18% |
| | 720 | 0.4104 | 0.4105 | **0.3976** | **0.3976** | TC/QC | **+3.12%** |
| **ETTm1** | 24 | 0.6716 | 0.6654 | **0.6634** | 0.6656 | TC | +1.22% |
| | 48 | **0.7163** | 0.7171 | 0.7196 | 0.7219 | fixed | baseline |
| | 96 | **0.8815** | 0.9008 | 0.8942 | 0.9092 | fixed | baseline |
| | 192 | **0.9285** | 0.9488 | 0.9422 | 0.9622 | fixed | baseline |
| | 336 | **0.9626** | 0.9828 | 0.9782 | 1.0057 | fixed | baseline |
| | 720 | **1.0300** | 1.0512 | 1.0421 | 1.0741 | fixed | baseline |
| **Exchange** | 24 | **0.10281** | 0.10282 | 0.10287 | 0.10287 | fixed | baseline |
| | 48 | 0.12738 | 0.12740 | **0.12733** | **0.12733** | TC/QC | +0.04% |
| | 96 | 0.17882 | 0.17884 | **0.17874** | **0.17874** | TC/QC | +0.04% |
| | 192 | 0.29334 | 0.29335 | **0.29305** | **0.29305** | TC/QC | +0.10% |
| | 336 | 0.48682 | 0.48683 | **0.48612** | **0.48612** | TC/QC | +0.14% |
| | 720 | 1.23203 | 1.23208 | **1.23100** | **1.23100** | TC/QC | +0.08% |

### 7.2 ETTh2 long-horizon — seed-by-seed

Source: `results/final_audit/etth2_long_horizon_analysis.md`.

| O | Fixed (mean) | QC (mean) | Δ% (mean) | 3/3 seeds positive? |
|---|---|---|---|---|
| 24 | 0.22679 | 0.22798 | −0.52% | No (2/3) |
| 48 | 0.24747 | 0.24816 | −0.28% | No (2/3) |
| 96 | 0.27840 | 0.27854 | −0.05% | No (1/3) |
| 192 | 0.31048 | 0.30758 | +0.93% | No (2/3) |
| 336 | 0.34225 | 0.33479 | +2.18% | **Yes (3/3)** |
| 720 | 0.41038 | 0.39757 | **+3.12%** | **Yes (3/3)** |

Per-seed O = 720: Seed 42 +2.10%, Seed 43 +5.18%, Seed 44 +2.03%. Cohen's d = 1.67. Paired t(2) = 2.88, p = 0.102.

### 7.3 Validation-selected τ (Phase 4, scalar τ, for narrative)

For comparison with the dynamic-τ result:

| Dataset | τ* (42) | τ* (43) | τ* (44) |
|---|---|---|---|
| ETTh1 | 4.0 | 2.0 | 4.0 |
| Exchange | 0.5 | 4.0 | 4.0 |
| ETTh2 | 1.0 | 2.0 | 4.0 |
| ETTm1 | 0.5 | 0.5 | 2.0 |

Source: `docs/methodology.md`, `RESEARCH_FINDINGS.md` Phase 4 table.

### 7.4 Phase 6 learned τ values (mean across seeds)

| Mode | ETTh1 | ETTh2 | ETTm1 | Exchange |
|---|---|---|---|---|
| `learned_global` | 0.7285 | 0.9562 | 0.4171 | 0.9731 |
| `temporal_context` | 0.21 | 0.9013 | 0.174 | 0.9622 |
| `query_conditioned` | 0.26 | 0.9041 | 0.18 | 0.9628 |

Source: `demo/index.html` `D` payload, cross-checked with `attention_diagnostics.csv`.

### 7.5 Attention entropy results (Phase 6)

Source: `PHASE6_CONCLUSION.md`, `attention_diagnostics.csv`.

| Dataset | Fixed H_norm → Dynamic H_norm | Fixed N_eff → Dynamic N_eff |
|---|---|---|
| ETTh1 | 0.9714 → 0.8654 (temporal_context) | 87.1 → 60.95 |
| ETTh2 | 0.9828 → 0.9796 (query_conditioned) | 90.0 → 89.0 |
| ETTm1 | 0.9666 → 0.9022 (query_conditioned) | 85.9 → 67.3 |
| Exchange | 0.9696 → 0.9702 (query_conditioned) | 86.4 → 86.6 |

ETTh2's learned τ ≈ 0.90 is "gentle" — barely changes H_norm or N_eff. ETTm1's learned τ ≈ 0.18 is "aggressive" — collapses H_norm and N_eff.

### 7.6 Phase 4 vs Phase 6 — same metric, different intervention

| Result | Phase 4 (scalar τ) | Phase 6 (dynamic τ) |
|---|---|---|
| ETTh1 O=720 Δ | +1.32% | +0.25% |
| Exchange O=720 Δ | +1.34% | +0.08% |
| ETTh2 O=720 Δ | +0.71% | **+3.12%** |
| ETTm1 O=24 Δ | **+3.79%** | +1.22% |
| ETTm1 O=720 Δ | −0.88% | baseline (fixed is best) |

These are *different* interventions and should not be merged. Phase 6 superseded Phase 4 for the headline contribution claim.

### 7.7 Baselines for context (Phase 5)

Source: `results/tables/master_model_comparison.md`.

DLinear (retrained per O) remains the strongest baseline on ETTm1. TC-D2Vformer / TCD2Vformer is the contribution *within* the D2Vformer family — not a SOTA claim against DLinear.

---

## 8. Known Discrepancies (do not propagate)

### 8.1 Validation-value typo in markdown

`results/phase6/analysis.md` (Section 3) and `RESEARCH_FINDINGS.md` (Phase 6 summary) report validation MSE at O = 48 as:

| Dataset | Markdown says | True value (from `validation_results.csv`) |
|---|---|---|
| ETTh1 | 0.8037 | 0.56326 ± 0.0034 |
| ETTh2 | 0.2307 | 0.27554 ± 0.0008 |
| ETTm1 | 0.6723 | 0.45555 ± 0.0053 |
| Exchange | 0.1198 | 0.31498 ± 0.0076 |

**Classification:** Documentation-only. Checkpoints and CSVs verified 100%. Use the CSV values; do not propagate the markdown numbers.

### 8.2 Dataset-name capitalisation

Early phases use `Exchange` / `exchange_rate`; Phase 6 uses `exchange` (lowercase). All map to `datasets/exchange_rate/exchange_rate.csv`. Cosmetic.

### 8.3 Parameter counts in `docs/reproducibility.md`

States 44,150 parameters for c_in = 8. Audit-trusted values are 44,408 (fixed) / 44,409 (learned_global) / 44,713 (TC/QC) for c_in = 8. The `docs/reproducibility.md` figure is a typo — use the audit values.

### 8.4 Parameter count claim in README

`README.md` "Model Architecture" section states "Parameters: 44,021 (C_in=7, d_model=128, d_ff=256, k_freq=16)". This is correct *for Phase 1/3/5*. The Phase 6 numbers for c_in = 7 are 44,021 / 44,022 / 44,326. README omits the conditional, so it is technically correct but incomplete.

### 8.5 `docs/architecture.md` is pre-Phase-6

The file describes only `fixed` and `learnable` scalar τ. It does not describe `temp_mlp`, the four-mode ablation, or the per-query τ formulation. For Phase 6 architecture, refer to `models/tcd2vformer.py` and `FINAL_CONTRIBUTION.md`.

### 8.6 `docs/limitations.md` future-work is partly obsolete

- F1 (learnable temperature) — implemented as `learned_global` in Phase 6.
- F2 (layer-wise / head-wise temperature) — implemented as `query_conditioned` (different mechanism but addresses the same idea of per-position τ).
- F3 (expanded τ grid) — replaced by dynamic τ in Phase 6.
- F4 (larger seed set), F5 (additional datasets), F6 (entropy regularisation) — out of scope after Phase 6 freeze.

### 8.7 README's Key Results table is Phase 4

The table in `README.md` (lines 24–32) reports the Phase 4 scalar-τ result (ETTh1 +0.67%, Exchange +1.17%, ETTh2 +0.95%, ETTm1 +0.44%). The Phase 6 dynamic-τ result (ETTh2 +3.12% at O=720) is described in a different section of the same README ("Phase 6: Temporal-Conditioned Date2Vecformer (TCD2Vformer)" section). The two should be presented in the textbook as **separate headline results**, not merged.

### 8.8 README title uses TC-D2Vformer, body uses TCD2Vformer

This is a terminology drift: Phase 5 used **TC-D2Vformer** (Temperature-Controlled); Phase 6 introduced **TCD2Vformer** (Temporal-Conditioned). The latter is the canonical name for the Phase 6 architecture; the earlier name refers to the scalar-τ variant. Both appear in `models/__init__.py` (`PureD2Vformer`, `TemperaturePureD2Vformer`, `TCD2Vformer`).

### 8.9 Master model comparison numbers don't directly map to Phase 6

`results/tables/master_model_comparison.md` shows the Phase 4 master table (val-selected scalar τ) and includes baselines (Persistence, DLinear, Repo-D2Vformer). The Phase 6 dynamic-τ result is **not** in that table. The Phase 6 headline result lives in `PHASE6_CONCLUSION.md`.

---

## 10. The Statistical Reporting Boundary

### 10.1 What is statistically defensible (n = 3, df = 2)

| Claim | Status | Cite |
|---|---|---|
| ETTh2 O=720: Cohen's d = 1.67, p = 0.102 | ✅ Honest to report | `statistical_robustness.csv` |
| 3 / 3 seeds improve by > 2% at O = 720 | ✅ Honest to report | `etth2_long_horizon_analysis.md` |
| Relative improvement +0.04% to +3.79% across dataset × horizon | ✅ Honest descriptive aggregate | `results.md` |
| **p < 0.05 statistical significance** | ❌ Underpowered; do not claim | — |
| **TCD2Vformer universally improves all datasets** | ❌ ETTm1 O ≥ 96 is worse; do not claim | `ettm1_failure_analysis.md` |
| **TCD2Vformer is state-of-the-art** | ❌ DLinear retrained per O is stronger on ETTm1; do not claim | `master_model_comparison.md` |
| **TCD2Vformer invented arbitrary-length forecasting** | ❌ Inherited; do not claim | `contribution_boundary.md` |
| **TCD2Vformer is the first dynamic temperature mechanism** | ❌ Prior art exists in NLP/vision; do not claim | `contribution_boundary.md` |

### 10.2 The framing rule

> "Across n = 3 independent replications, TCD2Vformer demonstrates substantial long-horizon zero-shot forecasting improvements on ETTh2 (Cohen's d = 1.67, +3.12% at O = 720) and consistent minor gains on Exchange Rate. However, owing to sample-size constraints (df = 2), these results do not achieve conventional statistical significance (p < 0.05) and should be interpreted as strong descriptive evidence rather than universal statistical proof."
>
> — `statistical_robustness.md` §4

This is the only phrasing to use when summarising the result.

---

## 11. What the Textbook Must Explain (open checklist)

These are the items the master context promised the textbook would eventually cover. Use this as a checklist when writing.

### 11.1 First-principles foundations

- [ ] What is time-series forecasting
- [ ] Why horizons matter (and why a single model handling multiple horizons is non-trivial)
- [ ] What D2Vformer is (high level → equation level → code level)
- [ ] Why Date2Vec is used (calendar features → harmonic embedding)
- [ ] How Date2Vec representations are constructed (Eq. 4–7)
- [ ] What harmonic projections mean (frequency × phase parameterisation)
- [ ] How RevIN works and why reversible normalisation is needed
- [ ] What cross-temporal attention means (queries = future, keys/values = past)
- [ ] What Q, K, V mean in this architecture (here K and V are different)
- [ ] How the similarity matrix is constructed (Eq. 8)
- [ ] Why softmax is used (probability distribution over past)
- [ ] What temperature scaling is mathematically (τ in denominator of softmax)
- [ ] Why τ changes attention sharpness (one-hot limit vs uniform limit)
- [ ] What entropy means (Shannon entropy)
- [ ] What H_norm means (entropy normalised to [0, 1] over L)
- [ ] What N_eff means (`exp(H)`, the "effective number" of attended positions)
- [ ] Why attention can become diffuse (small logit variance → near-uniform softmax)
- [ ] Why uniform attention can sometimes be useful (regulariser)

### 11.2 Project-specific foundations

- [ ] What zero-shot forecasting means (train at O_train, eval at different O)
- [ ] Why horizon-dependent output layers break parameter independence (Linear(d_model, O) introduces O weights)
- [ ] How PureD2Vformer removes that dependence (Y = A · T^T)
- [ ] How the temperature experiments were designed (grid τ × validation selection)
- [ ] Why validation / test separation matters (zero-lookahead protocol)
- [ ] How TCD2Vformer generates τ (Linear(17,16) → GELU → Linear(16,1))
- [ ] Why temporal conditioning can help (regime-appropriate τ)
- [ ] Why it can also fail (phase overfitting on 15-min data)
- [ ] What the ETTh2 result means (Cohen's d = 1.67, +3.12% at O = 720, 3/3 seeds)
- [ ] What the ETTm1 negative result means (15-min sampling ÷ 7.5-day horizon)
- [ ] What every tensor dimension means (see §4.1 above)
- [ ] How data flows through the network (see §4.1)
- [ ] What each major code component does (see §4 and code files)
- [ ] What each experiment proves (audited; do not over-claim)
- [ ] What each result does NOT prove (audited; do not under-claim)
- [ ] How to explain the project to a professor (use `FINAL_CONTRIBUTION.md` §12)
- [ ] How to defend the contribution during viva (use the "what to say / what NOT to claim" list)

---

## 12. Open questions / things still to resolve

These are gaps in *my* understanding (the study agent) — they are not bugs in the project, just points the textbook author should double-check.

1. **README title inconsistency** — should the README be updated to reflect Phase 6 terminology, or is the mixed terminology acceptable for historical reasons? *(Audit doesn't forbid the edit but the demo README serves as a snapshot of the project timeline.)*
2. **Test of `tests/test_tcd2vformer.py`** — I have not read the test file contents. Should be reviewed to verify it covers the four-mode horizon independence claim and not just the scalar-τ variant.
3. **Pre-Phase-6 demo content** — the demo HTML has a `tau_simulator` based on scalar τ. It now also has an "overlay all τ" toggle. Whether the demo is for Phase 4 (scalar τ) or Phase 6 (dynamic τ) needs to be explicit.
4. **Whether `models/temperature_d2vformer.py` and `models/tcd2vformer.py` produce the same Phase 1 baseline numbers** — both wrap `pure_d2vformer.py` but with different temperature layers. The fixed-τ baseline should be identical between them.
5. **Whether the model in `demo/index.html`'s D payload matches the final `ablation_results.csv`** — the demo has 96 rows of test MSE values, H_norm, N_eff and improvement values. The textbook should treat the demo as illustrative and the CSV as ground truth.

---

## 13. Quick-reference card

### 13.1 The five numbers the textbook must state correctly

| Number | Value | Source of truth |
|---|---|---|
| ETTh2 O=720 Δ (Phase 6, query_conditioned vs fixed) | **+3.12%** | `PHASE6_CONCLUSION.md`, `etth2_long_horizon_analysis.md` |
| Cohen's d on the above | **1.67** | `statistical_robustness.csv` |
| p-value on the above | **0.102** (df = 2) | `statistical_robustness.csv` |
| Phase 6 trained checkpoints | **48** (4 datasets × 3 seeds × 4 modes) | `FINAL_AUDIT_SUMMARY.md` |
| Horizon-independence audit | **96 / 96** (4 × 4 × 6) | `parameter_invariance_report.md` |

### 13.2 The five numbers the textbook must NOT state

| Anti-claim | Why not |
|---|---|
| "statistically significant at p < 0.05" | df = 2 underpowered |
| "TCD2Vformer universally improves everything" | ETTm1 O ≥ 96 is worse |
| "TCD2Vformer beats DLinear" | DLinear retrained per O is stronger on ETTm1 |
| "TCD2Vformer invented zero-shot forecasting" | Inherited from Wang et al. 2024 |
| "TCD2Vformer is the first dynamic softmax temperature" | Prior art in NLP/vision |

### 13.3 The five files to open first when starting the textbook

1. `results/final_audit/FINAL_CONTRIBUTION.md` — what to claim and what not to
2. `results/phase6/PHASE6_CONCLUSION.md` — the master results table
3. `models/tcd2vformer.py` — the architecture of record
4. `results/final_audit/etth2_long_horizon_analysis.md` — the headline result's per-seed proof
5. `results/final_audit/ettm1_failure_analysis.md` — the most important negative result