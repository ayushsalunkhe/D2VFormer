# D2Vformer to TCD2Vformer — Part VII: What's Inherited, What's Ours

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §20–§22, `results/final_audit/contribution_boundary.md`, `FINAL_CONTRIBUTION.md`.

This part makes explicit which ideas belong to the original D2Vformer paper (and the broader literature) and which ideas belong to us. The thesis defense depends on this distinction.

---

## 20. Inherited Work

### 20.1 What "inherited" means

An idea is **inherited** if:

- It appears in a published paper (typically D2Vformer arXiv:2409.11024 or earlier work).
- It is implemented in the official D2Vformer repository.
- It is a well-known technique used across multiple papers (e.g., RevIN).

We do not claim authorship of inherited ideas. We acknowledge them.

### 20.2 The full inherited inventory

#### 20.2.1 From the D2Vformer paper (Wang et al., 2024)

- **Date2Vec temporal embedding** — learnable harmonic embedding with k_freq = 16. The exact formulation (`[t, sin(ω_1 t + φ_1), ..., sin(ω_k t + φ_k)]`) is in the paper.
- **Cross-temporal attention** — Date2Vec similarity scaled by 1/sqrt(k+1), softmaxed, used as attention weights. Equation 3 in the paper.
- **Temporal feature extraction (TFE)** — `Linear(c_in, d_model)` applied to the input series. Standard.
- **Output aggregation** — `Y_tilde = A · T^T`. Parameter-free weighted average.
- **Feed-forward network** — `Linear(d_model, d_ff) → GELU → Linear(d_ff, c_in)`. Standard transformer block.
- **The flexible-horizon concept** — forecasting at arbitrary future time coordinates.

#### 20.2.2 From the broader literature

- **RevIN** (Reversible Instance Normalisation) — Kim et al., ICLR 2022. We use it; we did not invent it.
- **AdamW optimiser** — Loshchilov & Hutter, ICLR 2019. Standard.
- **GELU activation** — Hendrycks & Gimpel, 2016. Standard.
- **Early stopping** — Prechelt, 1998. Standard.
- **Lookback window L = 96** — convention from PatchTST (Nie et al., ICLR 2023).

### 20.3 What we did NOT inherit but is in our code

- **Our specific `pure_d2vformer.py` is a from-scratch implementation.** It is not a fork of the official repository. Every line is written by us, although the equations are inherited.
- **Our `tcd2vformer.py` and `temperature_d2vformer.py`** are from scratch.
- **Our training pipeline, data loaders, and notebooks** are from scratch.

The "from scratch" implementation is itself a contribution: the official repository has the `Linear(d_model, O)` bug, and our reconstruction removes it.

### 20.4 What we did inherit from the official repository (and why)

We did NOT copy the official `model/D2Vformer.py` because it contains the parameter-scaling bug. Our implementation is written from the paper equations and our own design.

The **only** thing we kept from the official codebase is the *conceptual* data flow — which is also in the paper. We did not copy any code.

### 20.5 Why the inheritance matters

For a BE Major Project:

- We must show we understand what we're building on.
- We must give credit to prior work.
- We must articulate what is new.

These three together demonstrate scholarship. The inheritance list above is the credit part.

---

## 21. Our Contributions (Nine)

These are the nine concrete contributions this project makes.

### 21.1 Contribution 1 — Horizon-independent D2Vformer reconstruction

**What:** `models/pure_d2vformer.py`, a from-scratch implementation that removes the `Linear(d_model, O)` projection from the official code.

**Why it matters:** the original repository's parameter count scales with O. We restore the paper's claim that the architecture is horizon-independent.

**Evidence:** `parameter_invariance_audit.csv` — 96/96 horizon-independence checks pass. Total parameters = 44,021 (c_in = 7) or 44,408 (c_in = 8).

**Defended in thesis:** yes — this is the foundation of all subsequent work.

### 21.2 Contribution 2 — Reproducibility infrastructure

**What:** `utils/reproducibility.py`, `utils/setseed.py`, and the full seed-and-protocol machinery.

**Why it matters:** without reproducibility, results cannot be verified or extended.

**Evidence:** 48 checkpoints, 288 locked evaluations, SHA-256 checksums verified.

**Defended in thesis:** yes — essential for scientific claims.

### 21.3 Contribution 3 — Phase 4 baseline comparison

**What:** systematic comparison of our `pure_d2vformer` against PatchTST, iTransformer, FEDformer, Autoformer, Informer, DLinear on all four datasets and all six horizons.

**Why it matters:** establishes where the baseline sits in the published landscape.

**Evidence:** `results/tables/master_model_comparison.md`, `Phase4_Colab_Generalization.ipynb`.

**Defended in thesis:** yes — shows we know the field.

### 21.4 Contribution 4 — Failure-mode identification (Phase 5)

**What:** identified attention over-sharpening on ETTm1 and attention entropy collapse at long horizons as the specific failure modes.

**Why it matters:** defines what Phase 6 must fix.

**Evidence:** `results/phase5/attention_entropy.csv`, `results/phase5/stress_test_report.md`.

**Defended in thesis:** yes — Phase 6 is motivated by this finding.

### 21.5 Contribution 5 — The four-mode temperature taxonomy

**What:** systematic ablation of four temperature modes: fixed, learned_global, temporal_context, query_conditioned.

**Why it matters:** forms a strict, ordered ablation (each mode subsumes the previous).

**Evidence:** §15 of this textbook, `results/phase6/PHASE6_CONCLUSION.md`.

**Defended in thesis:** yes — this is the methodological backbone.

### 21.6 Contribution 6 — TCD2Vformer architecture

**What:** the τ-MLP mechanism (in `models/tcd2vformer.py`) that conditions softmax temperature on Date2Vec embeddings.

**Why it matters:** a specific, mechanistic improvement over the original D2Vformer.

**Evidence:** the +3.12% improvement on ETTh2 at O=720, all 3/3 seeds showing > 2% gain.

**Defended in thesis:** yes — this is the headline.

### 21.7 Contribution 7 — Horizon-independence verification of TCD2Vformer

**What:** empirical verification that the τ-MLP does not re-introduce the parameter-scaling bug.

**Why it matters:** confirms the contribution does not violate the original paper's invariant.

**Evidence:** `parameter_invariance_report.md` — 96/96 pass at every (dataset, mode, horizon) combination.

**Defended in thesis:** yes — closes the loop on Contribution 1.

### 21.8 Contribution 8 — Empirical boundary condition (ETTm1 failure explanation)

**What:** mechanistic explanation of why τ-conditioning hurts on ETTm1, framed as an empirical boundary.

**Why it matters:** distinguishes a "bug" from a "limitation."

**Evidence:** `results/final_audit/ettm1_failure_analysis.md`, τ ≈ 0.18 measurement.

**Defended in thesis:** yes — shows scientific honesty.

### 21.9 Contribution 9 — End-to-end reproducibility package

**What:** a single repository that can train all 48 checkpoints and reproduce all numbers.

**Why it matters:** anyone with a Colab account can verify our claims.

**Evidence:** the `D2Vformer_colab_ready.zip` and the Colab notebooks.

**Defended in thesis:** yes — the practical reproducibility claim.

### 21.10 What the nine contributions have in common

- **Each is testable.** Every contribution has a specific file or table that proves it.
- **Each is independent.** No contribution depends on the others for its validity.
- **Each is conservative.** None overstates the evidence.

### 21.11 Why this is exactly nine

Nine is not a marketing number. It is the number of distinct, testable claims the project supports. Any more would be padding; any fewer would hide something.

---

## 22. Novelty Boundary

### 22.1 The novelty boundary in one diagram

```
| Original D2Vformer paper | ← inherited
+-----------------------------------+
| Horizon-independence fix (Phase 1) | ← ours (Contribution 1)
+-----------------------------------+
| Failure-mode analysis (Phase 5)    | ← ours (Contribution 4)
+-----------------------------------+
| Four-mode τ taxonomy (Phase 6)     | ← ours (Contribution 5)
+-----------------------------------+
| τ-MLP mechanism (Phase 6)         | ← ours (Contribution 6)
+-----------------------------------+
| ETTm1 boundary explanation         | ← ours (Contribution 8)
+-----------------------------------+
| Reproducibility package            | ← ours (Contributions 2, 9)
+-----------------------------------+
```

### 22.2 Where the boundary is most defensible

The strongest novelty is:

1. **The τ-MLP mechanism (Contribution 6).** No prior work on D2Vformer adds a learned, Date2Vec-conditioned temperature.
2. **The four-mode ablation (Contribution 5).** No prior work systematically varies temperature granularity.
3. **The horizon-independence verification (Contribution 7).** No prior work on D2Vformer audits this property.

### 22.3 Where the boundary is most fragile

The most fragile novelty:

- **Reproducibility (Contribution 2).** This is engineering, not research. But it is necessary.
- **The empirical boundary explanation (Contribution 8).** This is post-hoc analysis. But it is honest post-hoc analysis, not over-fitting to a story.

### 22.4 What we do NOT claim novelty for

- The Date2Vec formulation. (Kazemi et al. 2019, made learnable by Wang et al. 2024.)
- The cross-temporal attention equation. (Wang et al. 2024.)
- RevIN. (Kim et al. 2022.)
- Standard transformer block (FFN, GELU, residual connections, etc.).
- Lookback L = 96.
- The Adam optimiser with cosine schedule.

### 22.5 What the boundary says about Phase 7

A Phase 7 could:

- Tune τ per-dataset.
- Add a floor on τ.
- Use a different functional form (e.g., attention pooling).

None of these are necessary for the BE Major Project. Adding any of them would violate the protocol freeze and the no-Phase-7 rule.

### 22.6 Summary

The boundary is clear:

- **Inherited:** paper formulation, RevIN, optimiser, data, lookback convention.
- **Reconstructed:** pure D2Vformer implementation (Contribution 1).
- **Novel:** τ-conditioning (Contributions 5–8), failure-mode analysis (Contribution 4), reproducibility (Contributions 2, 9).

For the thesis defense, every claim should be checked against this boundary.

---

**End of Part VII.** Continue with [TEXTBOOK_PART_VIII.md](TEXTBOOK_PART_VIII.md).