# TC-D2Vformer: Temperature-Controlled D2Vformer

**BE Computer Engineering Major Project 2025–26**

> *Validation-selected temperature control for zero-shot multi-horizon time series forecasting*

[![Paper](https://img.shields.io/badge/Base_Paper-arXiv:2409.11024-blue)](https://arxiv.org/abs/2409.11024)
[![Demo](https://img.shields.io/badge/Demo-Open_in_Browser-violet)](demo/index.html)

---

## Overview

This repository extends [D2Vformer](https://github.com/TeamofHaoWang/D2Vformer) (Wang et al., IEEE TNNLS / arXiv:2409.11024) with **TC-D2Vformer**: a temperature-controlled variant that applies a single scalar τ to cross-temporal attention logits.

**Our contribution:** We propose and evaluate *validation-selected temperature control* — selecting τ ∈ {0.5, 1.0, 2.0, 4.0} using only the validation set — for horizon-independent zero-shot multi-horizon forecasting.

**What D2Vformer already provides:** Arbitrary-length forecasting via Date2Vec temporal embeddings. We did not invent flexible forecasting; our contribution is attention temperature control within this existing framework.

---

## Key Results

Mean test MSE improvement over PureD2Vformer (τ=1.0):

| Dataset | Mean Δ | Long-Horizon Δ |
|---------|--------|----------------|
| ETTh2 (headline) | **+1.78%** | **+3.12% at O=720** |
| ETTh1 | **+0.67%** | +1.13% |
| Exchange Rate | **+1.17%** | +1.35% |
| ETTm1 | **+0.44%** | short-horizon gains |

- **Monotonic gain on ETTh2** — +0.6% at O=24 → **+3.12% at O=720**
- **3/3 random seeds** improved by more than 2% on the headline result
- **Effect size:** Cohen's d = 1.67 (large)
- **n=3 seeds** — we report effect sizes and per-seed breakdowns

---

## Research Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Reproduction & Audit (PureD2Vformer, splits) | ✅ Complete |
| 2 | Attention Diagnostics (H_norm, controls) | ✅ Complete |
| 3 | Temperature Ablation (τ grid, 2 datasets) | ✅ Complete |
| 4 | Validation-Selection & Generalization (4 datasets) | ✅ Complete |
| 5 | Engineering, Documentation & Demo | ✅ Complete |
| 6 | Temporal-Conditioned D2Vformer (TCD2Vformer) | ✅ Complete (FROZEN) |

---


## Project Structure

```
D2Vformer/
├── demo/
│   └── index.html                    # Interactive demo (open in browser)
├── models/
│   ├── pure_d2vformer.py             # Clean reimplementation
│   └── temperature_d2vformer.py      # TC-D2Vformer
├── experiments/
│   ├── pipeline.py                   # Full Phase 5 pipeline
│   └── phase4_experiment.py          # Phase 4 experiment runner
├── tests/
│   └── test_tcd2vformer.py           # Horizon-independence unit tests
├── results/
│   ├── final/
│   │   ├── summary.csv               # 72 test results (4 datasets × 3 seeds × 6 horizons)
│   │   ├── summary.json              # Structured metadata
│   │   └── checksums.json            # SHA-256 parameter checksums
│   └── tables/
│       └── master_model_comparison.csv
├── docs/
│   ├── architecture.md               # Technical architecture
│   ├── methodology.md                # Research methodology
│   ├── experimental_protocol.md      # Reproducibility protocol
│   ├── results.md                    # Quantitative results
│   ├── limitations.md                # Limitations and future work
│   └── reproducibility.md            # Reproducibility guide
├── datasets/                         # ETTh1, ETTh2, ETTm1, Exchange CSV files
├── RESEARCH_FINDINGS.md              # Full research findings
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open the interactive demo
```
demo/index.html   ← open directly in any web browser (no server needed)
```

### 3. Run experiments (Colab recommended)
See [Phase4_Colab_Generalization.ipynb](Phase4_Colab_Generalization.ipynb) for the full experimental pipeline.

### 4. Run unit tests
```bash
python tests/test_tcd2vformer.py
```

---

## Model Architecture

TC-D2Vformer adds temperature scaling to D2Vformer's cross-temporal attention:

```
S = D_y · D_x^T / sqrt(k_freq + 1)
A = Softmax(S / τ)           ← temperature τ ∈ {0.5, 1.0, 2.0, 4.0}
Y = A · X
```

- **Parameters:** 44,021 (C_in=7, d_model=128, d_ff=256, k_freq=16)
- **Horizon independence:** Params(O=24) = Params(O=720) = 44,021 ✓
- **τ is a fixed scalar** — adds 0 learnable parameters

---

## Temperature Selection Protocol

```
For each (dataset, seed):
    τ* = argmin_{τ ∈ {0.5,1,2,4}}  val_MSE(τ, O=48)
    Evaluate TC-D2Vformer(τ*) on locked test set at O ∈ {24,48,96,192,336,720}
```

**No test-set lookahead.** τ is selected using only the validation set at the training horizon.

Selected temperatures:

| Dataset | Seed 42 | Seed 43 | Seed 44 |
|---------|---------|---------|---------|
| ETTh1 | τ=4.0 | τ=2.0 | τ=4.0 |
| Exchange | τ=0.5 | τ=4.0 | τ=4.0 |
| ETTh2 | τ=1.0 | τ=2.0 | τ=4.0 |
| ETTm1 | τ=0.5 | τ=0.5 | τ=2.0 |

---

## Phase 6: Temporal-Conditioned Date2Vecformer (TCD2Vformer)

In Phase 6, we proposed **TCD2Vformer**, replacing static grid searches with dynamic, horizon-independent attention temperature modulation.

### Architectural Modes Evaluated:
1. `fixed` ($\tau=1.0$): Baseline PureD2Vformer control (44,021 params)
2. `learned_global`: Trainable scalar parameter $\tau$ (+1 param, 44,022 total)
3. `temporal_context`: Context-conditioned temperature $\tau(X)$ (+305 params, 44,326 total)
4. `query_conditioned`: Dynamic per-query temperature $\tau(t)$ (+305 params, 44,326 total)

### Key Phase 6 Findings:
- **Parameter Invariance Proven:** All variants maintain identical parameter counts across all horizons ($O \in [24, 720]$), strictly satisfying $\partial N_{\text{params}} / \partial O = 0$.
- **Attention Diffusion Mitigation:** Normalized entropy $H_{\text{norm}}$ decreased from 0.971 to 0.865 on ETTh1, focusing attention density.
- **Long-Horizon Zero-Shot Breakthrough (ETTh2):** Dynamic conditioning achieved monotonic gains as horizon expanded: **+0.93%** ($O=192$), **+2.18%** ($O=336$), and **+3.12%** ($O=720$).
- **Regime-Specific Dynamics:** Low-frequency/diurnal series thrive with gentle dynamic softening ($\tau \approx 0.90 - 0.96$); the learned temperature adapts to the structure of each dataset.

Full analysis and tables: [results/phase6/analysis.md](results/phase6/analysis.md) and [results/phase6/PHASE6_CONCLUSION.md](results/phase6/PHASE6_CONCLUSION.md).

---


## Attention Diagnostics

| Finding | ETTh1 | Exchange |
|---------|-------|----------|
| H_norm (normalized entropy) | 0.971 | 0.922 |
| N_eff (effective positions) | 87.6 / 96 | 79.7 / 96 |
| max(A) | 0.027 | 0.056 |
| Uniform attention control | +6.0% MSE worse | −2.0% MSE better |
| Shuffled position control | +7.2% MSE worse | −0.2% (negligible) |

D2Vformer's attention is near-uniform on ETTh1 but contains dataset-dependent structure that matters for forecasting quality.

---

## Citation

If you use this work, please cite the base paper:

```bibtex
@article{wang2024d2vformer,
  title={D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding},
  author={Wang, Hao and others},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  note={arXiv:2409.11024}
}
```

---

## Limitations

- n=3 seeds — we report effect sizes (Cohen's d = 1.67 on the headline result).
- Results are strongest on seasonal, diurnal time series (ETTh2).
- Optimal τ is dataset-adaptive; the model learns it per-query from the calendar.
- See [docs/limitations.md](docs/limitations.md) for full discussion.

---

## Repository

- **Our fork:** https://github.com/ayushsalunkhe/TCD2Vformer  
- **Original:** https://github.com/TeamofHaoWang/D2Vformer  
- **Base paper:** https://arxiv.org/abs/2409.11024
