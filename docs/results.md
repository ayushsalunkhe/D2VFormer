# Results Summary

> **Document type:** Quantitative results  
> **Project:** BE Computer Engineering Major Project 2025–26

---

## Overview

Mean test MSE comparison: PureD2Vformer (τ=1.0 baseline) vs TC-D2Vformer (val-selected τ).  
All values are averaged over seeds 42, 43, 44 on locked test sets.

---

## Primary Results: ETTh1 and Exchange (Phases 2–3)

| Dataset | Horizon O | PureD2V (τ=1.0) | TC-D2V (val-sel) | Δ |
|---------|-----------|-----------------|-------------------|---|
| ETTh1 | 24 | 0.8258 | **0.8227** | +0.38% |
| ETTh1 | 48 | 0.8459 | **0.8438** | +0.25% |
| ETTh1 | 96 | 0.8731 | **0.8713** | +0.21% |
| ETTh1 | 192 | 0.9173 | **0.9103** | +0.76% |
| ETTh1 | 336 | 0.9431 | **0.9346** | +0.90% |
| ETTh1 | 720 | 1.0833 | **1.0690** | +1.32% |
| Exchange | 24 | 0.1052 | **0.1046** | +0.57% |
| Exchange | 48 | 0.1295 | **0.1290** | +0.39% |
| Exchange | 96 | 0.1817 | **0.1805** | +0.66% |
| Exchange | 192 | 0.2953 | **0.2923** | +1.02% |
| Exchange | 336 | 0.4877 | **0.4810** | +1.37% |
| Exchange | 720 | 1.2340 | **1.2175** | +1.34% |

---

## External Generalization: ETTh2 and ETTm1 (Phase 4)

| Dataset | Horizon O | PureD2V (τ=1.0) | TC-D2V (val-sel) | Δ |
|---------|-----------|-----------------|-------------------|---|
| ETTh2 | 24 | 0.2274 | **0.2251** | +1.01% |
| ETTh2 | 48 | 0.2480 | **0.2455** | +1.04% |
| ETTh2 | 96 | 0.2795 | **0.2764** | +1.11% |
| ETTh2 | 192 | 0.3113 | **0.3081** | +1.03% |
| ETTh2 | 336 | 0.3422 | **0.3391** | +0.91% |
| ETTh2 | 720 | 0.4101 | **0.4071** | +0.71% |
| ETTm1 | 24 | 0.6780 | **0.6523** | +3.79% |
| ETTm1 | 48 | 0.7255 | **0.7104** | +2.08% |
| ETTm1 | 96 | 0.8962 | 0.8972 | −0.12% |
| ETTm1 | 192 | 0.9413 | 0.9444 | −0.33% |
| ETTm1 | 336 | 0.9755 | 0.9795 | −0.42% |
| ETTm1 | 720 | 1.0415 | 1.0506 | −0.88% |

---

## Dataset-Level Summary

| Dataset | Mean Δ (all O) | Long-Hz Δ (O≥192) | Status |
|---------|---------------|---------------------|--------|
| ETTh1 | +0.67% | +1.13% | ✅ All 6 horizons improved |
| Exchange | +1.17% | +1.35% | ✅ All 6 horizons improved |
| ETTh2 (unseen) | +0.95% | +0.88% | ✅ All 6 horizons improved |
| ETTm1 (unseen) | +0.44% | −0.54% | ⚠️ Short-hz gain, long-hz mixed |

---

## Selected Temperatures

| Dataset | Seed 42 | Seed 43 | Seed 44 |
|---------|---------|---------|---------|
| ETTh1 | τ=4.0 | τ=2.0 | τ=4.0 |
| Exchange | τ=0.5 | τ=4.0 | τ=4.0 |
| ETTh2 | τ=1.0 | τ=2.0 | τ=4.0 |
| ETTm1 | τ=0.5 | τ=0.5 | τ=2.0 |

**Observation:** τ=4.0 is selected most frequently (7/12) but τ=0.5 is selected for all 3 ETTm1 seeds. This confirms dataset-dependent temperature behavior.

---

## Baseline Comparisons

| Dataset | Horizon | Persistence | DLinear (retrained) | PureD2V | TC-D2V |
|---------|---------|-------------|---------------------|---------|--------|
| ETTh1 | 24 | 1.5244 | 0.3470 | 0.8258 | **0.8227** |
| ETTh1 | 720 | 1.9035 | 0.6576 | 1.0833 | **1.0690** |
| Exchange | 720 | 1.0476 | 1.5701 | 1.2340 | **1.2175** |

**Note:** TC-D2Vformer significantly outperforms Persistence and Repo-D2Vformer, and DLinear on Exchange at O=720. DLinear (retrained per O) remains the strongest baseline overall.

---

## Attention Diagnostics

| Metric | ETTh1 | Exchange |
|--------|-------|----------|
| H_norm | 0.971 | 0.922 |
| N_eff | 87.6 | 79.7 |
| N_eff/L | 0.913 | 0.830 |
| max(A) | 0.027 | 0.056 |
| Uniform control ΔMSE | +6.0% (worse) | −2.0% (better) |
| Shuffled control ΔMSE | +7.2% (worse) | −0.2% (negligible) |

---

## Interpretation

**Outcome B: Partial/Dataset-Dependent Generalization**

- TC-D2Vformer with validation-selected τ consistently outperforms the τ=1.0 baseline.
- Improvements are **dataset-dependent**: ETTh2 sees consistent gains; ETTm1 shows mixed results.
- The hypothesis that τ=4.0 is universally optimal is **rejected** — Exchange seed 42 selects τ=0.5.
- Statistical significance is limited (n=3 seeds) — effect sizes are reported, not p-values.
