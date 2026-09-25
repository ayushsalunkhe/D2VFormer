# Phase 4A: External Generalization — Dataset Selection

**Study:** External Validation and Robustness of Temperature-Controlled Cross-Temporal Attention  
**Date:** 2026-09-22  
**Purpose:** Pre-experiment documentation of unseen evaluation datasets to prevent post-hoc dataset selection bias.

---

## 1. Selected Datasets

To evaluate whether temperature scaling and validation-based temperature selection generalize beyond the initial discovery datasets (ETTh1 and Exchange Rate), we select **two external datasets** from the standard time-series benchmarking suite used in the original D2Vformer paper:

1. **ETTh2** (Electricity Transformer Temperature — Station 2, Hourly)
2. **ETTm1** (Electricity Transformer Temperature — Station 1, 15-Minute)

Both datasets were strictly excluded from the hypothesis generation and temperature ablation in Phases 2 and 3.

---

## 2. Dataset Specifications

| Property | Dataset 1: ETTh2 | Dataset 2: ETTm1 |
|---|---|---|
| **Domain** | Power Grid Electricity Transformers | Power Grid Electricity Transformers |
| **Physical Source** | Transformer Station 2 (independent location) | Transformer Station 1 (high-frequency) |
| **Number of Variables ($C$)** | 7 (`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`) | 7 (`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`) |
| **Sampling Frequency** | 1 hour | 15 minutes |
| **Total Timestamps** | 17,420 | 69,680 |
| **Temporal Range** | July 2016 – July 2018 | July 2016 – July 2018 |
| **Train / Val / Test Split** | 60% / 20% / 20% (chronological) | 60% / 20% / 20% (chronological) |
| **Train Timestamps** | 10,452 | 41,808 |
| **Val Timestamps** | 3,484 | 13,936 |
| **Test Timestamps** | 3,484 | 13,936 |
| **Lookback Window ($L$)** | 96 steps (4 days) | 96 steps (1 day) |
| **Training Horizon ($O_{\text{train}}$)** | 48 steps | 48 steps |
| **Evaluation Horizons ($O$)** | 24, 48, 96, 192, 336, 720 steps | 24, 48, 96, 192, 336, 720 steps |

---

## 3. Preprocessing & Normalization Protocol

1. **Feature Normalization:**
   - Standard z-score normalization: $x_{\text{norm}} = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$.
   - Mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are computed **strictly on the 60% training partition**. Zero test or validation information is used.
2. **Calendar Time Markers:**
   - 4 normalized cyclic calendar features: $[h, w, d, m] \in [-0.5, 0.5]$ (Hour of day, Day of week, Day of month, Month of year).
   - Extracted identically across train, val, and test splits without lookahead.
3. **Instance Normalization:**
   - Reversible Instance Normalization (RevIN) with learned affine parameters is applied within `PureD2Vformer` to handle non-stationary distribution shifts.

---

## 4. Scientific Rationale for Selection

1. **True External Hold-Outs:**
   - The temperature hypothesis was derived entirely on ETTh1 (where attention was diffuse but meaningful) and Exchange Rate (where attention was noisy and worse than uniform).
   - Neither ETTh2 nor ETTm1 was observed during temperature development.
2. **Structural & Granularity Diversity:**
   - **ETTh2:** Evaluates whether cross-temporal attention behaves identically across two distinct physical transformer stations at the same sampling rate (hourly). ETTh2 is known to exhibit sharper load spikes and distinct oil temperature dynamics compared to ETTh1.
   - **ETTm1:** Evaluates whether temporal granularity (15-minute vs. 1-hour) shifts the attention entropy profile or temperature sensitivity.
3. **Zero Architectural Modification:**
   - Both datasets have 7 multivariate dimensions and 4 temporal markers, perfectly matching `PureD2Vformer`'s input interface ($C=7, M=4$).
   - Completely preserves the zero-shot horizon-independent formulation (SHA-256 checksum validated).
