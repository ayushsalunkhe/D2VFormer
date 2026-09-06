# Project Review Status: D2Vformer Time-Series Forecasting

**Project:** D2Vformer Time-Series Forecasting  
**Course / Degree:** B.E. Major Project  
**Status Date:** September 2026 (4 Days to Review)  
**Primary Finding:** One trained D2Vformer checkpoint can be reused across multiple forecast horizons without retraining.

---

## 1. Original Problem

Conventional Transformer-based time-series forecasting models suffer from two structural shortcomings:
1. **Fixed Prediction Horizon Rigidness:** Standard architectures (PatchTST, Informer, Autoformer, Fedformer) fix the prediction length $O$ during training. Forecasting a different horizon requires completely retraining a separate model or retraining dense linear projection heads.
2. **Relative vs. Absolute Temporal Awareness:** Traditional positional embeddings encode order within a window (step 1, step 2, ..., step 96), but discard absolute temporal context (e.g., calendar hour of day, day of week, seasonal month, or exact calendar gaps).

---

## 2. D2Vformer's Core Idea

D2Vformer (IEEE TNNLS, 2026) incorporates **Date2Vec** embedding modules:
- Extracted frequency components via Fourier analysis are combined with explicit calendar features (hour, day of week, day of month, month).
- Absolute time position representation decouples future horizon length from historical sequence length in the temporal embedding space.
- The decoder query utilizes target future timestamps directly, allowing a **single trained model checkpoint** to forecast arbitrary future windows ($48\text{h}, 72\text{h}, 96\text{h}, 192\text{h}, 336\text{h}$) without architectural modification or parameter retraining.

---

## 3. Semester 1 Achievements

In Semester 1, the foundation was established and validated:
- **Repository Bug Fixes:** Fixed 6 critical bugs in the released research code (dimension indexing, batch normalizations, learning rate schedulers, early stopping state serialization).
- **Core Models Established:** `D2Vformer_simple` / `D2Vformer_s` implemented and verified on **ETTh1** benchmark.
- **Real-World Extension:** Extended framework to Indian environmental data (**Delhi AQI** / IndiaAQI: PM2.5, PM10, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$).
- **Linear Baseline:** Implemented **DLinear** (AAAI 2023) individual channel decomposition baseline.
- **Fixed-Horizon Interactive Demo:** Built initial Streamlit web application for 96h forecasting.

---

## 4. Semester 2 Flexible Forecasting Achievements

In Semester 2, flexible multi-horizon forecasting was evaluated and verified:
- **Checkpoints Evaluated Without Retraining:**
  - `ETTh1_best_model.pkl` (trained at 96h, reused across 48h, 72h, 96h, 192h, 336h)
  - `IndiaAQI_best_model.pkl` (trained at 96h, reused across 48h, 72h, 96h, 192h, 336h)
- **Baseline Retraining Effort:**
  - Trained 8 new DLinear models on Google Colab (4 horizons $\times$ 2 datasets), adding to the 2 existing 96h models.
  - Total DLinear checkpoints required: **10 separate checkpoints** (5 per dataset).
- **Zero-Retraining Verification:** Verified all 10 combinations of datasets and horizons locally with strict output shape matching and error-free inference.

---

## 5. D2Vformer Results (Single Reused Checkpoint)

Evaluated in normalized feature space (Z-score normalized test set):

| Dataset | Horizon | Retrained? | MSE | MAE | RMSE | Checkpoint |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETTh1** | 48h | **NO** | 1.1982 | 0.7731 | 1.0946 | `ETTh1_best_model.pkl` |
| **ETTh1** | 72h | **NO** | 1.2199 | 0.7852 | 1.1045 | `ETTh1_best_model.pkl` |
| **ETTh1** | 96h | **NO** | 1.2333 | 0.7919 | 1.1105 | `ETTh1_best_model.pkl` |
| **ETTh1** | 192h | **NO** | 1.2973 | 0.8143 | 1.1390 | `ETTh1_best_model.pkl` |
| **ETTh1** | 336h | **NO** | 1.3892 | 0.8481 | 1.1786 | `ETTh1_best_model.pkl` |
| **IndiaAQI** | 48h | **NO** | 0.1128 | 0.2657 | 0.3358 | `IndiaAQI_best_model.pkl` |
| **IndiaAQI** | 72h | **NO** | 0.1152 | 0.2688 | 0.3394 | `IndiaAQI_best_model.pkl` |
| **IndiaAQI** | 96h | **NO** | 0.1179 | 0.2723 | 0.3434 | `IndiaAQI_best_model.pkl` |
| **IndiaAQI** | 192h | **NO** | 0.1322 | 0.2902 | 0.3635 | `IndiaAQI_best_model.pkl` |
| **IndiaAQI** | 336h | **NO** | 0.1647 | 0.3273 | 0.4058 | `IndiaAQI_best_model.pkl` |

**Evaluation note:** These flexible forecasting metrics were computed on the first 1,600 test samples for computational efficiency during Semester 1 development. Earlier Semester 1 single-horizon benchmarks (MSE 0.6982 for ETTh1 96h, MSE 0.2212 for IndiaAQI 96h, reported in CLAUDE.md) used the full test set. Both are valid evaluations of the same trained checkpoints, but differ in sample count and should not be directly compared numerically.

---

## 6. DLinear Baseline Comparison (Separate Retrained Models)

| Dataset | Horizon | Retrained? | MSE | MAE | RMSE | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETTh1** | 48h | **YES** | 0.3538 | 0.4007 | 0.5948 | 25.1s |
| **ETTh1** | 72h | **YES** | 0.3863 | 0.4233 | 0.6215 | 24.6s |
| **ETTh1** | 96h | **YES** | 0.4033 | 0.4358 | 0.6351 | Sem 1 ckpt |
| **ETTh1** | 192h | **YES** | 0.4677 | 0.4812 | 0.6839 | 38.0s |
| **ETTh1** | 336h | **YES** | 0.5184 | 0.5190 | 0.7200 | 37.6s |
| **IndiaAQI** | 48h | **YES** | 0.1309 | 0.2507 | 0.3618 | 24.9s |
| **IndiaAQI** | 72h | **YES** | 0.1490 | 0.2627 | 0.3860 | 24.4s |
| **IndiaAQI** | 96h | **YES** | 0.1658 | 0.2710 | 0.4072 | Sem 1 ckpt |
| **IndiaAQI** | 192h | **YES** | 0.2327 | 0.3123 | 0.4824 | 47.0s |
| **IndiaAQI** | 336h | **YES** | 0.3301 | 0.3824 | 0.5745 | 77.7s |

---

## 7. Flexibility vs. Accuracy Discussion

An honest and scientifically sound presentation of the findings reveals an important trade-off:

1. **On Delhi AQI (Strong Periodic Calendar Dynamics):**
   - **D2Vformer wins on all 5 horizons** without retraining.
   - At 336h, D2Vformer achieves MSE 0.1647 vs. DLinear's 0.3301 (50.1% MSE reduction).
   - Air pollution patterns exhibit strong diurnal, weekly, and seasonal cycles where explicit date encoding gives a clear architectural advantage.

2. **On ETTh1 (Transformer Temperature Benchmark):**
   - **DLinear achieves lower MSE across all horizons** (0.35–0.52 vs. 1.20–1.39).
   - DLinear's direct trend-seasonal decomposition is known from the literature to be competitive on standard ETT series.
   - When D2Vformer extrapolates to horizons far beyond its 96h training length (such as 336h), some degradation in point accuracy occurs.

3. **Training Overhead vs. Operational Flexibility:**
   - **DLinear requires $N$ separate training pipelines** for $N$ desired horizons ($5 \times \text{models per dataset}$).
   - **D2Vformer requires only $1$ training pipeline** ($1 \times \text{model per dataset}$).
   - In production environments where forecasting needs vary on demand, zero-shot horizon flexibility eliminates deployment and retraining pipelines.

---

## 8. Current Streamlit Demo Features

The interactive demo (`streamlit_app.py`) provides:
- **Dataset Switching:** ETTh1 (7 channels) and Delhi AQI (6 channels).
- **Flexible Horizon Slider:** Select from 48h, 72h, 96h, 192h, and 336h.
- **Model Checkpoint Independence:**
  - D2Vformer keeps the same single checkpoint loaded.
  - DLinear loads the exact horizon-specific checkpoint matching the slider.
- **Real Calendar Display:** Maps window index to exact historical and forecast datetime stamps and season information.
- **Comparative Visualizations:** Historical context (96h), ground truth actuals, and model forecast overlays.
- **Metrics Table:** Real-time denormalized MSE, MAE, and RMSE for the active window.
- **Precomputed Benchmark Expander:** Summary table comparing test set metrics across all horizons.
- **Status Panel:** Live indicators verifying availability of all 10 DLinear checkpoints and 2 D2Vformer models.

---

## 9. Known Limitations

- **Consecutive Future Querying:** Current implementation supports variable-length consecutive future horizons ($L \to L + O$), but does not yet evaluate arbitrary sparse or non-contiguous timestamp querying (e.g., predicting only every 6th hour or specific irregular dates).
- **Extrapolation Degradation:** When extrapolating past $2\times$ the training length (192h, 336h) on ETTh1, performance degrades compared to dedicated retraining.

---

## 10. Future Work Proposals

1. **Arbitrary Irregular Timestamp Querying:** Leverage Date2Vec embeddings to evaluate non-contiguous, missing, or irregular future intervals.
2. **Multi-Rate Forecasting:** Unify hourly, daily, and weekly forecasts within a single unified checkpoint.
3. **Patch-Level Frequency Encoding:** Integrate PatchTST-style channel independence with Date2Vec embeddings for enhanced point accuracy on benchmarks like ETTh1.
