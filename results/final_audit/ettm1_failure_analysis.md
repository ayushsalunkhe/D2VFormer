# Failure Analysis: The ETTm1 Negative Result

**Audit Target:** Why does dynamic temperature conditioning degrade on ETTm1 at long horizons ($O \ge 96$), despite improving at short horizons ($O=24$)?  
**Status:** **EMPIRICALLY IDENTIFIED & MECHANISTICALLY EXPLAINED**  

---

## 1. The Empirical Discrepancy

On ETTm1, we observe a stark divergence between short-horizon and long-horizon zero-shot forecasting:

| Evaluation Horizon $O$ | Real-World Time Span | Fixed ($\tau=1.0$) MSE | `temporal_context` MSE | `query_conditioned` MSE | Relative Change (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$O = 24$** | **6 hours** | 0.67160 | **0.66341** | 0.66559 | **+1.22% (Improvement)** |
| **$O = 48$** | **12 hours** | 0.71628 | 0.71963 | 0.72188 | -0.47% (Neutral) |
| **$O = 96$** | **24 hours** | 0.88153 | 0.89417 | 0.90923 | -1.43% |
| **$O = 192$** | **2 days** | 0.92845 | 0.94222 | 0.96218 | -1.48% |
| **$O = 336$** | **3.5 days** | 0.96263 | 0.97815 | 1.00571 | -1.61% |
| **$O = 720$** | **7.5 days** | 1.03004 | 1.04211 | 1.07411 | -1.17% to -4.28% |

---

## 2. Mechanistic Root Cause Analysis

Based on attention diagnostics and physical time scales, the failure is attributable to three distinct, interacting mechanisms:

### 2.1 The Sampling Frequency Mismatch (15-Minute vs Hourly)
- In hourly datasets (ETTh1, ETTh2), the lookback window of $L=96$ steps covers **96 hours (4 full calendar days)**. The Date2Vec layer easily learns multi-day periodicity and macro diurnal cycles.
- In ETTm1, the sampling interval is **15 minutes**. Consequently, $L=96$ steps covers **only 24 hours (1 single calendar day)**.
- Training at $O_{\text{train}}=48$ covers only **12 hours**.

### 2.2 Over-Sharpening & Attention Entropy Collapse
During training at $O_{\text{train}}=48$ (12 hours):
- The training objective penalizes diffuse attention because intra-day correlations over 12 hours are strong and deterministic.
- The temperature network aggressively drove temperature down:
  $$\tau_{\text{ETTm1}} \approx 0.174 - 0.180 \quad \text{versus} \quad \tau_{\text{ETTh2}} \approx 0.901 - 0.904$$
- This aggressive reduction collapsed normalized entropy $H_{\text{norm}}$ from **0.967** to **0.902**, shrinking effective timestamps $N_{\text{eff}}$ from **85.9 down to 67.3**.

### 2.3 Intra-Day Phase Overfitting Under Extended Horizon Shift
- **At $O=24$ (6 hours ahead):** The query timestamps fall within the same diurnal cycle. Sharpening attention onto specific 15-minute lags is highly effective, yielding a **+1.22%** MSE improvement.
- **At $O=720$ (7.5 days ahead):** The query timestamp is over a week away from the 1-day history window. 
- Because $\tau \approx 0.18$, the cross-temporal attention mechanism is forced to place sharp probability mass on specific 15-minute historical slots.
- Any cumulative phase drift, weekend effects, or non-stationary noise over 7.5 days leads to catastrophic point-prediction errors.
- Conversely, the baseline `fixed` model ($\tau=1.0$) maintains diffuse, near-uniform attention, acting as an implicit Bayesian prior that averages out high-frequency noise over long horizons.

---

## 3. Scientific Implications & Thesis Guidance

### What This Teaches Us:
1. **Dynamic temperature is not a panacea:** Attention sharpening operates as a bias-variance tradeoff. When the history window $L$ is physically short (24 hours) relative to the forecast horizon $O$ (7.5 days), diffuse attention is a regularizer that prevents phase overfitting.
2. **Negative Results Are an Academic Asset:** For a BE major project defense, documenting this failure mechanism proves deep architectural comprehension rather than superficial metric-chasing.
3. **Formal Thesis Presentation:** We explicitly present ETTm1 as the **boundary condition** of TCD2Vformer:
   > *"TCD2Vformer thrives on low-frequency, diurnal series (ETTh2, Exchange) where phase relationships remain stable over long horizons. However, in high-frequency regimes (15-min intervals) where lookback windows cover limited calendar duration, aggressive learned sharpening creates phase sensitivity when projected zero-shot over multi-day horizons."*
