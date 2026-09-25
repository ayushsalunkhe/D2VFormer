# Master Benchmark Comparison: D2Vformer Horizon Generalization across 4 Datasets

**Evaluation Metrics:** Test Mean Squared Error (MSE) averaged across Seeds 42, 43, 44 on locked test sets.

| Dataset | Horizon $O$ | Persistence | DLinear (Retrained per $O$) | Repo-D2Vformer | PureD2Vformer (Baseline $\tau=1.0$) | PureD2Vformer (Val-Selected $\tau$) | Gain over Baseline $\tau=1.0$ |
|---|---|---|---|---|---|---|---|
| **ETTh1** | 24 | 1.5244 | 0.3470 | 1.0250 | 0.8258 | **0.8227** | +0.38% |
| **ETTh1** | 48 | 1.5788 | 0.3956 | 0.9584 | 0.8459 | **0.8438** | +0.25% |
| **ETTh1** | 96 | 1.6498 | 0.4489 | 0.9878 | 0.8731 | **0.8713** | +0.21% |
| **ETTh1** | 192 | 1.7188 | 0.5007 | 1.1023 | 0.9173 | **0.9103** | +0.76% |
| **ETTh1** | 336 | 1.7559 | 0.5407 | 1.1871 | 0.9431 | **0.9346** | +0.90% |
| **ETTh1** | 720 | 1.9035 | 0.6576 | 1.1934 | 1.0833 | **1.0690** | **+1.32%** |
| **Exchange**| 24 | 0.0318 | 0.0326 | 0.0752 | 0.1052 | **0.1046** | +0.57% |
| **Exchange**| 48 | 0.0548 | 0.0553 | 0.0901 | 0.1295 | **0.1290** | +0.39% |
| **Exchange**| 96 | 0.1010 | 0.1027 | 0.1747 | 0.1817 | **0.1805** | +0.66% |
| **Exchange**| 192 | 0.2013 | 0.2180 | 0.2592 | 0.2953 | **0.2923** | +1.02% |
| **Exchange**| 336 | 0.3763 | 0.4397 | 0.4804 | 0.4877 | **0.4810** | **+1.37%** |
| **Exchange**| 720 | 1.0476 | 1.5701 | 1.2075 | 1.2340 | **1.2175** | **+1.34%** |
| **ETTh2** (Unseen) | 24 | 0.2528 | 0.1543 | — | 0.2274 | **0.2251** | +1.01% |
| **ETTh2** (Unseen) | 48 | 0.2920 | 0.1849 | — | 0.2480 | **0.2455** | +1.04% |
| **ETTh2** (Unseen) | 96 | 0.3385 | 0.2271 | — | 0.2795 | **0.2764** | +1.11% |
| **ETTh2** (Unseen) | 192 | 0.3984 | 0.2662 | — | 0.3113 | **0.3081** | +1.03% |
| **ETTh2** (Unseen) | 336 | 0.4414 | 0.3060 | — | 0.3422 | **0.3391** | +0.91% |
| **ETTh2** (Unseen) | 720 | 0.5243 | 0.4028 | — | 0.4101 | **0.4071** | +0.71% |
| **ETTm1** (Unseen) | 24 | 0.8195 | 0.2692 | — | 0.6780 | **0.6523** | **+3.79%** |
| **ETTm1** (Unseen) | 48 | 1.4797 | 0.3314 | — | 0.7255 | **0.7104** | **+2.08%** |
| **ETTm1** (Unseen) | 96 | 1.5184 | 0.3636 | — | 0.8962 | 0.8972 | -0.12% |
| **ETTm1** (Unseen) | 192 | 1.5740 | 0.4306 | — | 0.9413 | 0.9444 | -0.33% |
| **ETTm1** (Unseen) | 336 | 1.6338 | 0.4979 | — | 0.9755 | 0.9795 | -0.42% |
| **ETTm1** (Unseen) | 720 | 1.7038 | 0.5607 | — | 1.0415 | 1.0506 | -0.88% |

---

## Key Synthesis Insights
1. **Four-Dataset Generality:** Overall mean test MSE improves across **all four datasets** when temperature is selected via validation loss.
2. **ETTh2 Confirms ETTh1:** On the independent hourly transformer station (ETTh2), validation selection beats $\tau=1.0$ across **100% of horizons (6/6)**.
3. **Temporal Granularity Dissection:** In 15-minute data (ETTm1), sharper attention is selected, driving substantial gains for short horizons ($+3.79\%$, $+2.08\%$) while showing limits for extreme multi-day extrapolation.
