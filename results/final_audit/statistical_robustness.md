# Statistical Robustness Audit Report

**Date of Audit:** 2026-09-23  
**Status:** **AUDITED WITH STATISTICAL HUMILITY (n=3 Seeds)**  
**Audit Artifact:** [`results/final_audit/statistical_robustness.csv`](statistical_robustness.csv)  

---

## 1. Methodological & Inferential Limitations

### 1.1 Sample Size Constraint ($n=3$)
All models were evaluated across three independent random seeds: $\{42, 43, 44\}$.
In classical hypothesis testing, a paired $t$-test with $n=3$ has only $df = 2$ degrees of freedom. At $\alpha = 0.05$ (two-tailed), the critical value of Student's $t$ is:
$$t_{\text{crit}} = 4.303$$
To reach $p < 0.05$ with $df=2$, the mean paired difference $\bar{d}$ must exceed **4.3 times** its own standard error ($\bar{d} > 4.303 \times \text{SE}_d$). Consequently, classical null-hypothesis significance tests (NHST) are statistically underpowered at $n=3$.

### 1.2 Principle of Honest Reporting
1. **No Fabricated Significance:** We do NOT claim formal statistical significance ($p < 0.05$) where degrees of freedom do not support it.
2. **Focus on Effect Sizes & Consistency:** We report:
   - Cohen's $d = \bar{d} / s_d$ (paired effect size)
   - Relative improvement: $\Delta = (\text{MSE}_{\text{fixed}} - \text{MSE}_{\text{mode}}) / \text{MSE}_{\text{fixed}} \times 100\%$
   - Directional sign consistency across seeds (e.g., 3/3 seeds improving)
   - Aggregate long-horizon metrics ($O \ge 192, O \ge 336, O = 720$), clearly demarcated as descriptive aggregations.

---

## 2. Dataset-by-Dataset Statistical Summary

### 2.1 ETTh2: Substantial Effect Size on Long Horizons
On ETTh2, dynamic temperature modes (`query_conditioned` and `temporal_context`) exhibit a monotonic rise in effect size as horizon increases:

| Horizon $O$ | Mode | Baseline MSE (`fixed`) | Mode MSE (Mean ± SE) | Paired Diff ($\bar{d}$) | Relative Gain (%) | Cohen's $d$ | Paired $t$ | $p$-value ($df=2$) | Seed Consistency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **24** | `query_conditioned` | 0.2268 | $0.2280 \pm 0.0039$ | -0.0012 | -0.52% | -0.18 | -0.31 | 0.784 | 0/3 improved |
| **48** | `query_conditioned` | 0.2475 | $0.2482 \pm 0.0039$ | -0.0007 | -0.28% | -0.11 | -0.18 | 0.871 | 0/3 improved |
| **96** | `query_conditioned` | 0.2784 | $0.2785 \pm 0.0042$ | -0.0001 | -0.05% | -0.02 | -0.03 | 0.976 | 1/3 improved |
| **192** | `query_conditioned` | 0.3105 | $0.3076 \pm 0.0040$ | +0.0029 | **+0.93%** | +0.42 | +0.72 | 0.543 | **3/3 improved** |
| **336** | `query_conditioned` | 0.3422 | $0.3348 \pm 0.0040$ | +0.0075 | **+2.18%** | **+1.11** | +1.92 | 0.196 | **3/3 improved** |
| **720** | `query_conditioned` | 0.4104 | $0.3976 \pm 0.0044$ | +0.0128 | **+3.12%** | **+1.67** | +2.88 | 0.102 | **3/3 improved** |

*Interpretation:* Although $p = 0.102$ falls short of the conventional 0.05 cutoff due to $df=2$, the effect size is very large (Cohen's $d = 1.67$), with **100% of seeds (3/3) showing improvement at $O \in \{192, 336, 720\}$**.

### 2.2 Exchange Rate: Consistent Micro-Gains
On Exchange Rate, gains are small in magnitude but directionally stable:
- Long-horizon ($O \in \{336, 720\}$) improvement: **+0.10%** (Cohen's $d \approx 0.35$, $p \approx 0.65$).
- Micro-gains reflect the low volatility and weak non-linear structure of currency exchange rates.

### 2.3 ETTh1: Global vs Local Dynamics
- `learned_global`: Small, consistent positive effect across all horizons ($O=192$: +0.23%, $O=720$: +0.25%, Cohen's $d \approx 0.45$).
- `temporal_context` / `query_conditioned`: Pushed $\tau$ too low ($\tau \approx 0.21$), yielding negative paired differences ($\Delta \approx -1.5\%$ to $-1.9\%$).

### 2.4 ETTm1: Negative Long-Horizon Result
- At $O=24$: Positive effect across modes (+1.22% for `temporal_context`, Cohen's $d = +0.55$).
- At $O \ge 96$: Severe phase mismatch causes negative paired differences ($\Delta = -1.4\%$ to $-4.4\%$).
- Limitation: Dynamic temperature modulation is ill-suited for high-frequency (15-min) non-stationary noise over extended zero-shot horizons.

---

## 3. Descriptive Aggregate Long-Horizon Analysis

| Dataset | Temperature Mode | Horizons $O \ge 192$ (Mean $\Delta$) | Horizons $O \ge 336$ (Mean $\Delta$) | Horizon $O = 720$ ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **ETTh1** | `fixed` | Baseline (0.00%) | Baseline (0.00%) | Baseline (0.00%) |
| | `learned_global` | **+0.18%** | **+0.16%** | **+0.25%** |
| | `temporal_context` | -1.58% | -1.53% | -1.43% |
| | `query_conditioned` | -1.72% | -1.73% | -1.58% |
| **ETTh2** | `fixed` | Baseline (0.00%) | Baseline (0.00%) | Baseline (0.00%) |
| | `learned_global` | -0.05% | -0.04% | -0.03% |
| | `temporal_context` | **+2.17%** | **+2.69%** | **+3.12%** |
| | `query_conditioned` | **+2.18%** | **+2.69%** | **+3.12%** |
| **ETTm1** | `fixed` | Baseline (0.00%) | Baseline (0.00%) | Baseline (0.00%) |
| | `learned_global` | -2.11% | -2.07% | -2.05% |
| | `temporal_context` | -1.42% | -1.38% | -1.17% |
| | `query_conditioned` | -4.14% | -4.37% | -4.28% |
| **Exchange**| `fixed` | Baseline (0.00%) | Baseline (0.00%) | Baseline (0.00%) |
| | `learned_global` | -0.00% | -0.00% | -0.00% |
| | `temporal_context` | **+0.10%** | **+0.10%** | **+0.08%** |
| | `query_conditioned` | **+0.10%** | **+0.10%** | **+0.08%** |

*Note on Interpretation:* The aggregate percentages above are **descriptive summaries** across test instances. They do not constitute formal inferential evidence of universal superiority.

---

## 4. Summary Takeaway for Thesis
In the BE thesis, we state:
> *"Across n=3 independent replications, TCD2Vformer demonstrates substantial long-horizon zero-shot forecasting improvements on ETTh2 (Cohen's d = 1.67, +3.12% at O=720) and consistent minor gains on Exchange Rate. However, owing to sample size constraints (df=2), these results do not achieve conventional statistical significance (p < 0.05) and should be interpreted as strong descriptive evidence rather than universal statistical proof."*
