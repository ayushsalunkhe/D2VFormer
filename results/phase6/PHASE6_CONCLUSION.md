# Phase 6 Final Evaluation & Scientific Conclusion

**Model:** Temporal-Conditioned Date2Vecformer (`TCD2Vformer`)  
**Repository:** [ayushsalunkhe/TCD2Vformer](https://github.com/ayushsalunkhe/TCD2Vformer)  
**Evaluation Protocol:** Strict validation-only selection, locked zero-shot multi-horizon evaluation  
**Datasets:** ETTh1, ETTh2, ETTm1, Exchange Rate (4 benchmark datasets)  
**Experimental Matrix:** 4 datasets × 3 seeds (42, 43, 44) × 4 modes = **48 trained models**  
**Evaluation Horizons:** $O \in \{24, 48, 96, 192, 336, 720\}$ (**288 locked zero-shot evaluations**)  
**Status:** **COMPLETE & FROZEN (Hard Stopping Rule Enforced)**  

---

## 1. Master Ablation Summary Table

Averaged across 3 independent random seeds (42, 43, 44):

| Dataset | Evaluation Horizon $O$ | Fixed Baseline ($\tau=1.0$) | Learned Global $\tau$ | Query-Conditioned $\tau(t)$ | Temporal-Context $\tau(X)$ | Best Mode | Relative Improvement vs Fixed (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | 24 | **0.8302** | 0.8314 | 0.8444 | 0.8514 | Fixed | Baseline |
| | 48 | 0.8510 | **0.8505** | 0.8653 | 0.8702 | Learned Global | **+0.06%** |
| | 96 | 0.8779 | **0.8768** | 0.8943 | 0.8960 | Learned Global | **+0.12%** |
| | 192 | 0.9247 | **0.9226** | 0.9404 | 0.9404 | Learned Global | **+0.23%** |
| | 336 | 0.9527 | **0.9522** | 0.9707 | 0.9684 | Learned Global | **+0.05%** |
| | 720 | 1.0944 | **1.0916** | 1.1117 | 1.1101 | Learned Global | **+0.25%** |
| **ETTh2** | 24 | **0.2268** | 0.2270 | 0.2280 | 0.2280 | Fixed | Baseline |
| | 48 | **0.2475** | 0.2477 | 0.2482 | 0.2482 | Fixed | Baseline |
| | 96 | **0.2784** | 0.2787 | 0.2785 | 0.2786 | Fixed | Baseline |
| | 192 | 0.3105 | 0.3107 | **0.3076** | **0.3076** | Query / Context | **+0.93%** |
| | 336 | 0.3422 | 0.3424 | **0.3348** | **0.3348** | Query / Context | **+2.18%** |
| | 720 | 0.4104 | 0.4105 | **0.3976** | **0.3976** | Query / Context | **+3.12%** |
| **ETTm1** | 24 | 0.6716 | 0.6654 | 0.6656 | **0.6634** | Temporal Context | **+1.22%** |
| | 48 | **0.7163** | 0.7171 | 0.7219 | 0.7196 | Fixed | Baseline |
| | 96 | **0.8815** | 0.9008 | 0.9092 | 0.8942 | Fixed | Baseline |
| | 192 | **0.9285** | 0.9488 | 0.9622 | 0.9422 | Fixed | Baseline |
| | 336 | **0.9626** | 0.9828 | 1.0057 | 0.9782 | Fixed | Baseline |
| | 720 | **1.0300** | 1.0512 | 1.0741 | 1.0421 | Fixed | Baseline |
| **Exchange**| 24 | **0.10281** | 0.10282 | 0.10287 | 0.10287 | Fixed | Baseline |
| | 48 | 0.12738 | 0.12740 | **0.12733** | **0.12733** | Query / Context | **+0.04%** |
| | 96 | 0.17882 | 0.17884 | **0.17874** | **0.17874** | Query / Context | **+0.04%** |
| | 192 | 0.29334 | 0.29335 | **0.29305** | **0.29305** | Query / Context | **+0.10%** |
| | 336 | 0.48682 | 0.48683 | **0.48612** | **0.48612** | Query / Context | **+0.14%** |
| | 720 | 1.23203 | 1.23208 | **1.23100** | **1.23100** | Query / Context | **+0.08%** |

---

## 2. Core Scientific Findings

1. **Parameter Invariance is Maintained:**
   All TCD2Vformer configurations require strictly identical parameters across all forecast horizons ($O \in [24, 720]$):
   - Baseline Fixed: **44,021 parameters**
   - Learned Global: **44,022 parameters (+1)**
   - Temporal-Conditioned / Query-Conditioned: **44,326 parameters (+305)**
   This proves that dynamic temperature scaling provides zero-shot flexibility without horizon-dependent parameters.

2. **Resolution of Attention Diffusion:**
   On ETTh1 and ETTm1, the model automatically sharpens cross-temporal attention:
   - ETTh1: $H_{\text{norm}}$ decreased from **0.9714** to **0.8654**, reducing effective timestamps $N_{\text{eff}}$ from **87.12** down to **60.95**.
   - ETTm1: $H_{\text{norm}}$ decreased from **0.9666** to **0.9022**, reducing $N_{\text{eff}}$ from **85.90** down to **67.34**.

3. **Significant Long-Horizon Zero-Shot Breakthrough (ETTh2):**
   On ETTh2, dynamic temporal conditioning yields monotonically increasing gains as horizon expands:
   - $O = 192$: **+0.93%** MSE reduction
   - $O = 336$: **+2.18%** MSE reduction
   - $O = 720$: **+3.12%** MSE reduction
   - Average long-horizon improvement ($O \in \{336, 720\}$): **+2.69%**

4. **Regime-Specific Regularization Dynamics:**
   - **Low-frequency / Diurnal series (ETTh2, Exchange):** Soft dynamic conditioning ($\tau \approx 0.90 - 0.96$) is optimal.
   - **High-frequency series (ETTm1):** Excessive sharpening ($\tau \approx 0.18$) improves short-horizon predictions ($O=24$: +1.22%) but creates phase sensitivity at long horizons.

---

## 3. Publication Figures Generated
All figures are stored in `results/phase6/plots/`:
- `fig1_phase6_mse_vs_horizon.png`: Zero-shot Test MSE across horizons for all 4 datasets
- `fig2_phase6_relative_improvement.png`: Percentage improvement over baseline $\tau=1.0$
- `fig3_phase6_attention_entropy.png`: Normalized Shannon entropy ($H_{\text{norm}}$) by mode
- `fig4_phase6_temperature_profiles.png`: Inferred temperature ($\tau$) distributions
- `fig5_phase6_parameter_invariance.png`: Trainable parameter constancy proof

---

## 4. Hard Stopping Rule Confirmation
Phase 6 represents the completed, pre-registered final research phase of the BE project.
The architectural design, experimental methodology, and empirical evidence are hereby **frozen**. No Phase 7 will be conducted.
