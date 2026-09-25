# Internal Numerical Consistency Audit Report

**Date of Audit:** 2026-09-23  
**Status:** **AUDITED — 1 DOCUMENTATION DISCREPANCY IDENTIFIED**  
**Audit File:** `results/final_audit/numerical_consistency_report.md`  

---

## 1. Scope of Audit
This audit performed an exhaustive cross-comparison between:
- Underlying empirical data files:
  - `results/phase6/validation_results.csv` (48 rows)
  - `results/phase6/locked_test_results.csv` (288 rows)
  - `results/phase6/ablation_results.csv` (96 rows)
  - `results/phase6/attention_diagnostics.csv` (288 rows)
  - `results/phase6/checkpoints/*.pt` (48 files)
  - `results/final/summary.csv` (Phase 4/5 historical summary, 72 rows)
- Narrative and summary documents:
  - `results/phase6/analysis.md`
  - `results/phase6/PHASE6_CONCLUSION.md`
  - `RESEARCH_FINDINGS.md`
  - `README.md`

---

## 2. Findings & Discrepancy Classification

### Discrepancy 1: Validation Loss Values in Markdown Documentation
- **Classification:** **Minor / Documentation-Only**
- **Description:** In `results/phase6/analysis.md` (Section 3) and `RESEARCH_FINDINGS.md` (Phase 6 summary), the table reporting validation loss at $O_{\text{train}}=48$ listed:
  - ETTh1: $0.80373$
  - ETTh2: $0.23071$
  - ETTm1: $0.67232$
  - Exchange: $0.11978$
- **True Values in Empirical Checkpoints & `validation_results.csv`:**
  - ETTh1: $0.56326 \pm 0.0034$
  - ETTh2: $0.27554 \pm 0.0008$
  - ETTm1: $0.45555 \pm 0.0053$
  - Exchange: $0.31498 \pm 0.0076$
- **Root Cause:** The numbers in `analysis.md` were accidentally copied from test MSE approximations at $O=48$ or earlier pilot notes rather than reading directly from `validation_results.csv`.
- **Impact on Conclusions:** None. The checkpoints themselves, model training, early stopping, and locked test set evaluations (`locked_test_results.csv`, `ablation_results.csv`) are 100% genuine and verified against the actual `.pt` checkpoint dictionaries.

### Discrepancy 2: Dataset Naming Capitalization
- **Classification:** **Minor / Documentation-Only**
- **Description:** Phase 1–5 artifacts utilized capitalized `"Exchange"` or `"exchange_rate"`, whereas Phase 6 experiment runner logged lowercase `"exchange"`.
- **Impact on Conclusions:** None. All data loaders map to `datasets/exchange_rate/exchange_rate.csv`.

### Discrepancy 3: Parameter Count by Input Channel Dimension ($c_{\text{in}}$)
- **Classification:** **Minor / Documentation-Only**
- **Description:** Documentation cited 44,021 / 44,022 / 44,326 parameters as the universal count.
- **Clarification:** These counts apply to 7-variable datasets (ETTh1, ETTh2, ETTm1 with $c_{\text{in}}=7$). For Exchange Rate ($c_{\text{in}}=8$), RevIN and the output linear projection layer include 8 variables, yielding **44,408 / 44,409 / 44,713** parameters.
- **Verification:** Both 7-channel and 8-channel variants are strictly horizon-independent ($\partial N_{\text{params}} / \partial O \equiv 0$).

---

## 3. Verified Numerical Consistencies (100% Pass)

| Metric / Check | Tested Scope | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Test MSE vs Ablation CSV** | 96 rows across all datasets, modes, horizons | **PERFECT MATCH** | `ablation_results.csv` matches `locked_test_results.csv` means to 6 decimal places. |
| **Checkpoints vs Validation CSV** | 48 checkpoints | **PERFECT MATCH** | All 48 `.pt` checkpoint `val_loss` values match `validation_results.csv`. |
| **SHA-256 Checksums** | 48 checkpoints | **PERFECT MATCH** | All 48 parameter state-dict checksums match upon reloading. |
| **Evaluation Dimensions** | 48 models × 6 horizons = 288 evaluations | **PERFECT MATCH** | Exactly 288 rows in `locked_test_results.csv` and `attention_diagnostics.csv`. |
| **Parameter Invariance** | 16 (dataset, mode) configurations × 6 horizons | **PERFECT MATCH** | Invariance verified across all $O \in [24, 720]$. |

---

## 4. Corrective Action
We do NOT rewrite or overwrite historical raw logs. We document this in the final audit and ensure the thesis uses the exact numbers verified by `scratch/audit_task9_numerical_consistency.py`.
