# Phase 3A — Methodology Audit: Temperature Selection Experiment

**Audited file:** `experiments/temperature_experiment.py`  
**Data split code:** `utils/data.py`  
**Date of audit:** 2026-09-22  

---

## Audit Questions and Findings

### 1. Which data split was used to TRAIN the models?

**CLEAN.** Training used the first 60% of each dataset (`data_raw[:n_train]`).  
Source: `utils/data.py` lines 71–76.  
All parameter updates (optimizer steps) were performed on this split only.

### 2. Which split was used to SELECT the best checkpoint (early stopping)?

**CLEAN.** The validation split (rows n_train : n_train+n_val, i.e. 60–80% of data)  
was used to compute `best_val_loss` for early stopping.  
Source: `temperature_experiment.py` lines 104–116.  
The best epoch checkpoint was saved when `val_loss < best_val_loss`.  
`val_loss` is recorded in each checkpoint file (line 143).

### 3. Which split was used for the REPORTED final MSE?

**CLEAN.** `evaluate_temperature_zeroshot()` uses the test loader  
(`data_raw[n_train+n_val:]`, rows 80–100% of data).  
Source: `temperature_experiment.py` lines 181–188.  
Test split was never used during training or early stopping.

### 4. Whether tau was selected AFTER observing test performance.

**CONTAMINATED. This is the methodological flaw.**

In Phase 2, all tau candidates (0.5, 1.0, 2.0, 4.0, learnable) were run,  
their test MSEs were recorded in `temperature_ablation_results.csv`,  
and tau=4.0 was identified as "best" by directly comparing those test MSEs.

This constitutes **test-set look-ahead during hyperparameter selection**.  
The correct procedure is: select tau using VALIDATION performance only,  
then report test MSE once, for the selected tau only.

### 5. Were other hyperparameters selected using the test set?

**NO.** All other hyperparameters (d_model=128, d_ff=256, k_freq=16, lr=1e-3,  
batch_size=64, epochs=10, patience=3, O_train=48) were fixed before any  
experiment was run and were not tuned using either the validation or test split.

### 6. Are the six forecast horizons evaluated on the same held-out test data?

**YES.** `get_data_loaders` uses the same absolute index range  
(`data_raw[n_train+n_val:]`) regardless of `pred_len`.  
The test data is the same 20% slice for O=24 and O=720.  
NOTE: The number of sliding windows differs by horizon (shorter pred_len  
yields more test windows), so effective test set sizes differ, but the  
raw data source is identical.

### 7. Do all seeds use identical splits?

**YES.** The split boundaries (`n_train = int(n_total * 0.6)`,  
`n_val = int(n_total * 0.2)`) are deterministic and do not depend on  
the random seed. The seed only affects model weight initialization and  
training batch shuffling.

### 8. Do all tau variants receive identical training budget?

**YES.** All variants are trained with:  
- lr = 1e-3, epochs = 10, patience = 3, batch_size = 64  
- Same optimizer (Adam), same loss (MSE), same O_train = 48  
- Same initialization scheme (temperature_mode='fixed' differs only in  
  the tau_fixed buffer value, not in weight initialization)  
- Early stopping fires at different epochs per run, but the stopping  
  criterion (3 epochs without val improvement) is identical.

---

## Classification of Phase 2 Temperature Results

The existing `temperature_ablation_results.csv` results are classified as:

> **EXPLORATORY ABLATION / HYPOTHESIS-GENERATING EVIDENCE**

Reason: tau=4.0 was identified as best by inspecting test MSE across all  
candidates. The reported "best tau" has implicit test-set contamination.  
The results are not invalid as exploratory evidence, but the improvement  
of tau=4 over tau=1 cannot be treated as a validated finding until  
proper validation-based selection is performed.

---

## What Phase 3B Must Do

A clean experiment must:

1. For each (dataset, seed):  
   a. Train all fixed-tau candidates at O_train=48  
   b. Record `best_val_loss` from the checkpoint for each candidate  
   c. Select `tau_selected = argmin(val_loss)` using ONLY validation MSE  
   d. Re-evaluate ONLY the selected tau on the test set  

2. Report: selected tau per seed, val_loss per candidate, final test MSE.

3. The `val_loss` field IS already saved in each checkpoint  
   (`temp_d2v_{dataset}_{variant}_seed{seed}.pt`, key `val_loss`).  
   If the Colab checkpoints are still on Drive, Phase 3B can be executed  
   by loading them without retraining. See `Phase3_Colab_Validation.ipynb`.

---

## Summary

| Question | Status |
|----------|--------|
| Train split used correctly | PASS |
| Val split used for early stopping | PASS |
| Test split isolated from training | PASS |
| Tau selected without test-set contamination | **FAIL** |
| Other hyperparameters test-free | PASS |
| Horizons evaluated on same test data | PASS |
| Seeds use identical splits | PASS |
| Equal training budget across variants | PASS |

**Overall verdict:** The training protocol is sound. The single flaw is  
post-hoc tau selection using test performance. All other aspects of the  
experimental design are methodologically correct.
