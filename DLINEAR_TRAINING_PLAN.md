# DLinear Multi-Horizon Training — Pre-Training Report
## Status: READY FOR GOOGLE COLAB TRAINING

**Date:** August 2024  
**Status:** ⏸️ AWAITING APPROVAL BEFORE TRAINING

---

## What We've Completed

### ✅ Code Audit
- Inspected existing `baselines/run_dlinear.py`
- Verified preprocessing matches D2Vformer (same `get_data()`, same splits, same normalization)
- Confirmed DLinear architecture requires separate training per pred_len
- No methodology inconsistencies found

### ✅ Multi-Horizon Training Script Created
- File: `baselines/run_dlinear_multihorizon.py`
- Accepts command-line arguments: `--data_name`, `--pred_len`, `--epochs`, etc.
- Saves checkpoints: `baselines/dlinear_{dataset}_pred{horizon}.pkl`
- Saves results: `experiments_flexible/dlinear_{dataset}_pred{horizon}_result.json`
- Will NOT overwrite existing 96h checkpoints

### ✅ Local Smoke Test Passed
- Tested: ETTh1, pred_len=48, 1 epoch, batch_size=16
- Result: Training completed in 6.99s, MSE=0.3682
- Checkpoint saved successfully
- Shape verification: output (batch, 48, 7) ✓
- Ground truth alignment: `by[:, -48:, :]` ✓

### ✅ Google Colab Notebook Prepared
- File: `DLinear_MultiHorizon_Training.ipynb`
- Steps: Mount Drive → Check GPU → Verify datasets → Train all horizons → Download results
- Includes error handling and progress tracking
- Automated result consolidation

---

## Training Plan

### Datasets & Horizons

**ETTh1:**
- ✅ 96h: Already trained (MSE=0.4033, checkpoint exists)
- ⏳ 48h: NEW — needs training
- ⏳ 72h: NEW — needs training
- ⏳ 192h: NEW — needs training
- ⏳ 336h: NEW — needs training

**Delhi AQI:**
- ✅ 96h: Already trained (MSE=0.1658, checkpoint exists)
- ⏳ 48h: NEW — needs training
- ⏳ 72h: NEW — needs training
- ⏳ 192h: NEW — needs training
- ⏳ 336h: NEW — needs training

**Total new training runs:** 8

---

## Expected Training Time

Based on Semester 1 training experience:

| Dataset | Horizon | Expected Epochs | Expected Time (T4 GPU) |
|---------|---------|-----------------|------------------------|
| ETTh1 | 48h | ~15-20 | ~8 min |
| ETTh1 | 72h | ~15-20 | ~8 min |
| ETTh1 | 192h | ~15-20 | ~10 min |
| ETTh1 | 336h | ~15-20 | ~12 min |
| IndiaAQI | 48h | ~10-15 | ~5 min |
| IndiaAQI | 72h | ~10-15 | ~5 min |
| IndiaAQI | 192h | ~10-15 | ~7 min |
| IndiaAQI | 336h | ~10-15 | ~8 min |

**Total estimated time:** ~60-70 minutes on Colab T4 GPU

---

## Training Configuration

### Hyperparameters (Match Semester 1)
```
seq_len: 96
label_len: 0 (DLinear doesn't use decoder input)
batch_size: 64
lr: 0.001
optimizer: Adam
loss: MSE
epochs: 50 (max)
patience: 5 (early stopping)
```

### Dataset Configuration
```
ETTh1:
  d_feature: 7
  data_path: ./datasets/ETT-small/ETTh1.csv
  mark_path: ./datasets/ETT-small/china.csv

IndiaAQI:
  d_feature: 6
  data_path: ./datasets/india_aqi/delhi_aqi.csv
  mark_path: ./datasets/india_aqi/delhi_mark.csv
```

### Train/Val/Test Split
```
70% / 10% / 20% (same as D2Vformer)
```

---

## Output Files

### Checkpoints (8 new files)
```
baselines/dlinear_ETTh1_pred48.pkl
baselines/dlinear_ETTh1_pred72.pkl
baselines/dlinear_ETTh1_pred192.pkl
baselines/dlinear_ETTh1_pred336.pkl

baselines/dlinear_IndiaAQI_pred48.pkl
baselines/dlinear_IndiaAQI_pred72.pkl
baselines/dlinear_IndiaAQI_pred192.pkl
baselines/dlinear_IndiaAQI_pred336.pkl
```

### Result Files (8 new JSON files)
```
experiments_flexible/dlinear_ETTh1_pred48_result.json
experiments_flexible/dlinear_ETTh1_pred72_result.json
experiments_flexible/dlinear_ETTh1_pred192_result.json
experiments_flexible/dlinear_ETTh1_pred336_result.json

experiments_flexible/dlinear_IndiaAQI_pred48_result.json
experiments_flexible/dlinear_IndiaAQI_pred72_result.json
experiments_flexible/dlinear_IndiaAQI_pred192_result.json
experiments_flexible/dlinear_IndiaAQI_pred336_result.json
```

### Consolidated Results
```
experiments_flexible/dlinear_all_horizons.json
```

---

## Verification Checklist

### ✅ Code Verified
- [x] Training script created and smoke-tested
- [x] Same preprocessing as D2Vformer
- [x] Same train/val/test split
- [x] Same normalization (mean/scale from get_data)
- [x] Metrics on same scale (normalized)
- [x] No train/test leakage

### ✅ Local Testing
- [x] 1-epoch smoke test completed successfully
- [x] Shape verification passed
- [x] Checkpoint saving works
- [x] Result JSON generation works

### ✅ Google Colab Ready
- [x] Notebook created with all steps
- [x] Dataset path verification included
- [x] GPU check included
- [x] Progress tracking included
- [x] Result consolidation automated
- [x] Download instructions included

### ⏸️ Awaiting Approval
- [ ] User approval to proceed with Colab training
- [ ] Confirmation: 60-70 minutes training time acceptable
- [ ] Confirmation: 8 new checkpoints will be created

---

## What Happens After Training

### Immediate Next Steps
1. Download checkpoints from Colab to local machine
2. Copy to `D2Vformer/D2Vformer/baselines/`
3. Copy results to `experiments_flexible/`
4. Verify all 8 checkpoints downloaded successfully

### Evaluation & Analysis
1. Evaluate all 8 new DLinear checkpoints on test sets
2. Consolidate with existing 96h DLinear results
3. Consolidate with D2Vformer flexible forecasting results
4. Generate comparison table
5. Create plots (MSE vs horizon, training cost comparison)

### Documentation
1. Update `FLEXIBLE_FORECASTING_RESULTS.md`
2. Create `FLEXIBLE_D2VFORMER_VS_DLINEAR.md`
3. Update project status documents
4. Prepare guide meeting slides

---

## Training Commands (for reference)

### Manual Command (if not using notebook)
```bash
# ETTh1
python baselines/run_dlinear_multihorizon.py --data_name ETTh1 --pred_len 48
python baselines/run_dlinear_multihorizon.py --data_name ETTh1 --pred_len 72
python baselines/run_dlinear_multihorizon.py --data_name ETTh1 --pred_len 192
python baselines/run_dlinear_multihorizon.py --data_name ETTh1 --pred_len 336

# IndiaAQI
python baselines/run_dlinear_multihorizon.py --data_name IndiaAQI --pred_len 48
python baselines/run_dlinear_multihorizon.py --data_name IndiaAQI --pred_len 72
python baselines/run_dlinear_multihorizon.py --data_name IndiaAQI --pred_len 192
python baselines/run_dlinear_multihorizon.py --data_name IndiaAQI --pred_len 336
```

### Automated via Notebook
The Colab notebook runs all 8 commands sequentially with progress tracking.

---

## Risk Assessment

### Low Risk
- ✅ Code tested locally (1-epoch smoke test passed)
- ✅ Same methodology as Semester 1 (no new experimental variables)
- ✅ Checkpoints named uniquely (won't overwrite existing 96h models)
- ✅ Results saved separately (won't corrupt existing data)

### Medium Risk
- ⚠️ Training time: Could take longer if Colab GPU is slow
  - **Mitigation:** Progress tracking shows per-epoch time, can monitor
- ⚠️ Early stopping: Some horizons might stop before 50 epochs
  - **Mitigation:** Expected behavior, captures convergence naturally

### Negligible Risk
- ✓ Colab session timeout: Notebook saves checkpoints after each horizon
- ✓ Out of memory: DLinear is lightweight (~100K params vs D2Vformer's ~2M)
- ✓ Data corruption: Read-only dataset access

---

## Expected Results Preview

### Prediction: DLinear Accuracy Pattern

Based on Semester 1 findings (DLinear beat D2Vformer at 96h), we expect:

**ETTh1:**
- DLinear will likely be more accurate than D2Vformer at ALL horizons
- Each DLinear model is optimized specifically for its horizon
- D2Vformer trained on 96h, extrapolating to 192h/336h

**Delhi AQI:**
- DLinear will likely be more accurate at most/all horizons
- Higher volatility may affect both models similarly

### The Key Finding
Even if DLinear is more accurate, the comparison shows:

| Aspect | D2Vformer | DLinear |
|--------|-----------|---------|
| **Accuracy** | ? (to be measured) | ? (to be measured) |
| **Training runs** | 1 per dataset | 5 per dataset |
| **Retraining cost** | 0 for new horizons | Full retrain per horizon |
| **Flexibility** | High | Low |

This is the research contribution: **flexibility vs accuracy trade-off**.

---

## Approval Checklist

Before proceeding to Google Colab training, confirm:

- [ ] **Training plan approved:** 8 new models, 60-70 min on Colab T4
- [ ] **Methodology verified:** Same preprocessing/splits as Semester 1
- [ ] **Output locations confirmed:** Checkpoints → `baselines/`, Results → `experiments_flexible/`
- [ ] **Google Drive path confirmed:** User has uploaded project to Drive
- [ ] **Ready to proceed:** User explicitly approves starting training

---

## Files Ready for Review

1. `baselines/run_dlinear_multihorizon.py` — Training script
2. `DLinear_MultiHorizon_Training.ipynb` — Colab notebook
3. `DLINEAR_TRAINING_PLAN.md` — This document

---

## Next Action

**STOP HERE — AWAITING USER APPROVAL**

Once approved, the user will:
1. Upload project to Google Drive (if not already done)
2. Open `DLinear_MultiHorizon_Training.ipynb` in Google Colab
3. Run all cells sequentially
4. Download results after ~70 minutes
5. Report back with checkpoint files and result JSONs

Then we proceed to evaluation and comparison analysis.

---

**Status:** ✅ Pre-training preparation COMPLETE  
**Awaiting:** User approval to start Google Colab training
