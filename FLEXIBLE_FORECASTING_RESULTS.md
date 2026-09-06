# D2Vformer Flexible Forecasting — Consolidated Results Report
## Demonstrating Single-Checkpoint Multi-Horizon Forecasting

**Date:** August 2024  
**Experiment:** Flexible forecasting with existing trained checkpoints  
**Status:** ✅ COMPLETE — No retraining required

---

## Executive Summary

**Key Finding:** D2Vformer can predict multiple horizons (48h, 72h, 96h, 192h, 336h) using a SINGLE trained checkpoint without retraining. This demonstrates the paper's core flexible forecasting innovation.

**Checkpoints Used:**
- ETTh1: `experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl` (trained on 96h→96h)
- Delhi AQI: `experiments/exp16/D2Vformer_s/IndiaAQI_best_model.pkl` (trained on 96h→96h)

**Both checkpoints successfully forecasted all tested horizons without modification.**

---

## Results Table — ETTh1 (Electricity Transformer Temperature)

| Horizon | Retrained? | MSE | MAE | RMSE | % Change vs 96h |
|---------|------------|-----|-----|------|-----------------|
| 48h | ❌ NO | 1.1982 | 0.7731 | 1.0946 | -2.8% (better) |
| 72h | ❌ NO | 1.2199 | 0.7852 | 1.1045 | -1.1% |
| **96h** | ❌ NO | **1.2333** | **0.7919** | **1.1105** | **0.0% (training)** |
| 192h | ❌ NO | 1.2973 | 0.8143 | 1.1390 | +5.2% |
| 336h | ❌ NO | 1.3892 | 0.8481 | 1.1786 | +12.6% |

**Training Configuration:**
- seq_len: 96, label_len: 48, pred_len: 96
- T2V_outmodel: 64
- Features: 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)

**Observations:**
- Shorter horizons (48h, 72h) perform slightly better than training horizon
- Accuracy degrades gracefully at longer horizons
- MSE increases by only 12.6% at 3.5× the training horizon (336h vs 96h)
- Model generalizes well beyond training distribution

---

## Results Table — Delhi AQI (India Air Quality)

| Horizon | Retrained? | MSE | MAE | RMSE | % Change vs 96h | Inference Time |
|---------|------------|-----|-----|------|-----------------|----------------|
| 48h | ❌ NO | 0.1128 | 0.2657 | 0.3358 | -4.3% (better) | 1.95s |
| 72h | ❌ NO | 0.1152 | 0.2688 | 0.3394 | -2.3% | 2.01s |
| **96h** | ❌ NO | **0.1179** | **0.2723** | **0.3434** | **0.0% (training)** | **2.67s** |
| 192h | ❌ NO | 0.1322 | 0.2902 | 0.3635 | +12.1% | 3.93s |
| 336h | ❌ NO | 0.1647 | 0.3273 | 0.4058 | +39.6% | 5.70s |

**Training Configuration:**
- seq_len: 96, label_len: 48, pred_len: 96
- T2V_outmodel: 36
- Features: 6 (PM2.5, PM10, NO2, SO2, CO, O3)

**Observations:**
- Similar pattern: shorter horizons perform better
- Higher degradation at 336h (+39.6%) compared to ETTh1 (+12.6%)
- Delhi AQI's high volatility makes long-horizon forecasting harder
- Inference time scales linearly with horizon length

---

## Verification Checklist

### ✅ Same Checkpoint Used
- **ETTh1:** Single checkpoint `exp11/ETTh1_best_model.pkl` for all 5 horizons
- **Delhi AQI:** Single checkpoint `exp16/IndiaAQI_best_model.pkl` for all 5 horizons
- Verified: Checkpoint weight signature unchanged across all tests

### ✅ No Model Weights Modified
- Checkpoint loaded once at start
- `model.eval()` mode throughout
- No gradient updates
- Weight signature verified before each horizon test

### ✅ Timestamps Correctly Aligned
- `x_mark_dec[:, -pred_len:, :]` dynamically slices future timestamps
- Date2Vec processes exactly `seq_len + pred_len` timestamps
- Output shape verified: `(batch, pred_len, d_feature)`

### ✅ Ground Truth Corresponds to Requested Horizon
- `batch_y[:, -pred_len:, :]` extracts correct ground truth slice
- Shape assertion: `preds.shape == trues.shape`
- Dimension assertion: `preds.shape[1] == pred_len`

### ✅ Metrics Calculated Correctly
- MSE: `mean((pred - true)²)` in normalized space
- MAE: `mean(|pred - true|)` in normalized space
- RMSE: `sqrt(MSE)`
- Denormalized metrics also computed for interpretability

---

## Accuracy Degradation Analysis

### ETTh1 Degradation Pattern

```
Horizon:  48h    72h    96h    192h   336h
MSE:      1.198  1.220  1.233  1.297  1.389
Change:   -2.8%  -1.1%  0.0%   +5.2%  +12.6%
```

**Interpretation:**
- Graceful degradation — model trained on 96h generalizes well
- 192h (2× training) only +5.2% worse
- 336h (3.5× training) only +12.6% worse
- Shorter horizons slightly better (less cumulative error)

### Delhi AQI Degradation Pattern

```
Horizon:  48h    72h    96h    192h   336h
MSE:      0.113  0.115  0.118  0.132  0.165
Change:   -4.3%  -2.3%  0.0%   +12.1% +39.6%
```

**Interpretation:**
- Steeper degradation at 336h (+39.6% vs ETTh1's +12.6%)
- Delhi AQI's high volatility makes extrapolation harder
- Rush-hour spikes and seasonal patterns are unpredictable at 14-day horizon
- Still usable at 192h (+12.1% degradation)

---

## Inference Time Scaling (Delhi AQI)

| Horizon | Inference Time | Samples/Sec | Time per Sample |
|---------|----------------|-------------|-----------------|
| 48h | 1.95s | 819.8 | 1.22ms |
| 72h | 2.01s | 798.0 | 1.25ms |
| 96h | 2.67s | 598.4 | 1.67ms |
| 192h | 3.93s | 406.8 | 2.46ms |
| 336h | 5.70s | 280.7 | 3.56ms |

**Observation:** Inference time scales approximately linearly with horizon length. Longer horizons require more attention computation over future timestamps.

---

## Comparison: D2Vformer vs Traditional Models

### Traditional Fixed-Horizon Models (e.g., LSTM, Transformer)

To forecast multiple horizons, you need:

| Horizon | Model Checkpoint | Training Required | Total Training Time |
|---------|------------------|-------------------|---------------------|
| 48h | model_48h.pkl | ✅ YES | ~20 min |
| 72h | model_72h.pkl | ✅ YES | ~20 min |
| 96h | model_96h.pkl | ✅ YES | ~20 min |
| 192h | model_192h.pkl | ✅ YES | ~30 min |
| 336h | model_336h.pkl | ✅ YES | ~30 min |
| **TOTAL** | **5 checkpoints** | **5 training runs** | **~120 min** |

### D2Vformer Flexible Forecasting

| Horizon | Model Checkpoint | Training Required | Total Training Time |
|---------|------------------|-------------------|---------------------|
| 48h | model_96h.pkl | ❌ NO (reuse) | 0 min |
| 72h | model_96h.pkl | ❌ NO (reuse) | 0 min |
| 96h | model_96h.pkl | ✅ YES (once) | ~20 min |
| 192h | model_96h.pkl | ❌ NO (reuse) | 0 min |
| 336h | model_96h.pkl | ❌ NO (reuse) | 0 min |
| **TOTAL** | **1 checkpoint** | **1 training run** | **~20 min** |

**D2Vformer Advantage:**
- **6× faster deployment** (20 min vs 120 min)
- **5× fewer checkpoints** (1 vs 5)
- **Instant adaptation** to new horizons (no retraining)

---

## Technical Implementation

### Code Changes Required

**File:** `model/D2Vformer_simple_flexible.py`

**Key Modifications:**

1. **Dynamic pred_len parameter:**
```python
def __init__(self, configs):
    self.default_pred_len = configs.pred_len  # Store default, allow override

def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode, pred_len=None):
    if pred_len is None:
        pred_len = self.default_pred_len
```

2. **Dynamic slicing:**
```python
# Old: D2V_input = torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len:, :]), dim=-2)
# New: 
D2V_input = torch.cat((x_mark_enc, x_mark_dec[:, -pred_len:, :]), dim=-2)

# Old: D2V_y_date = D2V_output[:, -self.pred_len:, :, :]
# New:
D2V_y_date = D2V_output[:, -pred_len:, :, :]
```

**No changes to:**
- Date2Vec module (already flexible)
- Fusion Block (already flexible)
- Model parameters (pred_len independent)

### Why This Works

**Architectural Capability:**
- Date2Vec encodes ANY set of timestamps (consecutive or sparse)
- Fusion Block attention matches future queries against history
- No learnable parameters depend on pred_len in D2Vformer_simple

**Training Distribution:**
- Model trained on 96h patterns
- Shorter horizons (48h, 72h): interpolation within training range
- Longer horizons (192h, 336h): extrapolation beyond training range
- Extrapolation causes gradual accuracy degradation

---

## Limitations & Caveats

### Limitation 1: Training Distribution Mismatch

**Issue:** Model trained on 96h consecutive windows, tested on 48h-336h.

**Impact:**
- 48h/72h: Within training distribution (good accuracy)
- 192h/336h: Extrapolation (accuracy degrades)

**Mitigation (for future work):** Train on multi-horizon mixture (mix 48h, 96h, 192h samples).

### Limitation 2: Not True Arbitrary Timestamps

**What we demonstrated:** Variable-length consecutive forecasting
- 48 consecutive hours
- 96 consecutive hours
- 192 consecutive hours

**What we did NOT demonstrate:** Sparse timestamp queries
- Predict exactly [April 7 12:00, April 12 09:00, April 20 18:00]
- Non-adjacent timestamps

**Status:** Theoretically possible with further modifications, not implemented yet.

### Limitation 3: Delhi AQI Volatility

**Issue:** High-frequency pollution spikes (rush hours, construction, weather)

**Impact:** Longer horizons struggle more (+39.6% MSE at 336h)

**Root Cause:** Missing external variables (traffic data, wind speed, construction schedules)

**Mitigation:** Add exogenous features or focus on shorter horizons (96h-192h).

---

## Next Steps for Complete Demonstration

### Phase 1: DLinear Multi-Horizon Comparison ⏳ (NOT STARTED)

Train DLinear separately for each horizon to demonstrate D2Vformer's advantage:

| Model | 48h | 72h | 96h | 192h | 336h | Total Training |
|-------|-----|-----|-----|------|------|----------------|
| D2Vformer | ✅ | ✅ | ✅ | ✅ | ✅ | 1 run (~20 min) |
| DLinear | ⏳ | ⏳ | ✅ | ⏳ | ⏳ | 5 runs (~100 min) |

**Why needed:** Shows that D2Vformer's flexibility has practical value over per-horizon models.

### Phase 2: Streamlit Integration ⏳ (NOT STARTED)

Add flexible horizon selector to existing demo:
- Dropdown: [48h, 72h, 96h, 192h, 336h]
- Single D2Vformer checkpoint
- Compare against multiple DLinear checkpoints
- Show training cost difference

### Phase 3: Documentation & Presentation ⏳ (NOT STARTED)

Create:
- Guide meeting slides
- Updated project documentation
- Black book section on flexible forecasting
- Results graphs for publication

---

## Conclusion

**We successfully demonstrated D2Vformer's flexible forecasting capability:**

✅ **Single checkpoint** works for all horizons (48h-336h)  
✅ **No retraining** required for new horizons  
✅ **Graceful accuracy degradation** at longer horizons  
✅ **Verified on two datasets** (ETTh1 and Delhi AQI)  
✅ **Proper ground truth alignment** confirmed  
✅ **Metrics correctly calculated** (MSE, MAE, RMSE)

**This is the core innovation of the D2Vformer paper.**

**Next decision point:** Proceed with DLinear multi-horizon training to complete the comparison, or finalize current results first?

---

**Files Generated:**
- `experiments_flexible/etth1_flexible_results.json`
- `experiments_flexible/indiaaqi_flexible_results.json`
- `model/D2Vformer_simple_flexible.py`
- `test_flexible_shapes.py`
- `test_flexible_checkpoint.py`
- `test_flexible_checkpoint_indiaaqi.py`

**Checkpoints Used (UNCHANGED):**
- `experiments/exp11/D2Vformer_s/ETTh1_best_model.pkl`
- `experiments/exp16/D2Vformer_s/IndiaAQI_best_model.pkl`
