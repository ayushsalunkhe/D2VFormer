# Understanding Your BE Major Project: D2Vformer Time-Series Forecasting
## Complete Beginner's Guide — From Zero to Expert

---

## Table of Contents

1. [What is Time-Series Forecasting?](#1-what-is-time-series-forecasting)
2. [Why Do We Need It?](#2-why-do-we-need-it)
3. [The Problem D2Vformer Solves](#3-the-problem-d2vformer-solves)
4. [Understanding the D2Vformer Paper](#4-understanding-the-d2vformer-paper)
5. [Key Concepts Explained](#5-key-concepts-explained)
6. [The D2Vformer Architecture](#6-the-d2vformer-architecture)
7. [Your Datasets Explained](#7-your-datasets-explained)
8. [How Training Works](#8-how-training-works)
9. [How Evaluation Works](#9-how-evaluation-works)
10. [The Baseline: DLinear](#10-the-baseline-dlinear)
11. [What Your Code Actually Does](#11-what-your-code-actually-does)
12. [The Complete Workflow](#12-the-complete-workflow)

---

## 1. What is Time-Series Forecasting?

### The Simplest Explanation

Imagine you have temperature readings for the last 4 days:
- Day 1: 25°C
- Day 2: 27°C
- Day 3: 26°C
- Day 4: 28°C

**Time-series forecasting asks:** "What will the temperature be on Day 5?"

### Definition

**Time-series data** = measurements taken at regular intervals over time
- Stock prices every minute
- Temperature every hour
- Sales every day
- Website traffic every second

**Forecasting** = using past data to predict future values

### Real Example from Your Project

**Delhi Air Quality (PM2.5 pollution):**
```
2022-04-06 00:00 → PM2.5 = 200 µg/m³
2022-04-06 01:00 → PM2.5 = 210 µg/m³
2022-04-06 02:00 → PM2.5 = 195 µg/m³
...
2022-04-10 00:00 → PM2.5 = ???  ← THIS is what we want to predict
```

---

## 2. Why Do We Need It?

### Real-World Applications

| Domain | Use Case | Why It Matters |
|--------|----------|----------------|
| **Healthcare** | Predict hospital patient arrivals | Staff scheduling, resource allocation |
| **Energy** | Predict electricity demand | Prevent blackouts, optimize generation |
| **Finance** | Predict stock prices | Investment decisions, risk management |
| **Environment** | Predict air pollution | Public health alerts, traffic control |
| **Retail** | Predict product demand | Inventory management, reduce waste |

### Your Project's Use Case: Delhi Air Quality

**Problem:** Delhi has severe air pollution. PM2.5 (fine particulate matter) causes:
- Respiratory diseases
- School closures
- Flight cancellations
- Economic losses

**Solution:** If we can predict pollution 96 hours (4 days) ahead:
- Government can issue health advisories
- People can plan outdoor activities
- Traffic management can reduce vehicle emissions
- Construction can be paused during peak pollution

---

## 3. The Problem D2Vformer Solves

### Traditional Forecasting Models Had Limitations

#### Problem 1: Fixed Prediction Horizons

**Old models** like LSTM, Transformer said:
> "I can only predict the next 96 hours. If you want to predict 48 hours or 192 hours, you need to retrain me from scratch."

**D2Vformer says:**
> "I use calendar timestamps (hour, day, weekday, month). I can predict ANY future time without retraining."

#### Example to Understand This

**Traditional Model:**
- Trained to predict "next 96 steps"
- Input: [step 1, step 2, ..., step 96] → Output: [step 97, step 98, ..., step 192]
- If you want step 150 specifically → TOO BAD, retrain the model

**D2Vformer:**
- Input: [timestamp 2022-04-01 00:00, ..., 2022-04-05 00:00] + "I want 2022-04-09 12:00"
- Output: Prediction for exactly 2022-04-09 12:00
- Want 2022-04-10 06:00? Just change the query timestamp!

#### Problem 2: Ignoring Calendar Patterns

**Example:** Electricity demand has patterns:
- **Hour of day:** Peak at 8 PM (dinner time), low at 4 AM (sleep)
- **Day of week:** Higher on weekdays (offices open), lower on Sunday
- **Month:** Higher in July (AC usage), lower in October

**Old models** saw these as just "step 1, step 2, step 3..." They didn't know "step 1" is Monday 9 AM vs Sunday 3 AM.

**D2Vformer** embeds calendar information:
- Learns "Monday mornings have X pattern"
- Learns "January has Y baseline"
- Learns "Hour 20 (8 PM) has Z spike"

---

## 4. Understanding the D2Vformer Paper

### Paper Details

- **Title:** D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding
- **Published:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS), May 2026
- **Impact:** Top-tier AI journal (Impact Factor ~14)
- **Authors:** Yumi et al., Chinese research team
- **Code:** https://github.com/yumiyumo/D2Vformer

### What the Paper Claims

1. **Date2Vec module** encodes calendar timestamps as vectors (embeddings)
2. **Fusion Block** matches future timestamps against past data using attention
3. **Flexible forecasting** at arbitrary future times without retraining
4. **State-of-art results** on 6 benchmark datasets

### The Innovation: Date2Vec

**Problem:** How do you tell a neural network "this datapoint is from Monday 9 AM in January"?

**Solution:** Date2Vec converts timestamps into numerical vectors:

```
Input timestamp: 2022-01-03 09:00 (Monday, 9 AM, January)

Date2Vec output: [0.23, -0.45, 0.67, ..., 0.12]  ← 36-dimensional vector
                  ↑        ↑       ↑           ↑
                  hour   weekday  day      month
                 pattern pattern pattern  pattern
```

**How it works:**
- Uses sine/cosine functions (Fourier series) for periodicity
- Hour: sin(2π × hour/24), cos(2π × hour/24) → captures 24-hour cycle
- Weekday: sin(2π × day/7), cos(2π × day/7) → captures weekly cycle
- Similar for day-of-month (monthly cycle) and month (yearly cycle)

---

## 5. Key Concepts Explained

### Concept 1: Multivariate Time-Series

**Univariate** = one variable over time
```
Time    Temperature
00:00   25°C
01:00   24°C
02:00   23°C
```

**Multivariate** = multiple related variables over time
```
Time    PM2.5   PM10    NO2    SO2    CO     O3
00:00   200     350     45     12     1.2    30
01:00   210     360     48     13     1.3    28
02:00   195     340     42     11     1.1    32
```

Your Delhi AQI dataset has **6 variables** (pollutants), so it's multivariate.

### Concept 2: Sequence Length (seq_len)

How much history do we look at?

```
seq_len = 96 means:
- Look at last 96 hours (4 days) of data
- Use this to predict the future
```

### Concept 3: Prediction Length (pred_len)

How far ahead do we predict?

```
pred_len = 96 means:
- Predict next 96 hours (4 days)
```

### Concept 4: Label Length (label_len)

Overlap between history and prediction (used in Transformer-based models for better context).

```
label_len = 48 means:
- Last 48 hours of history are also used as "decoder input"
- Helps the model transition from past to future smoothly
```

### Visual Example

```
Timeline: [-------- History --------][--- Future ---]
          Hour 0 ............... Hour 96  ...  Hour 192

seq_len=96:    [================]
                                  ↑
label_len=48:                 [====]====]  (overlap)
                                  ↑
pred_len=96:                      [================]
```

### Concept 5: Batch Size

How many examples we train on at once.

```
batch_size = 64 means:
- Take 64 different time windows
- Train on all 64 simultaneously (parallel processing on GPU)
- Faster training than doing one at a time
```

### Concept 6: Normalization

**Problem:** Different variables have different scales:
- PM2.5: 50-500 µg/m³
- CO: 0.5-3.0 mg/m³

**Solution:** Normalize to mean=0, std=1:
```python
normalized_value = (original_value - mean) / std

# Example for PM2.5:
mean_pm25 = 150
std_pm25 = 50
original = 200
normalized = (200 - 150) / 50 = 1.0
```

**Why?** Neural networks train better when all inputs are on similar scales.

### Concept 7: Train/Validation/Test Split

**Training set (70%):** Used to learn patterns
**Validation set (10%):** Used to tune hyperparameters and prevent overfitting
**Test set (20%):** Used ONLY for final evaluation (model has never seen this data)

```
Total data: 35,064 hours (Delhi AQI)
├── Train:      24,545 hours (70%)  → model learns from this
├── Validation:  3,506 hours (10%)  → check if learning is good
└── Test:        7,013 hours (20%)  → final exam for the model
```

### Concept 8: MSE and MAE (Evaluation Metrics)

**MSE (Mean Squared Error):**
```
MSE = average of (prediction - actual)²

Example:
Actual:     [100, 120, 110]
Predicted:  [105, 115, 112]
Errors:     [5, -5, 2]
Squared:    [25, 25, 4]
MSE = (25 + 25 + 4) / 3 = 18
```

**MAE (Mean Absolute Error):**
```
MAE = average of |prediction - actual|

Same example:
Errors:     [5, -5, 2]
Absolute:   [5, 5, 2]
MAE = (5 + 5 + 2) / 3 = 4
```

**Which is better?**
- Lower MSE/MAE = better predictions
- MSE penalizes large errors more (squared)
- MAE is more interpretable (same units as original data)

---

## 6. The D2Vformer Architecture

### High-Level Overview

```
Input: 
  - Historical data: [96 hours × 6 pollutants]
  - Historical timestamps: [hour, weekday, day, month] for each hour
  - Future timestamps: [hour, weekday, day, month] for next 96 hours

Processing:
  1. Patch the historical data into chunks
  2. Encode timestamps using Date2Vec
  3. Fusion Block: match future dates against past data
  4. Output: Predictions for next 96 hours

Output:
  - Predicted values: [96 hours × 6 pollutants]
```

### Step-by-Step Breakdown

#### Step 1: Patching

Instead of processing 96 hours one-by-one, group them:

```
Original: [h1, h2, h3, h4, h5, h6, h7, h8, ...]  (96 values)

Patched (patch_len=16, stride=8):
  Patch 1: [h1-h16]
  Patch 2: [h9-h24]   (overlaps with patch 1)
  Patch 3: [h17-h32]
  ...
  
Result: 12 patches instead of 96 individual points
```

**Why?** More efficient, captures local patterns.

#### Step 2: Date2Vec Embedding

For each hour in history and future:

```python
timestamp = (hour=9, weekday=1, day=3, month=1)  # Monday 9 AM, Jan 3

# Fourier encoding
hour_sin = sin(2π × 9/24)    = 0.707
hour_cos = cos(2π × 9/24)    = 0.707
day_sin  = sin(2π × 1/7)     = 0.782
day_cos  = cos(2π × 1/7)     = 0.623
...

# Concatenate all features
fourier_vec = [0.707, 0.707, 0.782, 0.623, ...]  # 20 dimensions

# Upsample to higher dimension (T2V_outmodel=36)
date_embedding = Linear(fourier_vec)  # [36 dimensions]
```

#### Step 3: Feature Fusion

Combine the patched data with date embeddings:

```
Patched data:       [12 patches × d_model]
Date embeddings:    [12 patches × 36]

Concatenate:        [12 patches × (d_model + 36)]
```

#### Step 4: Transformer Encoder

Process the fused features through 2 layers of self-attention:

```
Each patch "talks" to every other patch to understand:
- Dependencies (e.g., hour 8 affects hour 12)
- Patterns (e.g., morning rush repeats daily)
```

#### Step 5: Fusion Block (The Key Innovation)

**Goal:** Predict future values by matching future timestamps against learned past patterns.

```python
# Simplified pseudocode
past_features = encoder_output       # [12 patches × d_model]
future_dates = Date2Vec(future_times) # [96 hours × 36]

# Cross-attention: each future date queries the past
for each future_hour in range(96):
    # "Which past patterns match this future timestamp?"
    attention_weights = softmax(future_dates[future_hour] @ past_features.T)
    prediction[future_hour] = attention_weights @ past_features
```

**Intuition:** If predicting "next Monday 9 AM", the model looks for "previous Monday 9 AM" patterns in history.

#### Step 6: Output Projection

```
Fused future features: [96 hours × d_model]
↓
Linear layer: [96 hours × 6 pollutants]
↓
Final predictions: [96 hours × 6 pollutants]
```

---

## 7. Your Datasets Explained

### Dataset 1: ETTh1 (Electricity Transformer Temperature)

**What it is:**
- Temperature readings from an electricity transformer in China
- Recorded hourly from July 2016 to July 2018
- 7 variables: 6 temperature sensors + 1 oil temperature

**Why it's used:**
- Standard benchmark in time-series forecasting research
- Represents industrial monitoring use case
- Has clear daily and seasonal patterns

**Data format:**
```csv
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,3.491,50.57
2016-07-01 01:00:00,5.726,1.964,1.492,0.426,5.577,3.356,50.25
...
```

**Statistics:**
- Total samples: 17,420 hours (~2 years)
- Train: 12,194 hours (70%)
- Val: 1,742 hours (10%)
- Test: 3,484 hours (20%)

### Dataset 2: Delhi AQI (Your Original Contribution)

**What it is:**
- Air quality measurements from Delhi, India
- Recorded hourly from 2019 to 2022
- 6 pollutants: PM2.5, PM10, NO2, SO2, CO, O3

**Why you created it:**
- The D2Vformer paper never tested on Indian data
- Air quality forecasting is crucial for Delhi (world's most polluted capital)
- Shows your model can generalize to new domains

**Data format:**
```csv
date,PM2.5,PM10,NO2,SO2,CO,O3
2019-01-01 00:00:00,200.5,350.2,45.3,12.1,1.2,30.5
2019-01-01 01:00:00,210.3,360.1,48.2,13.0,1.3,28.3
...
```

**Statistics:**
- Total samples: 35,064 hours (~4 years)
- Train: 24,545 hours (70%)
- Val: 3,506 hours (10%)
- Test: 7,013 hours (20%)

**Challenges with Delhi AQI:**
- **High volatility:** PM2.5 can jump 100 µg/m³ in one hour (traffic, weather)
- **Seasonal extremes:** Winter (Nov-Feb) has 3× higher pollution than monsoon (Jul-Sep)
- **Rush hour spikes:** 8-10 AM and 7-9 PM show dramatic increases
- **Unpredictable events:** Construction, festivals (Diwali crackers), stubble burning

---

## 8. How Training Works

### The Training Loop (Simplified)

```python
for epoch in range(50):  # Train for 50 passes through data
    for batch in training_data:
        # 1. Get a batch of examples
        history_x, future_y, history_marks, future_marks = batch
        # history_x: [64 samples × 96 hours × 6 pollutants]
        # future_y: [64 samples × 96 hours × 6 pollutants]
        
        # 2. Forward pass: model makes predictions
        predictions = model(history_x, history_marks, future_y, future_marks)
        # predictions: [64 samples × 96 hours × 6 pollutants]
        
        # 3. Calculate loss: how wrong were the predictions?
        actual = future_y[:, -96:, :]  # Last 96 hours of ground truth
        loss = MSE(predictions, actual)
        # loss: single number, e.g., 0.352
        
        # 4. Backward pass: calculate gradients
        loss.backward()
        # Gradients: how should we adjust each weight to reduce loss?
        
        # 5. Update model weights
        optimizer.step()
        # Weights adjusted slightly to make better predictions next time
        
        # 6. Reset gradients for next batch
        optimizer.zero_grad()
    
    # After each epoch, check validation performance
    val_loss = evaluate(model, validation_data)
    
    # Early stopping: if validation loss stops improving, stop training
    if val_loss hasn't improved in 5 epochs:
        print("Early stopping at epoch", epoch)
        break
```

### Hyperparameters You Tuned

| Hyperparameter | Values Tested | Best Value | What It Does |
|----------------|---------------|------------|--------------|
| **Learning Rate (lr)** | 0.001, 0.0005, 0.0001, 0.00005, 0.00001 | 0.001 | How big a step to take when updating weights |
| **T2V_outmodel** | 36, 64 | 36 (IndiaAQI), 64 (ETTh1) | Dimension of Date2Vec embeddings |
| **Batch Size** | 64, 128 | 64 | How many examples per training step |
| **Epochs** | Up to 100 | ~13-21 (early stopped) | How many passes through training data |
| **Patience** | 5 | 5 | Stop if no improvement after 5 epochs |

### What Happens on Google Colab

```
1. Upload D2Vformer folder to Google Drive
2. Mount Drive in Colab notebook
3. Install dependencies (torch, pandas, etc.)
4. Run training script:
   - For each (lr, T2V) combination:
     * Train model for up to 100 epochs
     * Save best checkpoint (lowest validation loss)
     * Print final test MSE
5. Find best config across all 10 runs
6. Download best checkpoint to local machine
```

**Why Colab?**
- Free GPU (Tesla T4) → 10× faster than your laptop CPU
- Training 10 configs takes ~2-3 hours on GPU vs ~30 hours on CPU

---

## 9. How Evaluation Works

### Test Set Evaluation

```python
# Load best model
model.load_state_dict(torch.load('best_model.pkl'))
model.eval()  # Turn off training mode (no gradient calculation)

all_predictions = []
all_actuals = []

for batch in test_data:
    history_x, future_y, history_marks, future_marks = batch
    
    # Make predictions (no gradient calculation, faster)
    with torch.no_grad():
        predictions = model(history_x, history_marks, future_y, future_marks)
    
    all_predictions.append(predictions)
    all_actuals.append(future_y[:, -96:, :])

# Concatenate all batches
all_predictions = np.concatenate(all_predictions)  # [~110 batches × 96 hours × 6]
all_actuals = np.concatenate(all_actuals)

# Calculate metrics
test_mse = np.mean((all_predictions - all_actuals) ** 2)
test_mae = np.mean(np.abs(all_predictions - all_actuals))

print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
```

### Window-by-Window Visualization

```python
# Pick a specific test window (e.g., window 550)
test_window = testset[550]
history_x, future_y, history_marks, future_marks = test_window

# Make prediction for this window
prediction = model(history_x, history_marks, future_y, future_marks)

# Plot
plt.plot(range(96), history_x[:, 0], label='History (PM2.5)')
plt.plot(range(96, 192), future_y[-96:, 0], label='Actual')
plt.plot(range(96, 192), prediction[:, 0], label='Predicted')
plt.legend()
plt.show()
```

---

## 10. The Baseline: DLinear

### Why We Need a Baseline

**Research principle:** Always compare against a simple model.

**Question:** Is D2Vformer's complexity (Date2Vec, attention, transformers) justified?

**Answer:** Only if it beats a simple baseline.

### What is DLinear?

**DLinear** (Zeng et al., AAAI 2023) = Dead simple linear model:

```
1. Decompose time series into trend + seasonal
2. Apply one linear layer to each component
3. Add them back together
```

### DLinear Architecture (Simplified)

```python
# Step 1: Decomposition
trend = moving_average(history_x)      # [96 hours × 6]
seasonal = history_x - trend           # [96 hours × 6]

# Step 2: Linear projection (separate for each pollutant)
for i in range(6):  # For each pollutant
    trend_pred[i] = Linear_trend[i](trend[:, i])       # [96] → [96]
    seasonal_pred[i] = Linear_seasonal[i](seasonal[:, i])  # [96] → [96]

# Step 3: Add them
prediction = trend_pred + seasonal_pred  # [96 hours × 6]
```

**Total parameters:** ~100K (vs D2Vformer's ~2M)

### Why DLinear is a Good Baseline

1. **Simple:** No attention, no transformers, just linear layers
2. **Fast:** Trains in ~5 minutes vs D2Vformer's ~30 minutes
3. **Interpretable:** Easy to understand what it's learning
4. **Strong:** Despite simplicity, often beats complex models

### Your Results

| Dataset | Model | Test MSE | Test MAE | Winner |
|---------|-------|----------|----------|--------|
| ETTh1 | DLinear | 0.4033 | 0.4358 | ✓ |
| ETTh1 | D2Vformer | 0.6982 | 0.6193 | |
| Delhi AQI | DLinear | 0.1658 | 0.2710 | ✓ |
| Delhi AQI | D2Vformer | 0.2212 | 0.3072 | |

**Finding:** DLinear wins on both datasets.

**Interpretation:**
- ETTh1 and Delhi AQI may not have strong calendar effects
- Simple linear trends + seasonality capture most patterns
- D2Vformer's Date2Vec overhead doesn't help here

---

## 11. What Your Code Actually Does

### File Structure

```
Major Project/
├── D2Vformer/
│   └── D2Vformer/
│       ├── main.py                    # Training entry point
│       ├── model/
│       │   ├── D2Vformer_simple.py   # Model architecture
│       │   ├── Date2Vec.py            # Calendar embedding
│       │   └── layers/
│       │       ├── Fusion_Block.py    # Future-past attention
│       │       └── Patch_embedding.py # Patching module
│       ├── data/
│       │   └── dataset.py             # Data loading
│       ├── utils/
│       │   ├── get_data.py            # CSV → normalized arrays
│       │   └── earlystopping.py       # Training control
│       ├── baselines/
│       │   ├── run_dlinear.py         # DLinear training
│       │   └── plot_comparison.py     # Comparison graphs
│       ├── datasets/
│       │   ├── ETT-small/
│       │   │   ├── ETTh1.csv          # Electricity data
│       │   │   └── china.csv          # Timestamps
│       │   └── india_aqi/
│       │       ├── delhi_aqi.csv      # Air quality data
│       │       └── delhi_mark.csv     # Timestamps
│       └── experiments/
│           ├── exp11/                 # ETTh1 best run
│           │   └── D2Vformer_s/
│           │       └── ETTh1_best_model.pkl
│           └── exp16/                 # IndiaAQI best run
│               └── D2Vformer_s/
│                   └── IndiaAQI_best_model.pkl
├── streamlit_app.py                   # Interactive demo
├── results/
│   ├── comparison_bar.png
│   ├── comparison_forecast.png
│   ├── ETTh1_pred96_forecast.png
│   └── IndiaAQI_forecast.png
└── D2Vformer_Colab.ipynb              # Google Colab notebook
```

### Key Scripts

#### 1. Training: `main.py`

```bash
python main.py \
  --model_name D2Vformer_s \
  --data_name IndiaAQI \
  --seq_len 96 \
  --pred_len 96 \
  --lr 0.001 \
  --T2V_outmodel 36 \
  --epoches 50 \
  --batch_size 64
```

**What it does:**
1. Load delhi_aqi.csv and delhi_mark.csv
2. Split into train/val/test (70/10/20)
3. Normalize data
4. Create D2Vformer_simple model
5. Train for up to 50 epochs
6. Save best checkpoint to experiments/expXX/
7. Print final test MSE/MAE

#### 2. Plotting: `plot_forecast.py`

```bash
python plot_forecast.py --exp_dir experiments/exp11
```

**What it does:**
1. Load checkpoint from exp11
2. Load test data
3. Pick a test window
4. Generate prediction
5. Plot history vs actual vs predicted
6. Save to results/ETTh1_pred96_forecast.png

#### 3. Baseline: `baselines/run_dlinear.py`

```bash
python baselines/run_dlinear.py --data_name IndiaAQI --pred_len 96
```

**What it does:**
1. Load same data as D2Vformer
2. Create DLinear model (much simpler)
3. Train for up to 50 epochs
4. Save checkpoint
5. Print test MSE/MAE

#### 4. Demo: `streamlit_app.py`

```bash
streamlit run streamlit_app.py
```

**What it does:**
1. Create web interface at localhost:8502
2. Load both D2Vformer and DLinear checkpoints
3. Allow user to:
   - Select dataset (ETTh1/Delhi AQI)
   - Select feature (PM2.5, OT, etc.)
   - Select test window (0-2000)
   - Toggle models on/off
4. Generate interactive forecast plot
5. Show MSE/MAE metrics

---

## 12. The Complete Workflow

### Phase 1: Local Setup (Week 1-2)

```
1. Clone D2Vformer repo from GitHub
2. Fix 6 bugs to make it run
   - Pandas 3.0 compatibility
   - NumPy 2.0 compatibility
   - Missing arguments
   - Fusion_Block residual bug
   - Early stopping scheduler
3. Run single ETTh1 training locally
   → Get MSE 0.7067 (baseline)
```

### Phase 2: Hyperparameter Sweep on Colab (Week 3)

```
1. Upload D2Vformer to Google Drive
2. Create Colab notebook with sweep script
3. Run 10 configs (5 LRs × 2 T2V sizes) on ETTh1
   → Takes ~2 hours on T4 GPU
4. Best config: lr=0.001, T2V=36, MSE=0.6982
5. Download best checkpoint
6. Generate forecast graph locally
```

### Phase 3: India Extension (Week 4-5)

```
1. Download Delhi AQI data from government portal
2. Preprocess into same format as ETTh1
   - Hourly samples, 6 pollutants
   - Create delhi_aqi.csv + delhi_mark.csv
3. Register in exp.py as 'IndiaAQI' dataset
4. Upload to Google Drive
5. Add IndiaAQI training to Colab notebook
6. Run same 10-config sweep
   → Best: lr=0.001, T2V=36, MSE=0.2212
7. Download checkpoint and generate graph
```

### Phase 4: Baseline Comparison (Week 6)

```
1. Implement DLinear from scratch
   - Read paper, understand architecture
   - Write run_dlinear.py
2. Train DLinear on ETTh1 (pred=96)
   → MSE 0.4033 (beats D2Vformer's 0.6982!)
3. Train DLinear on IndiaAQI (pred=96)
   → MSE 0.1658 (beats D2Vformer's 0.2212)
4. Write plot_comparison.py
   - Bar chart comparing MSE/MAE
   - Forecast overlay for both models
5. Generate 4 comparison graphs
```

### Phase 5: Demo (Week 7)

```
1. Install Streamlit
2. Write streamlit_app.py
   - Load both model checkpoints
   - Create interactive UI
   - Add date/season context
3. Test locally at localhost:8502
4. Create requirements.txt for deployment
5. Write DEMO_README.md
```

### Current Status (End of Semester 1)

✅ **Phase 1-5 Complete**
- Working reproduction with bug fixes documented
- India AQI extension dataset created
- Honest baseline comparison done
- Interactive demo running

📊 **Deliverables Ready:**
- 4 result graphs (comparison_bar, comparison_forecast, ETTh1, IndiaAQI)
- Streamlit web demo
- Trained checkpoints for both datasets
- Google Colab notebook for reproduction

📝 **Documentation Ready:**
- CLAUDE.md: Project tracking
- DEMO_README.md: How to run the demo
- Bug fixes documented as "reproduction engineering"

---

## Summary: The Big Picture

### What Problem Are You Solving?

Predicting future values of time-series data (electricity, pollution) 96 hours ahead.

### What Model Are You Using?

D2Vformer: A transformer-based model that uses calendar timestamps (Date2Vec) to enable flexible forecasting.

### What Did You Do?

1. Reproduced D2Vformer (fixed 6 bugs)
2. Extended to Indian air quality data (your contribution)
3. Compared against DLinear baseline (honest evaluation)
4. Built interactive demo

### What Did You Find?

DLinear (simple linear model) beats D2Vformer on both datasets, suggesting:
- These time series may not have strong calendar effects, OR
- D2Vformer needs more tuning, OR
- The datasets are too small/noisy

This is a valid research finding → honest evaluation > cherry-picked results.

### What Can You Say When Asked "What's New?"

"We extended D2Vformer to Indian air quality data and ran honest baseline comparisons. Our key finding: simpler models (DLinear) outperform complex calendar-aware architectures on these domains, suggesting that Date2Vec's inductive bias may not be universally beneficial."

---

## Next Steps (Semester 2)

If your guide asks what's next:

1. **Investigate why DLinear wins**
   - Ablation study: remove Date2Vec, compare
   - Try datasets with stronger calendar patterns (e.g., retail sales)

2. **Uncertainty quantification**
   - Add prediction intervals (not just point forecasts)
   - Probabilistic forecasting for risk-aware decisions

3. **Anomaly detection**
   - Use D2Vformer residuals to detect pollution spikes
   - Real-time alerting system for Delhi

4. **Ensemble methods**
   - Combine DLinear + D2Vformer
   - Simple for baseline, complex for calendar effects

---

## Glossary of Terms

| Term | Simple Definition | Example |
|------|-------------------|---------|
| **Time-series** | Data points measured over time | Stock prices, temperature |
| **Forecasting** | Predicting future values | Tomorrow's weather |
| **Multivariate** | Multiple variables tracked together | PM2.5 + PM10 + NO2 |
| **Seq_len** | How much history to look at | Last 96 hours |
| **Pred_len** | How far ahead to predict | Next 96 hours |
| **Batch** | Group of examples trained together | 64 windows at once |
| **Epoch** | One pass through all training data | 50 epochs = 50 loops |
| **MSE** | Mean squared error (loss metric) | 0.2212 (lower = better) |
| **MAE** | Mean absolute error | 0.3072 (lower = better) |
| **Normalization** | Scaling data to mean=0, std=1 | Make all features comparable |
| **Embedding** | Converting categorical/temporal data to vectors | Date → [0.23, -0.45, ...] |
| **Attention** | Mechanism to focus on relevant past data | "Which past Monday matters?" |
| **Transformer** | Neural network using attention | BERT, GPT, D2Vformer |
| **Baseline** | Simple model for comparison | DLinear |
| **Hyperparameter** | Settings you choose before training | Learning rate, batch size |
| **Checkpoint** | Saved model weights | best_model.pkl |

---

**End of Document 1: Project Explanation**

This document explained the project from absolute scratch. You should now understand:
- What time-series forecasting is
- Why D2Vformer was created
- How the model works (Date2Vec, Fusion Block)
- Your two datasets (ETTh1, Delhi AQI)
- How training and evaluation work
- What DLinear is and why it's a baseline
- Your complete workflow from Week 1 to Week 7

**Next:** Read Document 2 for detailed improvements and achievements.
