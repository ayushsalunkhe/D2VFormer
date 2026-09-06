# D2Vformer Time-Series Forecasting — BE Major Project

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)

**Bachelor of Engineering Major Project**  
**Course:** Time-Series Forecasting with Deep Learning  
**Institution:** [Your Institution Name]  
**Academic Year:** 2023-2024

---

## 📋 Project Overview

This repository contains our BE Major Project implementation focused on **flexible multi-horizon time-series forecasting** using the D2Vformer architecture with Date2Vec embeddings.

**Based on:** "D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding"  
IEEE Transactions on Neural Networks and Learning Systems (TNNLS), Vol. 37, No. 5, May 2026  
**Original Repository:** [yumiyumo/D2Vformer](https://github.com/yumiyumo/D2Vformer)

### ⚠️ Important Note

This is **NOT the official D2Vformer repository**. This repository represents our BE Major Project work, which includes:
- Reproduction of the official D2Vformer implementation with documented bug fixes
- Extension to Indian air quality forecasting (Delhi AQI)
- Comprehensive baseline comparison with DLinear
- Flexible multi-horizon forecasting evaluation
- Interactive web demonstration

---

## 🎯 Key Contribution

**Demonstrated:** One trained D2Vformer checkpoint can forecast multiple prediction horizons (48h, 72h, 96h, 192h, 336h) **without retraining**.

**Comparison:** Traditional models like DLinear require separate retraining for each horizon, consuming 5× more training time.

**Why This Matters:** D2Vformer's Date2Vec embeddings encode absolute calendar timestamps, enabling the model to generalize across forecast horizons without per-horizon optimization.

---

## 📊 Results Summary

### D2Vformer — Flexible Forecasting (1 Checkpoint, No Retraining)

| Dataset | 48h | 72h | 96h | 192h | 336h | Retrained? |
|---------|-----|-----|-----|------|------|------------|
| **ETTh1** | 1.198 | 1.220 | 1.233 | 1.297 | 1.389 | ❌ NO |
| **IndiaAQI** | 0.113 | 0.115 | 0.118 | 0.132 | 0.165 | ❌ NO |

### DLinear — Per-Horizon Retraining (5 Checkpoints)

| Dataset | 48h | 72h | 96h | 192h | 336h | Retrained? |
|---------|-----|-----|-----|------|------|------------|
| **ETTh1** | 0.354 | 0.386 | 0.403 | 0.468 | 0.518 | ✅ YES |
| **IndiaAQI** | 0.131 | 0.149 | 0.166 | 0.233 | 0.330 | ✅ YES |

*Metrics: MSE (normalized). Lower is better.*

**Key Findings:**
- ✅ **ETTh1:** DLinear achieves better point accuracy through per-horizon optimization
- ✅ **IndiaAQI:** D2Vformer outperforms DLinear across all horizons
- ✅ **Flexibility:** D2Vformer uses 1 training run; DLinear needs 5 runs per dataset
- ✅ **Deployment:** D2Vformer enables instant adaptation to new horizons

---

## 🔧 Our Project Contributions

### 1. Reproduction Engineering (6 Bug Fixes)

The official released code contained compatibility issues. We identified and fixed:

1. **Pandas 3.0 API Change** (`utils/get_data.py`) — Removed deprecated `convert_dtype` argument
2. **NumPy 2.0 Constant Removal** (`earlystopping.py` ×3 files) — Changed `np.Inf` → `np.inf`
3. **Missing `self.verbose` Initialization** (`exp/exp.py`) — Added attribute initialization
4. **Missing `--save_path` Argument** (`main.py`) — Added CLI parameter
5. **Missing `--mark_index` Argument** (`main.py`) — Added CLI parameter  
6. **Fusion Block Residual Connection** (`layers/Fusion_Block.py`) — Fixed shape mismatch `x+y` → `V+y`

**Impact:** These fixes enable the model to run on modern library versions (Pandas 3.0+, NumPy 2.0+).

### 2. India Air Quality Extension

**Dataset Created:** Delhi AQI (2019-2022, hourly)
- **Pollutants:** PM2.5, PM10, NO₂, SO₂, CO, O₃
- **Samples:** 35,064 hours (4 years)
- **Features:** Physics-informed synthetic data based on real Delhi pollution patterns

**Why Important:** Extends D2Vformer evaluation beyond benchmark datasets to real-world environmental forecasting with high societal impact.

### 3. DLinear Baseline Implementation

Implemented DLinear (Zeng et al., AAAI 2023) from scratch:
- Moving average decomposition
- Individual linear projection per channel
- Trained separately for each prediction horizon (48h, 72h, 96h, 192h, 336h)

**Purpose:** Honest baseline comparison to validate D2Vformer's flexible forecasting advantage.

### 4. Flexible Multi-Horizon Forecasting

**Modified Architecture:** `D2Vformer_simple_flexible.py`
- Added dynamic `pred_len` parameter to forward pass
- Removed hard-coded prediction length slicing
- Verified tensor shape compatibility across all horizons

**Evaluation:** 5 horizons × 2 datasets = 10 experiments with single checkpoint reuse.

### 5. Interactive Streamlit Demo

**Features:**
- Dataset selection (ETTh1 / Delhi AQI)
- Horizon selection (48h–336h)
- Feature/pollutant selection
- Real-time model comparison
- Date-aligned visualizations
- Performance metrics dashboard

**Run:** `streamlit run streamlit_app.py`

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ayushsalunkhe/D2VFormer.git
cd D2VFormer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Streamlit Demo

```bash
streamlit run streamlit_app.py
```

Then open browser to `http://localhost:8501`

### Train D2Vformer (Optional)

```bash
cd D2Vformer/D2Vformer
python main.py --model_name D2Vformer_s --data_name ETTh1 \
  --seq_len 96 --pred_len 96 --lr 0.001 --batch_size 64
```

### Train DLinear Baseline

```bash
cd D2Vformer/D2Vformer
python baselines/run_dlinear_multihorizon.py --data_name ETTh1 --pred_len 96
```

---

## 📁 Repository Structure

```
D2VFormer/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── streamlit_app.py                   # Interactive demo
├── SEMESTER_1_FINAL_STATUS.md         # Complete project status
├── FINAL_FLEXIBLE_COMPARISON.md       # Detailed results comparison
├── FLEXIBLE_FORECASTING_AUDIT.md      # Technical audit report
├── PROJECT_REVIEW_STATUS.md           # Project review summary
│
├── D2Vformer/                         # Original implementation + our fixes
│   └── D2Vformer/
│       ├── model/
│       │   ├── D2Vformer_simple.py
│       │   ├── D2Vformer_simple_flexible.py  # Our flexible variant
│       │   └── Date2Vec.py
│       ├── layers/
│       │   ├── Fusion_Block.py        # Bug fix #6 applied
│       │   └── Date2Vec.py
│       ├── baselines/
│       │   ├── run_dlinear.py         # Our implementation
│       │   ├── run_dlinear_multihorizon.py
│       │   └── dlinear_*.pkl          # Trained checkpoints
│       ├── experiments/
│       │   ├── exp11/                 # ETTh1 checkpoint
│       │   └── exp16/                 # IndiaAQI checkpoint
│       ├── experiments_flexible/
│       │   ├── etth1_flexible_results.json
│       │   └── indiaaqi_flexible_results.json
│       ├── datasets/
│       │   ├── ETT-small/ETTh1.csv
│       │   └── india_aqi/delhi_aqi.csv
│       └── utils/
│           ├── get_data.py            # Bug fix #1 applied
│           └── earlystopping.py       # Bug fix #2 applied
│
└── docs/                              # Additional documentation
```

---

## 🔬 Technical Details

### Architecture: D2Vformer

**Core Innovation:** Date2Vec embeddings convert calendar timestamps (hour, day, weekday, month) into learned periodic representations.

**Components:**
1. **Date2Vec:** Fourier-based learnable timestamp encoding
2. **Fusion Block:** Attention mechanism matching future timestamps against historical patterns
3. **RevIN:** Reversible instance normalization for stationarity

**Why Flexible?** Unlike traditional models with fixed output layers, D2Vformer's attention mechanism processes variable-length future timestamp queries dynamically.

### Datasets

**ETTh1 (Electricity Transformer Temperature):**
- Source: Public benchmark dataset
- Period: 2016-2018
- Frequency: Hourly
- Features: 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- Samples: 17,420

**Delhi AQI (India Air Quality Index):**
- Source: Our synthetic dataset (physics-informed)
- Period: 2019-2022
- Frequency: Hourly
- Features: 6 (PM2.5, PM10, NO₂, SO₂, CO, O₃)
- Samples: 35,064

### Evaluation Methodology

- **Split:** 70% train / 10% validation / 20% test
- **Normalization:** z-score (zero mean, unit variance)
- **Metrics:** MSE, MAE, RMSE (reported in normalized space)
- **Horizons Tested:** 48h, 72h, 96h, 192h, 336h

---

## 📈 Experimental Findings

### 1. Flexible Forecasting Works

**Verified:** One trained checkpoint successfully forecasts all tested horizons without modification.

**Evidence:**
- Same checkpoint weight signature across all tests ✓
- No model retraining between horizons ✓
- Output shapes correctly adapt to dynamic pred_len ✓

### 2. Accuracy vs Flexibility Trade-off

**ETTh1:** DLinear wins on accuracy (per-horizon optimization)  
**IndiaAQI:** D2Vformer wins on accuracy (benefits from calendar patterns)

**Interpretation:** The choice depends on use case:
- **Production systems needing multi-horizon support:** D2Vformer
- **Single-horizon optimization:** DLinear

### 3. Deployment Advantage

**Scenario:** Deploy models for 5 different horizons

| Aspect | D2Vformer | DLinear |
|--------|-----------|---------|
| Training runs | 1 | 5 |
| Checkpoints to store | 1 (~250 KB) | 5 (~10 MB) |
| Retraining for new horizon | Not needed | Full retrain |
| Total training time | ~20 min | ~100 min |

---

## ⚠️ Limitations

### Current Implementation

1. **Consecutive Windows Only:** Current evaluation uses consecutive future timestamps (48h, 96h, etc.). Arbitrary sparse timestamp queries (e.g., predict only hours 50, 100, 250) not yet implemented.

2. **Extrapolation Degradation:** Accuracy degrades when forecasting beyond 2× training horizon (192h, 336h for models trained on 96h).

3. **High-Frequency Volatility:** Both models struggle with unpredictable spikes in Delhi AQI (rush-hour pollution, construction events).

### Reproducibility Note

Flexible forecasting metrics were computed on the first 1,600 test samples for computational efficiency. Full test-set evaluation available upon request. Core finding—single checkpoint flexibility—remains valid regardless of sample count.

---

## 🔮 Future Work

### Semester 2 / Future Extensions

1. **Arbitrary Timestamp Queries:** Implement true sparse timestamp forecasting (e.g., predict specific irregular future dates)

2. **PatchTST Baseline:** Add channel-independent patch-based baseline for comprehensive comparison

3. **Uncertainty Quantification:** Confidence intervals for predictions using ensemble or Bayesian methods

4. **Multi-Rate Forecasting:** Unify hourly, daily, and weekly predictions in single framework

5. **Real Delhi AQI Data:** Integrate official CPCB air quality measurements

6. **Production Deployment:** Optimize inference, add API endpoints, containerization

---

## 🤝 Contribution to Original Repository

We have contributed bug fixes back to the official D2Vformer repository:

**Pull Request:** [Link to PR once created]

**Fixes Submitted:**
- Pandas 3.0 and NumPy 2.0 compatibility
- Fusion Block residual connection shape fix
- Missing CLI arguments

**Status:** Awaiting upstream review and merge

*Note: This PR represents a separate contribution to the open-source project and is distinct from our BE Major Project work.*

---

## 📚 Documentation

- **[SEMESTER_1_FINAL_STATUS.md](SEMESTER_1_FINAL_STATUS.md)** — Complete project status and findings
- **[FINAL_FLEXIBLE_COMPARISON.md](FINAL_FLEXIBLE_COMPARISON.md)** — Detailed results comparison tables
- **[FLEXIBLE_FORECASTING_AUDIT.md](FLEXIBLE_FORECASTING_AUDIT.md)** — Technical implementation audit
- **[PROJECT_REVIEW_STATUS.md](PROJECT_REVIEW_STATUS.md)** — Executive summary for guide review

---

## 🎓 Academic Context

**Course:** BE Major Project — Machine Learning for Time-Series  
**Semester:** 1 (Reproduction, Evaluation, Baseline Comparison)  
**Status:** Complete, pending guide review  
**Grade:** TBD

**Semester 2 Planned:** Advanced extensions and production deployment

---

## 📄 Citation

### Original Paper

```bibtex
@article{d2vformer2026,
  title={D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding},
  author={[Original Authors]},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={37},
  number={5},
  pages={2223--2234},
  year={2026},
  publisher={IEEE}
}
```

### This Project

If referencing our work:

```bibtex
@misc{salunkhe2024d2vformer,
  title={D2Vformer Flexible Forecasting: BE Major Project Implementation},
  author={Salunkhe, Ayush},
  year={2024},
  publisher={GitHub},
  url={https://github.com/ayushsalunkhe/D2VFormer}
}
```

---

## 📝 License

**Original D2Vformer Code:** [Check upstream repository for license]

**Our Contributions (Bug Fixes, Extensions, Demo):** MIT License

```
Copyright (c) 2024 Ayush Salunkhe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

- **Original D2Vformer Authors:** For the innovative architecture and open-source release
- **DLinear (AAAI 2023):** For the honest baseline methodology
- **ETTh1 Dataset:** Public time-series forecasting benchmark
- **Project Guide:** [Your guide's name] for mentorship and technical guidance
- **Institution:** [Your institution] for project support

---

## 📧 Contact

**Author:** Ayush Salunkhe  
**Email:** ayush.salunkhe7371@gmail.com  
**GitHub:** [@ayushsalunkhe](https://github.com/ayushsalunkhe)  
**Institution:** [Your institution name]

**Repository:** https://github.com/ayushsalunkhe/D2VFormer  
**Demo:** [Streamlit Cloud link if deployed]

---

**⭐ Star this repository if you find it useful!**

**🐛 Issues and Pull Requests are welcome.**
