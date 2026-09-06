# D2Vformer Forecasting Demo — Instructions & Quickstart

**BE Major Project** — D2Vformer Time-Series Forecasting  
Flexible Multi-Horizon Forecasting Demonstration (Semester 2)

---

## Quickstart

Run the following command from the project root directory:

```powershell
.\venv\Scripts\streamlit run streamlit_app.py
```

Or using python module syntax:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The web interface will open in your browser at `http://localhost:8501`.

---

## Verification & Features

1. **Dataset Selection:**
   - **ETTh1 (Electricity Transformer Temperature):** 7 electrical & oil temperature features (`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`).
   - **Delhi AQI (India Air Quality):** 6 major pollutant features (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`).

2. **Flexible Horizon Selection:**
   - Use the slider to pick from: **48h, 72h, 96h, 192h, 336h**.
   - **D2Vformer:** Uses the **exact same single checkpoint** across all horizons without retraining.
   - **DLinear:** Automatically switches to the corresponding horizon-specific checkpoint (5 models per dataset).

3. **Browse Windows & Timestamps:**
   - Change `Window Index` to inspect different timeframes in the test set.
   - Exact calendar dates and seasons are calculated from the source data timestamps.

4. **Offline Local Execution:**
   - All models, datasets, and visualizations run 100% locally from the workspace without needing Google Colab, external APIs, or network calls.

---

## Directory Structure Overview

```
Major Project/
├── streamlit_app.py                          # Upgraded interactive demo
├── PROJECT_REVIEW_STATUS.md                  # Comprehensive review report
├── FINAL_FLEXIBLE_COMPARISON.md              # Full empirical comparison
├── results/
│   └── flexible/                             # Review-ready plots
│       ├── mse_vs_horizon_ETTh1.png
│       ├── mse_vs_horizon_IndiaAQI.png
│       ├── mae_vs_horizon_ETTh1.png
│       ├── mae_vs_horizon_IndiaAQI.png
│       └── training_cost_comparison.png
└── D2Vformer/D2Vformer/
    ├── experiments_flexible/                 # Consolidated results & JSON files
    │   ├── final_flexible_comparison.csv
    │   └── dlinear_all_results_consolidated.json
    ├── baselines/                            # 10 DLinear model checkpoints
    └── experiments/                          # Trained D2Vformer checkpoints
```
