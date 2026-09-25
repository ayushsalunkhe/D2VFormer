# Reproducibility Report: TC-D2Vformer

**Project:** Temperature-Controlled D2Vformer (TC-D2Vformer)  
**Date:** 2026-09-22  
**Standard:** Open Science & Academic Rigor Protocol

---

## 1. Environment & Software Specifications

- **Operating System:** Windows 10/11 x86_64 / Linux (Google Colab T4 GPU Environment)
- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch >= 2.0.0
- **Supporting Libraries:**
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `scipy >= 1.10.0`
  - `matplotlib >= 3.7.0`
  - `tqdm >= 4.65.0`

---

## 2. Model Parameterization & Horizon Invariance

The final model (`TCD2Vformer`) was mathematically constructed and unit-tested to guarantee complete horizon independence:

- **Lookback Window ($L$):** 96 steps
- **Number of Variables ($C$):** 7 (ETT datasets), 8 (Exchange Rate)
- **Model Dimension ($d_{	ext{model}}$):** 128
- **Feedforward Dimension ($d_{ff}$):** 256
- **Harmonic Frequencies ($k_{	ext{freq}}$):** 16
- **Dropout:** 0.05
- **Trainable Parameters:** Exactly **44,021** (for $C=7$) / **44,150** (for $C=8$)
- **Horizon-Dependent Parameters:** **0 (Zero)**

### Invariance Proof
Unit tests in `tests/test_tcd2vformer.py` verify that evaluating forecast horizons $O \in \{24, 48, 96, 192, 336, 720\}$ on the same model instance leaves the model parameter shapes and parameter checksum identical before, during, and after inference.

---

## 3. Pre-Registered Random Seeds

All experiments were executed across 3 independent random seeds:
- `seed = 42`
- `seed = 43`
- `seed = 44`

`torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`, and `torch.cuda.manual_seed_all(seed)` were strictly set before data loading, model initialization, and training.

---

## 4. Zero Test-Set Lookahead Protocol

To eliminate data leakage:
1. Data splits are strictly chronological: 60% Train, 20% Validation, 20% Test.
2. Feature standardization ($\mu, \sigma$) is computed **exclusively on the training partition**.
3. Temperature candidates $	au \in \{0.5, 1.0, 2.0, 4.0\}$ are evaluated **only on the validation partition at $O_{	ext{train}}=48$**.
4. The selected $	au^*$ is locked into `validation_selection.csv` before the test set is touched.
5. The locked test set is evaluated **once** across horizons $O \in \{24, 48, 96, 192, 336, 720\}$.

---

## 5. Artifact Integrity & SHA-256 Checksums

| File | SHA-256 Checksum |
|---|---|
| `results/final/summary.csv` | 35c079e4016024b411d286abc77814c15cfe2cd741c7515f140ee38fe4931811 |
| `results/final/summary.json` | 6774b02f64c8ba7c038623c9a8abab8596782573ae42242fd261eedbb3fedd3d |
| `models/pure_d2vformer.py` | fadd77ae6f4ed62063279251405b9d738e78bf8088f3475aaa0704da389a61c3 |
| `models/temperature_d2vformer.py` | a40280d7c6b263f3ad2efea10b1f915105f52aee66c7e0804cf53c361d1bf332 |
| `experiments/pipeline.py` | 145a382d25754939d354a63acbc23f65bbd0ee9ec0a5d53a000970ddb71145b0 |

---

## 6. Commands to Reproduce

### Run Unit Tests
```bash
python -m unittest tests/test_tcd2vformer.py
python tests/test_temperature_horizon_independence.py
```

### Run Full Pipeline
```bash
python -c "from experiments.pipeline import select_temperature, evaluate_locked_test; print(select_temperature('ETTh1', 42))"
```
