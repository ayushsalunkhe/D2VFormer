# Experimental Protocol

> **Document type:** Reproducibility protocol  
> **Project:** BE Computer Engineering Major Project 2025–26

---

## Environment

```
Python  : 3.10+
PyTorch : 2.0+
NumPy   : 1.24+
Pandas  : 2.0+
```

Install: `pip install -r requirements.txt`

---

## Dataset Preparation

All datasets are placed in `datasets/`:

| Dataset | Source | Frequency | Channels | Samples |
|---------|--------|-----------|----------|---------|
| ETTh1 | ETT benchmark | Hourly | 7 | 17,420 |
| ETTh2 | ETT benchmark | Hourly | 7 | 17,420 |
| ETTm1 | ETT benchmark | 15-min | 7 | 69,680 |
| Exchange Rate | Open source | Daily | 8 | 7,588 |

**Split:** chronological 60/20/20 train/val/test. No shuffling, no lookahead.

---

## Reproducibility Steps

### Step 1: Train Baseline (PureD2Vformer, τ=1.0)

```bash
python experiments/pipeline.py     --dataset ETTh1     --tau 1.0     --seed 42     --epochs 10     --patience 3     --lr 0.001     --batch_size 64     --lookback 96     --train_horizon 48
```

Repeat for seeds 43, 44 and all τ ∈ {0.5, 1.0, 2.0, 4.0}.

### Step 2: Temperature Selection

```bash
python experiments/pipeline.py     --mode select     --dataset ETTh1     --seed 42
```

Reads validation MSE from saved checkpoints, outputs selected τ*.

### Step 3: Zero-Shot Test Evaluation

```bash
python experiments/pipeline.py     --mode evaluate     --dataset ETTh1     --seed 42     --tau <selected_tau>     --horizons 24 48 96 192 336 720
```

### Step 4: Verify Parameter Invariance

```bash
python tests/test_tcd2vformer.py
```

Outputs SHA-256 checksums confirming params(O=24) = params(O=720).

---

## Checkpoint Naming Convention

```
checkpoints/temp_d2v_<dataset>_tau<tau>_seed<seed>.pt
```

Example: `temp_d2v_ETTh1_tau4.0_seed42.pt`

---

## Results Location

| File | Contents |
|------|----------|
| `results/final/summary.csv` | All 72 (dataset × seed × horizon) test results |
| `results/final/summary.json` | Structured metadata + dataset-level summaries |
| `results/final/checksums.json` | SHA-256 parameter checksums |
| `results/tables/master_model_comparison.csv` | 5-model benchmark table |

---

## Random Seed Control

All PyTorch and NumPy random seeds are set before model initialization:
```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
```

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `L` | Lookback window (96) |
| `O` | Forecast horizon |
| `O_train` | Training horizon (48) |
| `τ` | Temperature scalar |
| `τ*` | Validation-selected temperature |
| `H_norm` | Normalized attention entropy |
| `N_eff` | Effective number of attended positions |
