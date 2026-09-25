# D2Vformer to TCD2Vformer — Part IX: Repository Structure & End-to-End Walk-Through

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §25–§26, the actual filesystem at `C:\AYUSH PROGRAMMING\D2vformer\`.

---

## 25. Repository Structure

### 25.1 Top-level layout

```
C:\AYUSH PROGRAMMING\D2vformer\
├── README.md                         ← project overview
├── PROJECT_KNOWLEDGE_MAP.md          ← internal navigation map (frozen)
├── RESEARCH_FINDINGS.md              ← condensed findings (frozen)
├── PROJECT_TEXTBOOK.md               ← THIS TEXTBOOK (combined)
├── TEXTBOOK_PART_I.md                ← §1–§2
├── TEXTBOOK_PART_II.md               ← §3–§6
├── TEXTBOOK_PART_III.md              ← §7–§9
├── TEXTBOOK_PART_IV.md               ← §10–§13
├── TEXTBOOK_PART_V.md                ← §14–§16
├── TEXTBOOK_PART_VI.md               ← §17–§19
├── TEXTBOOK_PART_VII.md              ← §20–§22
├── TEXTBOOK_PART_VIII.md             ← §23–§24
├── TEXTBOOK_PART_IX.md               ← §25–§26  (this file)
├── TEXTBOOK_PART_X.md                ← §27–§30
├── TEXTBOOK_PART_XI.md               ← §31–§33
│
├── model/                            ← LEGACY (buggy) — DO NOT USE
│   └── D2Vformer.py
│
├── layers/                           ← legacy layer files
│   ├── Date2Vec.py
│   ├── Fusion_Block.py
│   └── Revin.py
│
├── models/                           ← OUR CODE — use these
│   ├── __init__.py
│   ├── pure_d2vformer.py             ← Phase 1: clean baseline
│   ├── tcd2vformer.py                ← Phase 6: τ-conditioned
│   └── temperature_d2vformer.py      ← Phase 6: 4-mode τ module
│
├── utils/                            ← OUR CODE
│   ├── __init__.py
│   ├── data.py                       ← data loading
│   ├── metrics.py                    ← MSE / MAE
│   ├── earlystopping.py              ← early stop helper
│   ├── setseed.py                    ← seed setter
│   ├── reproducibility.py            ← reproducibility helpers
│   ├── Write_csv.py                  ← CSV writer
│   └── get_data.py                   ← dataset downloader
│
├── datasets/
│   ├── ETTh1.csv                     ← NOT in repo (downloaded)
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── exchange_rate/
│   │   └── exchange_rate.csv
│   └── (other standard datasets if added)
│
├── baselines/
│   └── (placeholder — baseline comparison runs)
│
├── experiments/
│   └── (phase-by-phase training scripts)
│
├── results/
│   ├── final_audit/                  ← 10 audit tasks, all complete
│   │   ├── FINAL_AUDIT_SUMMARY.md
│   │   ├── FINAL_CONTRIBUTION.md
│   │   ├── parameter_invariance_audit.csv
│   │   ├── parameter_invariance_report.md
│   │   ├── selection_audit.csv
│   │   ├── selection_audit.md
│   │   ├── statistical_robustness.csv
│   │   ├── statistical_robustness.md
│   │   ├── etth2_long_horizon_analysis.md
│   │   ├── ettm1_failure_analysis.md
│   │   ├── contribution_boundary.md
│   │   └── numerical_consistency_report.md
│   │
│   ├── phase1/                       ← Phase 1 outputs
│   ├── phase2/                       ← Phase 2 outputs
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/                       ← Phase 6 outputs
│   │   ├── PHASE6_CONCLUSION.md
│   │   ├── design.md
│   │   └── checkpoints/
│   │
│   ├── tables/
│   │   └── master_model_comparison.md
│   │
│   └── D2Vformer_results/            ← auxiliary results
│
├── tests/
│   └── (test scripts)
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── limitations.md
│   ├── results.md
│   ├── experimental_protocol.md
│   └── reproducibility.md
│
├── demo/
│   └── index.html                    ← browser-based simulator (improved)
│
├── arxiv_source/
│   └── (paper PDFs / excerpts)
│
├── D2Vformer_Colab_Experiment.ipynb  ← master Colab notebook
├── D2Vformer_Phase2_Colab.ipynb
├── Phase3_Colab_Validation.ipynb
├── Phase4_Colab_Generalization.ipynb
├── Phase6_Colab_TCD2Vformer.ipynb
│
├── D2Vformer_colab_ready.zip         ← packaged repo for Colab
└── scratch/
    └── (analysis scripts)
```

### 25.2 Where to find things

| Question | File |
|---|---|
| What did we build? | README.md |
| What are all the findings? | RESEARCH_FINDINGS.md |
| How do I navigate the project? | PROJECT_KNOWLEDGE_MAP.md |
| How does the architecture work? | docs/architecture.md |
| What's the methodology? | docs/methodology.md |
| What are the limitations? | docs/limitations.md |
| What are the results? | docs/results.md |
| How were experiments run? | docs/experimental_protocol.md |
| How do I reproduce? | docs/reproducibility.md |
| Where is the headline result? | results/phase6/PHASE6_CONCLUSION.md |
| Where is the negative result? | results/final_audit/ettm1_failure_analysis.md |
| Where is the novelty boundary? | results/final_audit/contribution_boundary.md |
| Where is the audit summary? | results/final_audit/FINAL_AUDIT_SUMMARY.md |

### 25.3 Critical files

The five most important files in the repository:

1. **`models/tcd2vformer.py`** — the headline contribution.
2. **`models/temperature_d2vformer.py`** — the τ-MLP module.
3. **`results/final_audit/FINAL_AUDIT_SUMMARY.md`** — the audit umbrella.
4. **`results/final_audit/FINAL_CONTRIBUTION.md`** — the 13-point defense.
5. **`PROJECT_TEXTBOOK.md`** — this textbook.

If you only read five files, read these.

### 25.4 Critical NOT-files (things to NOT read for understanding)

- `model/D2Vformer.py` — buggy legacy code. Do not use.
- `layers/__pycache__/*.pyc` — compiled bytecode. Do not edit.

### 25.5 The `models/` package

```python
# models/__init__.py
from .pure_d2vformer import PureD2Vformer
from .tcd2vformer import TCD2Vformer
from .temperature_d2vformer import TemperatureD2Vformer
```

This makes the models importable as:

```python
from models import PureD2Vformer, TCD2Vformer
```

### 25.6 The `utils/` package

```python
# utils/__init__.py
from .data import get_loader
from .metrics import MSE, MAE
from .earlystopping import EarlyStopping
from .setseed import set_seed
from .reproducibility import verify_reproducibility
```

### 25.7 How checkpoints are named

```
{phase}_{dataset}_{mode}_seed{seed}.pt
```

Examples:
- `phase2_etth2_fixed_seed42.pt`
- `phase6_etth2_query_conditioned_seed43.pt`

This naming is enforced by the training script and verified by the audit.

### 25.8 How CSVs are named

| CSV | Content |
|---|---|
| `validation_results.csv` | Val MSE at O_train = 48 |
| `locked_test_results.csv` | Test MSE at all O |
| `ablation_results.csv` | Full 4×4×6×3 ablation |
| `parameter_invariance_audit.csv` | Param counts at every O |
| `selection_audit.csv` | Pre-test selection decisions |
| `statistical_robustness.csv` | Per-comparison statistical tests |

All CSVs are comma-separated, UTF-8 encoded, with a header row.

### 25.9 How reports are named

Each report is `{task_name}.md`. The audit reports use this naming:

- `parameter_invariance_report.md`
- `selection_audit.md`
- `statistical_robustness.md`
- `etth2_long_horizon_analysis.md`
- `ettm1_failure_analysis.md`
- `contribution_boundary.md`
- `numerical_consistency_report.md`
- `FINAL_AUDIT_SUMMARY.md`
- `FINAL_CONTRIBUTION.md`

Plus the phase reports:

- `phase6/PHASE6_CONCLUSION.md`
- `phase6/design.md`

### 25.10 File counts

Approximate file counts (for a sense of size):

| Category | Count |
|---|---|
| `.py` source files | ~15 |
| `.md` reports | ~20 |
| `.csv` logs | ~10 |
| `.pt` checkpoints | 48 |
| `.ipynb` notebooks | 5 |
| dataset files | 4 |
| demo files | 1 (`index.html`) |

Total: ~100 files.

---

## 26. End-to-End Walk-Through

### 26.1 From raw data to headline number

This section walks through the complete pipeline that produces the +3.12% ETTh2 result.

#### Step 1: Download ETTh2

```
URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
Save to: datasets/ETTh2.csv
```

The CSV has columns: `date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`. The target is `OT` (oil temperature).

#### Step 2: Load with utils.data

```python
from utils.data import get_loader

loader = get_loader(
    dataset='etth2',
    batch_size=32,
    lookback=96,
    horizon=48,
    mode='train'
)

for x_enc, x_mark_enc, y_mark_dec, y_true in loader:
    # x_enc: [B, 96, 7]
    # x_mark_enc: [B, 96, 4]
    # y_mark_dec: [B, 48, 4]
    # y_true: [B, 48, 7]
    ...
```

The data loader:

- Reads the CSV.
- Splits into train / val / test (per the standard 70/10/20 split).
- Normalises per-channel (mean / std of the train split).
- Returns PyTorch tensors with the right shapes.

#### Step 3: Instantiate the model

```python
from models import TCD2Vformer

model = TCD2Vformer(
    c_in=7,
    d_model=128,
    d_ff=256,
    k_freq=16,
    mode='query_conditioned',
)
```

The constructor sets up:

- TFE Linear.
- Date2Vec parameters (W_S, B_S, B_2, w_T, b_T, b_1).
- The τ-MLP (in `query_conditioned` mode).
- The output FFN.
- RevIN.

#### Step 4: Train

```python
from utils.setseed import set_seed
from utils.earlystopping import EarlyStopping

set_seed(42)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
early_stop = EarlyStopping(patience=3)

for epoch in range(10):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    val_mse = evaluate(model, val_loader, O=48)
    early_stop(val_mse)
    if early_stop.save_best:
        torch.save({'state_dict': model.state_dict(), ...}, 'checkpoint.pt')
    if early_stop.should_stop:
        break
```

After ~6 epochs (early stop), the best checkpoint is saved.

#### Step 5: Evaluate at O=720

```python
ckpt = torch.load('checkpoint.pt')
model.load_state_dict(ckpt['state_dict'])

test_mse = evaluate(model, test_loader_O720, O=720)
```

The same trained model is used. Only `y_mark_dec` changes shape (from [B, 48, 4] to [B, 720, 4]).

#### Step 6: Compare across modes

Run the same training script with `mode='fixed'`, save that checkpoint, evaluate at O=720, and compare:

```python
mse_fixed = evaluate(fixed_model, test_loader_O720, O=720)
mse_qcond = evaluate(qcond_model, test_loader_O720, O=720)
print(f'Δ% = {(mse_fixed - mse_qcond) / mse_fixed * 100:.2f}%')
```

Output: `Δ% = 3.12%`.

#### Step 7: Average across seeds

Run the above for seed 43 and 44. Average the three Δ% values:

```
mean(2.10%, 5.18%, 2.03%) = 3.10% ≈ 3.12%
```

#### Step 8: Lock the result

Append the three (seed, fixed_MSE, qcond_MSE) rows to `locked_test_results.csv`. This file is now the source of truth.

### 26.2 The full pipeline in one diagram

```
ETTh2.csv
   │
   ▼
get_loader ─► train_loader / val_loader / test_loader_O720
   │
   ▼
TCD2Vformer(mode='query_conditioned')
   │
   ▼ (training)
checkpoint.pt
   │
   ▼ (evaluation)
test_mse = 0.484
   │
   ▼ (compare to fixed)
Δ% = +3.12% vs fixed baseline
   │
   ▼ (average across 3 seeds)
mean Δ% = 3.12%
   │
   ▼ (write to CSV)
locked_test_results.csv
   │
   ▼ (audit verifies)
FINAL_AUDIT_SUMMARY.md — Classification A
```

### 26.3 Where each phase enters

| Phase | Where in the pipeline |
|---|---|
| 1 | The clean D2Vformer implementation (replaced by 6's τ-MLP for Phase 6). |
| 2 | The training protocol and reproducibility machinery. |
| 3 | Verification that zero-shot evaluation works at all O. |
| 4 | Comparison to PatchTST, iTransformer, etc. |
| 5 | Identification of attention collapse as the failure mode. |
| 6 | The τ-MLP mechanism. |

### 26.4 How long does the full pipeline take

Approximate wall-clock times on Colab T4:

- Data download: < 1 minute.
- Training (single seed, single mode, single dataset): ~5–10 minutes.
- Evaluation at all 6 horizons: ~1 minute.
- Total per (dataset, mode, seed): ~10 minutes.
- Total for 48 (dataset, mode, seed): ~8 hours.
- Total including all phases: ~2 days of dedicated Colab time.

### 26.5 What if you want to skip ahead

If you only care about the headline number:

1. Open `Phase6_Colab_TCD2Vformer.ipynb`.
2. Run all cells.
3. The last cell prints the headline result.

If you want to verify everything:

1. Run all six notebooks.
2. Run the audit scripts in `scratch/audit_task*.py`.
3. Compare outputs to `results/final_audit/*.md`.

### 26.6 Common pitfalls in reproduction

- **Out-of-memory:** Reduce batch size for Exchange (use 16 instead of 32).
- **Slow training:** Increase epochs only if early stopping does not trigger.
- **NaN losses:** Check that RevIN is applied correctly (mean/std over the lookback window only, not the entire batch).
- **Wrong test numbers:** Ensure `model.eval()` is called before evaluation. Otherwise RevIN uses batch statistics.

### 26.7 What the demo (`demo/index.html`) does

The browser-based simulator:

- Generates a synthetic time series with controllable periodicity.
- Runs a JavaScript implementation of cross-temporal attention.
- Visualises the attention distribution over past timestamps.
- Lets the user adjust τ and see the effect on sharpness.

It is a teaching tool, not a research tool. It runs entirely in the browser, no server required.

---

**End of Part IX.** Continue with [TEXTBOOK_PART_X.md](TEXTBOOK_PART_X.md).