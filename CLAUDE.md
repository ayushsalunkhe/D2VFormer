# BE Major Project — D2Vformer (Flexible Time-Series Forecasting)

Two-semester BE final-year major project. Claude Code acts as main developer; Ayush is the student presenting to guide/examiners.

## Selected paper
- **D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding**
- IEEE Transactions on Neural Networks and Learning Systems, Vol. 37, No. 5, pp. 2223–2234, May 2026
- DOI: 10.1109/TNNLS.2025.3630792 · arXiv:2409.11024
- Official code: https://github.com/yumiyumo/D2Vformer
- Datasets: six free Autoformer benchmarks (ETT, Electricity, Weather, Traffic, Exchange, ILI) via https://github.com/thuml/Autoformer links

## Core idea (elevator pitch)
Existing forecasters predict a fixed "next N steps" window. D2Vformer's Date2Vec module embeds actual calendar timestamps (hour/day/weekday/month periodicity) conditioned on input features; an attention fusion block matches future-date embeddings against the past, enabling prediction at **arbitrary, non-adjacent future timestamps without retraining**.

## Two-semester roadmap
**Semester 1:** (1) reproduce paper results on ETT/Weather; (2) apply to a new India-relevant dataset (city electricity load / Delhi AQI); (3) benchmark vs DLinear, PatchTST + Date2Vec ablation; (4) Streamlit demo v1 (pick date range → live forecast graph).
**Semester 2:** (5) uncertainty quantification (quantile loss / MC-dropout confidence bands); (6) anomaly-detection mode (flag reality-vs-forecast deviations); (7) stretch: free-tier LLM (Groq/Gemini) auto-generates plain-English forecast reports; (8) final dashboard + black book + paper-style write-up.

Every phase ends with a chart or live demo for guide meetings — that's deliberate.

## Toolstack
Python 3.11, PyTorch, pandas, scikit-learn; matplotlib/Plotly; Streamlit; W&B free tier for experiment tracking; free Colab/Kaggle T4 for training sweeps; zero budget throughout.

## Key documents (in C:\Users\ayush\Downloads)
- `IEEE_Paper_Selection_Report.pdf` — 15-paper comparison, top 5 ranking
- `D2Vformer_Project_Plan.pdf` — full roadmap shared with group
- `Literature_Survey_Top5_and_Base_Papers.pdf` — top 5 + 12 base papers (title/authors/abstract/limitations/inference)

## Base papers for lit survey (D2Vformer lineage)
Transformer (NeurIPS 2017), Time2Vec (arXiv 2019), Autoformer (NeurIPS 2021, defines the 6-benchmark suite), DLinear (AAAI 2023, honest baseline), PatchTST (ICLR 2023, strongest Transformer baseline).

## Status
- [x] Paper selected and verified (journal/DOI/code/datasets)
- [x] Roadmap + literature survey documents delivered
- [x] Clone D2Vformer repo, set up environment (venv/ at project root, Python 3.12, CPU torch)
- [x] Datasets: ETT/exchange/illness bundled in repo; junction `D2Vformer/D2Vformer/datasets` → `../datasets`
- [x] First reproduction run: D2Vformer_s, ETTh1 96→96, early-stopped ep40 — Test MSE 0.7067 / MAE 0.6186 (paper ~0.44; single config, no sweep yet)
- [x] Forecast graphs in `results/` via `D2Vformer/D2Vformer/plot_forecast.py`
- [x] LR × T2V_outmodel sweep (official script grid) — best: lr=0.001, T2V=36, MSE=0.6982 (Colab T4)
- [x] ETTh1 checkpoint downloaded to Downloads/ETTh1_best_model.pkl; copied to experiments/exp11
- [x] Forecast graphs generated: results/ETTh1_pred96_forecast.png + windows 500/1500
- [x] Full D2Vformer model: two additional release bugs confirmed (4D patched x_var vs 3D einsum + linear_out never called). Confirmed D2Vformer_s as project variant. Documented as reproduction engineering.
- [x] India AQI dataset created: datasets/india_aqi/delhi_aqi.csv + delhi_mark.csv (35064 hourly rows, 2019-2022, 6 pollutants: PM2.5/PM10/NO2/SO2/CO/O3)
- [x] IndiaAQI registered in exp.py; plot_india_aqi.py written
- [x] D2Vformer_Colab.ipynb updated — Step 6b trains IndiaAQI sweep; Step 7 evaluates; Step 8 saves forecast graph
- [x] Run IndiaAQI sweep on Colab (Step 6b) and download IndiaAQI_best_model.pkl
- [x] Forecast graphs generated: results/IndiaAQI_forecast.png (MSE 0.2212, MAE 0.3072 on Colab; local test MSE ~0.1182)
- [x] Baseline comparison: DLinear vs D2Vformer on ETTh1 and IndiaAQI
  - ETTh1: DLinear MSE=0.4033 (WINS) vs D2Vformer MSE=0.6982
  - IndiaAQI: DLinear MSE=0.1658 (WINS) vs D2Vformer MSE=0.2212
  - Graphs: results/comparison_bar.png + comparison_forecast.png
  - **Key finding:** DLinear (simple linear baseline) outperforms D2Vformer on both datasets — suggests the Date2Vec calendar attention may not be the right inductive bias for these domains, or that hyperparameters need further tuning. This is an honest research finding for the black book.
- [x] Streamlit demo v1 — Interactive web app deployed locally
  - URL: http://localhost:8502 (or 8501)
  - Features: dataset selection (ETTh1/IndiaAQI), horizon picker (96/192/336h), model comparison (D2Vformer vs DLinear), interactive window selection
  - Files: streamlit_app.py, requirements.txt, DEMO_README.md

## Repo fixes applied (document in black book as "reproduction engineering")
- pandas 3.0: removed obsolete 2nd arg in `Series.apply` (utils/get_data.py)
- numpy 2.0: `np.Inf` → `np.inf` (all three earlystopping.py)
- exp/exp.py: added missing `self.verbose` init
- main.py: added missing `--save_path` and `--mark_index` (default [0,1,2,3] = hour/weekday/day/month) args
- layers/Fusion_Block.py: alias `Fusion_Block = Fusion_Block_s` (class renamed in release); fixed residual `x + y` → `V + y` (released shapes could never broadcast — genuine release bug)
- Run command template: see scripts/D2Vformer_s_train.sh + `--desc X --T2V_outmodel 64 --mark_index 0 1 2 3`
- Chinese font (SimHei) matplotlib warnings are harmless; filter with `grep -v findfont`
