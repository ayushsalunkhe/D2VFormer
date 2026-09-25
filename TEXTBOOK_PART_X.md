# D2Vformer to TCD2Vformer — Part X: Defenses — 30 Seconds to 50 Vivas

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §27–§30, all prior parts.
> **Purpose:** give you a complete set of pre-prepared answers at four escalating levels of detail.

---

## 27. The 30-Second Pitch

If a panelist asks "What is your project?", say this:

> "D2Vformer is a time-series forecasting model that uses a Date2Vec embedding to attend across time. The original implementation has a bug — its parameter count scales with the forecast horizon, breaking zero-shot forecasting. We reconstructed the model from scratch to fix the bug, then extended it with a learned softmax temperature conditioned on the future calendar. On ETTh2 at a 30-day horizon, this conditioning improves MSE by 3.12% across all three random seeds. We also identified a failure mode on ETTm1 where the conditioning over-sharpens attention, and we explain it as a boundary condition."

That's about 30 seconds spoken.

### 27.1 The 30-second pitch keywords

The pitch must hit these keywords (panelists will look for them):

- "time series forecasting"
- "Date2Vec"
- "cross-temporal attention"
- "softmax temperature"
- "horizon independence"
- "ETTh2"
- "3.12%"
- "boundary condition"

If the pitch hits all eight, it's complete.

### 27.2 The 30-second pitch — annotated

```
"D2Vformer"                                    ← project name
"is a time-series forecasting model"            ← domain
"that uses a Date2Vec embedding"                ← key technique
"to attend across time."                        ← mechanism
"The original implementation has a bug"         ← honest critique
" — its parameter count scales with"            ← specifics
"the forecast horizon,"                         ← specifics
"breaking zero-shot forecasting."               ← why it matters
"We reconstructed the model from scratch"       ← our work 1
"to fix the bug,"                               ← fix
"then extended it with a learned softmax"       ← our work 2
"temperature conditioned on the future calendar."← mechanism
"On ETTh2 at a 30-day horizon,"                 ← specific result
"this conditioning improves MSE by 3.12%"       ← specific number
"across all three random seeds."                ← robustness
"We also identified a failure mode on ETTm1"    ← honesty
"where the conditioning over-sharpens"          ← mechanism
"attention, and we explain it as a"             ← framing
"boundary condition."                           ← framing
```

### 27.3 The 30-second pitch — common mistakes

- ❌ "We made the model better." (Too vague.)
- ❌ "We improved accuracy by 3%." (No dataset, no horizon, no context.)
- ❌ "We got statistical significance." (We did not — n = 3 is too small.)
- ❌ "We beat SOTA." (We did not — PatchTST/iTransformer are still better.)
- ❌ "We fixed a paper." (We fixed an implementation, not the paper.)

The right pitch is **specific, honest, and modest**.

---

## 28. The 2-Minute Pitch

If you have 2 minutes (typical thesis introduction):

> **"Our project is titled TCD2Vformer — Temporal-Conditioned Date2Vecformer. It's an extension of the D2Vformer architecture for time-series forecasting. The original D2Vformer uses a learned harmonic embedding of timestamps called Date2Vec, combined with cross-temporal attention to forecast at arbitrary horizons. The contribution of our project is two-fold.**
>
> **First, we noticed that the official D2Vformer repository had a bug: its output projection was a linear layer whose width equaled the forecast horizon. This meant the parameter count scaled with the horizon, which violates the paper's claim of horizon-independent forecasting. We reconstructed the model from scratch, removing this dependency. Our reconstructed model has exactly 44,021 parameters regardless of whether you forecast 24 hours or 720 hours ahead. This is verified across 96 dataset-mode-horizon combinations.**
>
> **Second, we identified that the original model uses a fixed softmax temperature of 1.0 in cross-temporal attention. We hypothesized that the optimal temperature depends on the prediction task. We added a small MLP that computes the temperature from the future Date2Vec embedding, allowing the model to learn per-query temperature.**
>
> **We evaluated four variants of this mechanism on four datasets — ETTh1, ETTh2, ETTm1, and Exchange — across six horizons — 24 to 720 steps — with three random seeds.**
>
> **On ETTh2 at the 720-step horizon, our query-conditioned variant improves MSE by 3.12% over the fixed-temperature baseline. All three seeds improve by more than 2%. Cohen's d is 1.67, indicating a large effect size. However, with only three seeds, we cannot claim classical statistical significance at p < 0.05.**
>
> **On ETTm1, the conditioning hurts performance at long horizons by 10%. We explain this mechanistically: the 15-minute sampling rate combined with a 24-hour lookback gives the model only one full daily cycle of past data. The learned temperature collapses to around 0.18, which over-sharpens the attention distribution. This is an empirical boundary condition, not a bug.**
>
> **Our contributions are: the reconstruction, the four-mode taxonomy of temperature mechanisms, the τ-MLP mechanism itself, the empirical boundary on ETTm1, and a fully reproducible end-to-end pipeline."**

That is roughly 2 minutes spoken at a measured pace.

### 28.1 The 2-minute pitch — checklist

| Required element | Included? |
|---|---|
| Project name | ✓ |
| Domain (time-series forecasting) | ✓ |
| Inherited technique (Date2Vec) | ✓ |
| Our fix (horizon-independence) | ✓ |
| Our extension (τ-MLP) | ✓ |
| Datasets and horizons | ✓ |
| Specific result (+3.12% ETTh2 O=720) | ✓ |
| Statistical caveat (n = 3, p ≈ 0.102) | ✓ |
| Negative result (ETTm1) | ✓ |
| Mechanistic explanation | ✓ |
| Reproducibility | ✓ |

### 28.2 The 2-minute pitch — when to use

- Opening slide of the thesis defense.
- Abstract of the final report.
- First response to "Tell me about your project."

---

## 29. The 5-Minute Pitch

If you have 5 minutes (e.g., a conference-style talk or a detailed panel question):

> **"TCD2Vformer: Temporal-Conditioned Date2Vecformer. Major Project, BE Computer Engineering.**
>
> **[0:00–0:30] Problem.** Time-series forecasting matters for energy, finance, transport. Most models are trained for a single horizon. D2Vformer (Wang et al., 2024) proposes flexible-horizon forecasting using a Date2Vec temporal embedding. The official implementation, however, contains a `Linear(d_model, O)` projection, which means the parameter count scales with the forecast horizon. The paper's main conceptual claim — train once, query anywhere — is technically violated.
>
> **[0:30–1:30] Phase 1: Reconstruction.** We re-implemented D2Vformer from scratch, removing the parameter-scaling projection. Our `PureD2Vformer` model has 44,021 trainable parameters, constant across horizons. We verified this on 96 (dataset, mode, horizon) combinations.
>
> **[1:30–3:00] Phase 6: TCD2Vformer.** Beyond the bug fix, we identified that the original model uses a fixed softmax temperature. We hypothesised that the optimal temperature depends on the prediction task — specifically, on the calendar context of the future query. We added a small MLP that computes a per-query temperature from the future Date2Vec embeddings.
>
> **We systematically ablated four temperature modes:** fixed (baseline), learned_global (single learned scalar), temporal_context (per-batch from mean of future Date2Vec), and query_conditioned (per-query from each future Date2Vec).
>
> **On ETTh2, query_conditioned monotonically improves MSE as horizon grows:**
> - O=24: +0.58%
> - O=192: +2.79%
> - O=720: +3.12%
>
> **All 3 random seeds show > 2% improvement at O=720.** Cohen's d = 1.67 — large effect size. However, n = 3 is too small for p < 0.05 significance (p ≈ 0.102).
>
> **[3:00–4:00] Phase 6: Negative result.** On ETTm1 at O=720, query_conditioned hurts by 10%. We explain this mechanistically: the dataset has 15-minute sampling, giving a 24-hour effective lookback. For a 7.5-day forecast, the model has too few past analogues. The learned τ collapses to ~0.18, over-sharpening attention. This is an empirical boundary condition for our mechanism.
>
> **[4:00–5:00] Reproducibility and conclusion.** All 48 checkpoints are SHA-256 verified. All 288 test evaluations are locked. The full pipeline runs in ~8 hours on Colab. We do not claim universal improvement — we claim a specific improvement with a specific boundary.
>
> **In summary:** we contribute a clean reconstruction of D2Vformer, a learnable temperature mechanism, a four-mode ablation, an empirical boundary explanation, and a fully reproducible end-to-end pipeline."**

### 29.1 The 5-minute pitch — slide map

| Time | Slide | Content |
|---|---|---|
| 0:00 | Title | TCD2Vformer, names, institution |
| 0:15 | Problem | Forecasting horizon rigidity |
| 0:30 | D2Vformer | Date2Vec + cross-temporal attention |
| 1:00 | The bug | Linear(d_model, O) violates horizon-independence |
| 1:30 | Phase 1 | 44,021 params, 96/96 verification |
| 2:00 | Phase 6 | τ-MLP idea |
| 2:30 | Four modes | fixed, learned_global, temporal_context, query_conditioned |
| 3:00 | Headline result | ETTh2 O=720, +3.12%, all 3 seeds, Cohen's d = 1.67 |
| 3:30 | Statistical caveat | p ≈ 0.102, n = 3 underpowered |
| 4:00 | Negative result | ETTm1 O=720, -10.34% |
| 4:15 | Boundary | τ ≈ 0.18, over-sharpening, short lookback |
| 4:30 | Reproducibility | 48 checkpoints, SHA-256 verified |
| 4:45 | Conclusion | 9 contributions, no Phase 7 |
| 5:00 | Q&A | — |

### 29.2 The 5-minute pitch — common Q&A follow-ups

After a 5-minute pitch, the typical first question is "Why ETTh2 and not the others?" — see Q&A #5 below.

---

## 30. Fifty Viva Q&As

These are the 50 questions most likely to be asked in a BE Major Project viva, with concise answers.

### 30.1 Conceptual questions (Q1–Q10)

**Q1: What is D2Vformer?**

D2Vformer is a time-series forecasting model that uses a learnable harmonic embedding of timestamps (Date2Vec) to perform cross-temporal attention. It was proposed by Wang et al. in 2024 (arXiv:2409.11024).

**Q2: What is Date2Vec?**

Date2Vec is a learnable Fourier-like embedding of timestamps. It produces a vector of dimension `k_freq + 1`, where the linear component captures the raw time value and the harmonic components capture periodic structure at multiple frequencies.

**Q3: What is cross-temporal attention?**

Cross-temporal attention is a mechanism where each future query attends to each past key based on the similarity of their Date2Vec embeddings. The output is a weighted average of past temporal features.

**Q4: What is zero-shot horizon forecasting?**

Zero-shot horizon forecasting means training a model at one horizon and evaluating it at a different horizon without retraining. The model uses continuous time coordinates instead of positional indices.

**Q5: What is the bug you found in the original D2Vformer?**

The official repository uses `Linear(d_model, O)` as the output projection. This makes the parameter count scale with the forecast horizon O, violating the paper's claim of horizon-independent forecasting.

**Q6: What is softmax temperature?**

Softmax temperature is a scalar τ in the denominator of softmax: `softmax(S / τ)`. Smaller τ makes the distribution sharper (peaked); larger τ makes it softer (uniform). The original D2Vformer uses τ = 1.0 implicitly.

**Q7: Why does the temperature matter?**

The optimal temperature depends on the prediction task. For some queries, a sharp distribution over a few relevant past steps is best; for others, a soft distribution over many past steps is better. A fixed τ cannot handle both.

**Q8: What is attention entropy?**

Attention entropy is `H(A) = -Σ_l A[l] log A[l]`, the Shannon entropy of the attention distribution. Lower H means more peaked (focused); higher H means more diffuse (spread out).

**Q9: What is N_eff?**

The effective number of attended steps is `N_eff = exp(H)`. For a uniform distribution over L steps, N_eff = L. For a one-hot distribution, N_eff = 1.

**Q10: What is RevIN?**

RevIN (Reversible Instance Normalisation) is a normalisation technique (Kim et al., ICLR 2022) that subtracts the per-channel mean and divides by the per-channel std over the lookback window, then inverts this at the output. It helps with non-stationary time series.

### 30.2 Implementation questions (Q11–Q25)

**Q11: How many parameters does your model have?**

44,021 for `c_in = 7` (ETT datasets), 44,408 for `c_in = 8` (Exchange). This is constant across horizons.

**Q12: What is d_model?**

d_model = 128. The dimension of the temporal feature representation.

**Q13: What is k_freq?**

k_freq = 16. The number of learnable frequencies in Date2Vec. The output dimension of Date2Vec is `k_freq + 1 = 17`.

**Q14: What is d_ff?**

d_ff = 256. The hidden dimension of the feed-forward network.

**Q15: What is the lookback L?**

L = 96. All four datasets use the same lookback.

**Q16: What is the training horizon?**

O_train = 48. All models are trained to forecast 48 steps ahead.

**Q17: What evaluation horizons do you use?**

O ∈ {24, 48, 96, 192, 336, 720}. Six horizons covering short to long range.

**Q18: What seeds do you use?**

42, 43, 44. Three random seeds for all experiments.

**Q19: What optimiser do you use?**

AdamW with learning rate 1e-4, weight decay 1e-4, gradient clipping max norm 1.0.

**Q20: What is early stopping?**

Training stops if validation MSE does not improve for 3 consecutive epochs. The best checkpoint (lowest validation MSE) is saved.

**Q21: How is the validation MSE computed?**

On the validation split, with O = 48 (the training horizon), on the normalised series, using MSE loss.

**Q22: What loss function do you use?**

MSE on the normalised series. The output is denormalised before metric computation.

**Q23: What is `softplus`?**

`softplus(x) = log(1 + exp(x))`. A differentiable, smooth approximation to `max(0, x)`. We use it to ensure τ > 0.

**Q24: What is the τ-MLP architecture?**

For `query_conditioned`: `Linear((k+1)*d_model, 64) → GELU → Linear(64, 1) → softplus`. The output is reshaped to [B, O].

**Q25: How many parameters does the τ-MLP add?**

Approximately 9,000. The exact number is in `parameter_invariance_audit.csv`.

### 30.3 Results questions (Q26–Q40)

**Q26: What is your headline result?**

On ETTh2 at O=720, query_conditioned improves MSE by 3.12% over the fixed-temperature baseline. All 3 random seeds show > 2% improvement.

**Q27: What is Cohen's d for this result?**

Cohen's d = 1.67 (large effect size by Cohen's convention).

**Q28: What is the p-value?**

p ≈ 0.102. With n = 3 seeds, the classical significance threshold of p < 0.05 is not met.

**Q29: Why is the p-value not significant?**

Because n = 3 is too small. With df = 2, even a large effect size does not produce a significant t-statistic.

**Q30: How do you handle the p-value in your report?**

We report the effect size (Cohen's d), the per-seed breakdown, and the p-value, and we explicitly state that we do not claim statistical significance at p < 0.05.

**Q31: What about ETTh1?**

ETTh1 shows small consistent gains of ~1–1.7% across all horizons. Directionally positive but not as large as ETTh2.

**Q32: What about Exchange?**

Exchange shows small consistent gains of ~0.1–0.2% across all horizons. The data is small and noisy, so absolute improvements are small.

**Q33: What about ETTm1?**

ETTm1 shows -10.34% at O=720. This is a negative result, explained in §18 as τ over-sharpening due to short effective lookback (24 hours).

**Q34: Why does ETTm1 fail?**

ETTm1 has 15-minute sampling, giving a 24-hour effective lookback span (96 × 15 min). For a 7.5-day forecast, the model has too few past analogues. The learned τ collapses to ~0.18, over-sharpening attention.

**Q35: Is the ETTm1 result a bug?**

No, it's an empirical boundary condition. The model is correctly learning that τ should be small, but small τ over-sharpens the noisy short-lookback data. The mechanism works in principle; the data structure doesn't support it.

**Q36: Do you beat SOTA?**

No. PatchTST and iTransformer are still better than our model on most metrics. Our contribution is a specific mechanism improvement, not SOTA.

**Q37: Does your model beat the official D2Vformer?**

Yes, on ETTh2 at long horizons. The official D2Vformer has the parameter-scaling bug and uses τ = 1.0; we fix the bug and add τ-conditioning.

**Q38: What if you had more seeds?**

With more seeds, the p-value would likely drop. With n = 10, even a moderate effect size would reach p < 0.05. But the project freezes at n = 3.

**Q39: What if you trained longer?**

We use 10 epochs with early stopping. Longer training might give different results, but we follow the protocol freeze.

**Q40: What if you used different hyperparameters?**

Hyperparameter selection is on validation, not test. Different hyperparameters might give different numbers, but our protocol uses a single fixed configuration.

### 30.4 Methodological questions (Q41–Q50)

**Q41: How do you avoid test-set leakage?**

All model selection uses validation MSE at O=48. Test data is only used for final evaluation after selection is locked.

**Q42: What is selection audit?**

The selection audit reconstructs the pre-test selection decisions, verifying they were made on validation data only. See `selection_audit.md`.

**Q43: What is parameter invariance audit?**

The audit verifies that the parameter count is constant across horizons. See `parameter_invariance_audit.csv` and `parameter_invariance_report.md`.

**Q44: What is the difference between validation and test loss?**

Validation loss is computed on the validation split and used for early stopping. Test loss is computed on the test split and reported in results. They are different splits of the data.

**Q45: How do you ensure reproducibility?**

`torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `torch.backends.cudnn.deterministic = True`, and SHA-256 checksums of all checkpoints.

**Q46: What is the most important file in your repository?**

`models/tcd2vformer.py` — the headline contribution.

**Q47: How long does training take?**

~10 minutes per (dataset, mode, seed) on Colab T4. Total: ~8 hours for 48 runs.

**Q48: Can you reproduce the headline number?**

Yes. Open `Phase6_Colab_TCD2Vformer.ipynb`, run all cells, and check the printed output.

**Q49: What is the future work?**

Per the freeze, none within this project. Future work could include more seeds, more datasets, longer training, or different temperature mechanisms — but these are explicitly out of scope.

**Q50: What is the one-sentence summary of your project?**

> We reconstructed D2Vformer to fix a horizon-scaling bug, then added a learned softmax temperature conditioned on the future calendar, achieving +3.12% MSE improvement on ETTh2 at O=720 across all 3 seeds — with a clearly explained failure mode on ETTm1.

### 30.5 Q&A strategy

If asked a question you don't know:

1. **Don't bluff.** Say "I don't know the exact value, but the audit file `results/final_audit/X.md` would have it."
2. **Reference the file.** Every answer should ideally point to a specific document.
3. **Be honest about limits.** "n = 3 is too small for significance" is a stronger answer than "we achieved significance."

### 30.6 The most likely first question

The most likely first question is **Q26: "What is your headline result?"** Be ready with the ETTh2 +3.12% answer, the Cohen's d, and the n = 3 caveat.

### 30.7 The most likely trap question

The most likely trap is **"Did you achieve statistical significance?"** The wrong answer is "Yes." The right answer is "Cohen's d = 1.67 indicates a large effect size, but with n = 3 seeds, the p-value is approximately 0.102, which does not reach the classical significance threshold of p < 0.05. We report the effect size and the per-seed consistency, not classical significance."

---

**End of Part X.** Continue with [TEXTBOOK_PART_XI.md](TEXTBOOK_PART_XI.md).