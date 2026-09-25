# D2Vformer to TCD2Vformer — Part XI: Limitations, Claims, Conclusion

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §31–§33, `docs/limitations.md`, `FINAL_CONTRIBUTION.md`.

---

## 31. Limitations

This section makes the project's limits explicit. Knowing these limits is essential for an honest defense.

### 31.1 Sample size

**Limitation:** only 3 random seeds per (dataset, mode, horizon) combination.

**Why it matters:** with n = 3, classical statistical tests have low power. Our headline result has Cohen's d = 1.67 (large) but p ≈ 0.102 (not significant).

**What we do about it:** we report the per-seed breakdown, the effect size, and the p-value. We do not claim p < 0.05 significance.

### 31.2 Dataset scope

**Limitation:** only four datasets (ETTh1, ETTh2, ETTm1, Exchange). No Weather, Electricity, Traffic, ILI, or other common benchmarks.

**Why it matters:** generalisation beyond these four datasets is not tested. The mechanism might fail on datasets with very different characteristics.

**What we do about it:** we do not claim universal applicability. We identify ETTh2 as the headline and explain ETTm1 as a boundary.

### 31.3 Horizon scope

**Limitation:** only six horizons (24, 48, 96, 192, 336, 720).

**Why it matters:** behaviour at horizons outside this range (e.g., O=1440, O=12) is unknown.

**What we do about it:** we report results only for tested horizons and do not extrapolate.

### 31.4 Training duration

**Limitation:** only 10 epochs with early stopping.

**Why it matters:** longer training might give different results, especially for the larger τ-MLP variants.

**What we do about it:** we use a fixed protocol. The freeze prevents re-running with longer schedules.

### 31.5 Hyperparameter scope

**Limitation:** we use a single learning rate (1e-4), batch size (32), weight decay (1e-4). No hyperparameter search.

**Why it matters:** better hyperparameters might improve all numbers, including the baselines.

**What we do about it:** we use standard choices from the literature. No dataset-specific tuning.

### 31.6 Channel independence

**Limitation:** the model treats channels mostly independently (the FFN projects across channels at the end, but cross-channel attention is not used).

**Why it matters:** some datasets (e.g., Exchange) have strong cross-channel correlations. iTransformer-style cross-channel attention might help.

**What we do about it:** we follow the original D2Vformer design. Adding cross-channel attention would be Phase 7 (forbidden).

### 31.7 The negative result on ETTm1

**Limitation:** on ETTm1 at long horizons, our method degrades performance by 10.34%.

**Why it matters:** this is the most visible failure mode. It limits the claim of "universal improvement."

**What we do about it:** we explain the mechanism and frame it as an empirical boundary, not a bug. The explanation is in `ettm1_failure_analysis.md`.

### 31.8 Lookback window

**Limitation:** L = 96 for all datasets and horizons.

**Why it matters:** longer or shorter L might change the optimal τ dynamics. The relationship between L and τ-conditioning is unexplored.

**What we do about it:** we use the standard L = 96. We do not ablate over L.

### 31.9 No theoretical analysis

**Limitation:** we have empirical results but no theoretical proof of why τ-conditioning should help (or hurt).

**Why it matters:** a theoretical foundation would strengthen the contribution. We do not provide one.

**What we do about it:** we provide mechanistic intuition (entropy, over-sharpening, boundary conditions). We do not provide proofs.

### 31.10 No comparison to recent SOTA

**Limitation:** we do not run PatchTST, iTransformer, FEDformer, or other baselines ourselves. We use their published numbers.

**Why it matters:** our model is below SOTA on most metrics. The contribution is not "beating SOTA" but "improving the specific mechanism."

**What we do about it:** we state this explicitly. The thesis defense is not "we beat SOTA."

### 31.11 Summary of limitations

| Limitation | Severity | Mitigation |
|---|---|---|
| n = 3 seeds | Medium | Report effect size + per-seed breakdown |
| 4 datasets only | Medium | Explicit dataset list |
| 6 horizons only | Low | Explicit horizon list |
| 10-epoch training | Low | Frozen protocol |
| Single hyperparameter set | Medium | Standard choices |
| No cross-channel attention | Medium | Out of scope |
| ETTm1 negative | High | Mechanistic explanation |
| L = 96 only | Low | Standard convention |
| No theoretical analysis | Medium | Mechanistic intuition |
| No SOTA comparison | Low | Published baseline numbers |

---

## 32. What We Can and Cannot Claim

### 32.1 What we CAN claim

The following are claims directly supported by the experiments and frozen artifacts.

#### 32.1.1 About the reconstruction

✅ **"We reconstructed D2Vformer from scratch, removing the parameter-scaling projection from the official repository."**
- Evidence: `models/pure_d2vformer.py` is from scratch.
- Evidence: 96/96 horizon-independence audits pass.

✅ **"Our reconstructed model has 44,021 parameters for c_in = 7 and 44,408 parameters for c_in = 8."**
- Evidence: `parameter_invariance_audit.csv`.

✅ **"The reconstruction is horizon-independent: ∂N_params / ∂O ≡ 0."**
- Evidence: 96/96 audits.

#### 32.1.2 About the τ-mechanism

✅ **"We introduced a learnable softmax temperature mechanism with four modes: fixed, learned_global, temporal_context, and query_conditioned."**
- Evidence: `models/temperature_d2vformer.py`.

✅ **"On ETTh2 at O=720, query_conditioned improves MSE by 3.12% over fixed."**
- Evidence: `locked_test_results.csv`, `ablation_results.csv`.

✅ **"All 3 / 3 random seeds improve by more than 2% on ETTh2 O=720."**
- Evidence: per-seed breakdown in `etth2_long_horizon_analysis.md`.

✅ **"Cohen's d = 1.67 for the ETTh2 O=720 comparison (large effect size)."**
- Evidence: `statistical_robustness.csv`.

✅ **"On ETTh2, the improvement grows monotonically with horizon: +0.58% at O=24 to +3.12% at O=720."**
- Evidence: `etth2_long_horizon_analysis.md`.

#### 32.1.3 About the negative result

✅ **"On ETTm1 at O=720, query_conditioned degrades MSE by 10.34%."**
- Evidence: `locked_test_results.csv`, `ablation_results.csv`.

✅ **"The learned τ on ETTm1 at O=720 averages ~0.18, indicating severe over-sharpening."**
- Evidence: `ettm1_failure_analysis.md`.

✅ **"ETTm1 has a 24-hour effective lookback due to 15-minute sampling; this is the empirical boundary for the mechanism."**
- Evidence: `ettm1_failure_analysis.md`.

#### 32.1.4 About reproducibility

✅ **"All 48 checkpoints are SHA-256 verified and match the locked hashes."**
- Evidence: `audit_task1_reproducibility.py`.

✅ **"All 288 test evaluations are reproducible from the locked CSVs."**
- Evidence: `locked_test_results.csv`.

✅ **"The full pipeline can be re-run on Colab in ~8 hours."**
- Evidence: `D2Vformer_colab_ready.zip`, `Phase6_Colab_TCD2Vformer.ipynb`.

### 32.2 What we CANNOT claim

These claims are not supported by the experiments and must NOT appear in the thesis or defense.

#### 32.2.1 Statistical overreach

❌ **"The result is statistically significant at p < 0.05."**
- Truth: p ≈ 0.102 with n = 3.

❌ **"We have proven that τ-conditioning works."**
- Truth: empirical evidence on 4 datasets, no proof.

#### 32.2.2 Universal applicability

❌ **"Our method improves all time-series forecasting."**
- Truth: improves on ETTh2, neutral on ETTh1, marginal on Exchange, hurts on ETTm1.

❌ **"Our method improves all horizons."**
- Truth: monotonic gain on ETTh2, neutral on others.

❌ **"Our method beats SOTA."**
- Truth: PatchTST and iTransformer are still better on most metrics.

#### 32.2.3 Mechanism overreach

❌ **"The mechanism works for any τ > 0."**
- Truth: only tested with τ = softplus(MLP output), not other forms.

❌ **"The mechanism works for any embedding."**
- Truth: only tested with Date2Vec, not positional or sinusoidal embeddings.

#### 32.2.4 Implementation overreach

❌ **"Our implementation is identical to the original paper."**
- Truth: our implementation fixes a bug in the official code. The paper's intent matches; the implementation differs.

❌ **"We added new transformer layers."**
- Truth: we added a small MLP (~9K params). No new attention layers.

#### 32.2.5 Phase 7 claims

❌ **"We could improve further with Phase 7."**
- Truth: Phase 7 is explicitly forbidden.

❌ **"With more compute, we'd get better results."**
- Truth: maybe, but the project freezes here.

### 32.3 Summary of claim boundaries

| Claim type | Status |
|---|---|
| Reconstruction is horizon-independent | ✅ |
| τ-MLP exists and works in 4 modes | ✅ |
| ETTh2 O=720 +3.12% with all 3 seeds | ✅ |
| Cohen's d = 1.67 (large) | ✅ |
| Monotonic gain with horizon on ETTh2 | ✅ |
| ETTm1 O=720 -10.34% | ✅ |
| ETTm1 τ ≈ 0.18 over-sharpening | ✅ |
| 48 checkpoints SHA-256 verified | ✅ |
| p < 0.05 significance | ❌ |
| Universal improvement | ❌ |
| SOTA results | ❌ |
| Phase 7 exists | ❌ |

### 32.4 How to handle "Can you claim X?" in the viva

If a panelist asks "Can you claim X?" and X is in the ❌ list:

- Acknowledge directly: "No, we cannot claim that."
- Explain why: "n = 3 is too small for significance" or "ETTm1 is a clear counter-example."
- Pivot: "What we can claim is Y."

If X is in the ✅ list:

- State it directly.
- Cite the file.

---

## 33. Final Conclusion

### 33.1 What we built

A clean, reproducible, horizon-independent extension of D2Vformer with a learnable softmax temperature mechanism conditioned on the future calendar. The extension is validated on four datasets, six horizons, and three seeds, with a clearly-explained failure mode.

### 33.2 What we learned

1. **The original D2Vformer repository had a hidden horizon-scaling bug.** Removing it restores the paper's claim.
2. **The optimal softmax temperature depends on the prediction task.** A fixed τ cannot adapt; a learned τ can.
3. **Conditioning on the future calendar is more useful than learning a single scalar.** Per-query τ (query_conditioned) is the strongest mode.
4. **Mechanisms have boundaries.** On ETTm1, the mechanism over-sharpens and hurts. The boundary is real, not a bug.
5. **Reproducibility requires discipline.** Seeds, hashes, audit trails, and protocol freezes are necessary, not optional.

### 33.3 The scientific story in one paragraph

Time-series forecasting models benefit from explicit, calendar-aware conditioning of their attention mechanisms. The original D2Vformer uses Date2Vec to embed timestamps, but the cross-temporal attention uses a fixed softmax temperature. We replaced this with a small MLP that produces a per-query temperature from the future Date2Vec embeddings. On ETTh2 — a dataset with strong calendar structure — this conditioning improves long-horizon (O=720) MSE by 3.12% across all 3 random seeds (Cohen's d = 1.67, large effect). On ETTm1 — a dataset with short effective lookback span due to fine-grained sampling — the mechanism over-sharpens attention (τ ≈ 0.18) and degrades performance by 10%. The empirical boundary condition is that τ-conditioning helps when there are enough past analogues for the prediction task; it hurts when there aren't.

### 33.4 What this thesis is

This is a BE Major Project that:

- Identifies and fixes a real bug in published code.
- Proposes a specific, mechanistic improvement.
- Validates the improvement empirically with rigor.
- Documents a failure mode honestly.
- Is fully reproducible from public artifacts.

### 33.5 What this thesis is not

This is not:

- A state-of-the-art forecasting model.
- A theoretical contribution to deep learning.
- A claim of universal improvement.
- A paper with n = 30 seeds and statistical significance.

### 33.6 The classification

`FINAL_AUDIT_SUMMARY.md` concludes with **Classification A**: contribution is sufficiently validated; freeze technical work.

The freeze holds. Phase 7 does not exist. The work is done.

### 33.7 What to do next

For the BE Major Project:

1. **Finalise the thesis report** using this textbook as the structure.
2. **Prepare the viva defense** using Part X (30-second, 2-minute, 5-minute pitches, 50 Q&As).
3. **Archive the repository** with the locked artifacts.
4. **Submit**.

For future research (not this project):

- Larger seed counts for proper significance tests.
- More datasets (Weather, Electricity, Traffic, ILI).
- Longer horizons (O > 1000).
- Different temperature mechanisms (e.g., attention pooling).
- Theoretical analysis of when τ-conditioning should help.

These are open questions. This thesis does not answer them, and does not need to.

### 33.8 The closing sentence

> We contributed a horizon-independent reconstruction of D2Vformer, a learned temperature mechanism conditioned on Date2Vec, a four-mode ablation, an empirically-identified boundary condition, and a fully reproducible pipeline — and we did so without overstating the evidence.

---

## Final Textbook Metadata

### Total sections: 33

### Approximate word count: ~75,000

### Major topics covered:

1. Project foundations and prerequisites (Part I)
2. Original D2Vformer and Date2Vec (Part II)
3. Phases 1–5 reconstruction (Part III)
4. Phase 6 TCD2Vformer architecture and math (Part IV)
5. Protocol, ablation, headline results (Part V)
6. Dataset deep dives — ETTh2 success, ETTm1 failure (Part VI)
7. Inherited vs contributed work (Part VII)
8. Reproducibility and final audit (Part VIII)
9. Repository structure and end-to-end pipeline (Part IX)
10. Defense pitches at three lengths and 50 viva Q&As (Part X)
11. Limitations, claim boundaries, conclusion (Part XI)

### Gaps in `PROJECT_KNOWLEDGE_MAP.md`:

The knowledge map was sufficient to write this textbook. Two minor items were inferred rather than directly stated:

- The exact τ values at training end for each (dataset, mode, horizon) were not in the knowledge map but were in `ettm1_failure_analysis.md` and `etth2_long_horizon_analysis.md`.
- The selection_audit pre-test decisions are in `selection_audit.csv`, not summarised in the knowledge map.

### Ambiguities resolved during writing:

1. **TC-D2Vformer vs TCD2Vformer** — resolved to TCD2Vformer for the final name (Phase 6); TC-D2Vformer (Phases 1–5) refers to the same family.
2. **Validation vs test numbers** — locked values are in `locked_test_results.csv`; the markdown typo (0.8037/0.23071/0.67232/0.11978) is documentation-only and not propagated.
3. **README ETTh2 numbers** — those are Phase 4 scalar-τ baseline numbers; Phase 6 dynamic-τ +3.12% is the headline.

### The textbook is complete.

---

**End of Part XI.** This is the final part. The textbook is now ready to be combined into `PROJECT_TEXTBOOK.md`.