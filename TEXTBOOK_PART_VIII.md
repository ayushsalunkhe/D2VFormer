# D2Vformer to TCD2Vformer — Part VIII: Reproducibility & Final Audit

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §23–§24, `docs/reproducibility.md`, all files in `results/final_audit/`.

---

## 23. Reproducibility — How to Verify Every Claim

### 23.1 What "reproducible" means here

Every numerical claim in this textbook can be reproduced from the public artifacts. Specifically:

- The 48 trained checkpoints can be reloaded and produce the same test-set MSE.
- The CSV logs match the checkpoint outputs exactly.
- The SHA-256 checksums verify no checkpoint was modified post-training.

### 23.2 The artifacts

#### 23.2.1 Checkpoints

48 `.pt` files in `results/phase{1,2,6}/checkpoints/`:

- Phase 1: 4 datasets × 1 mode × 3 seeds = 12.
- Phase 2: 4 datasets × 1 mode × 3 seeds = 12 (same as Phase 1, re-trained for Phase 2's protocol).
- Phase 6: 4 datasets × 3 modes × 3 seeds = 36.

Total: 48.

Each `.pt` file contains:

```python
{
    'state_dict': model.state_dict(),
    'optim_state': optimizer.state_dict(),
    'epoch': int,
    'val_loss': float,
    'config': dict
}
```

#### 23.2.2 CSVs

The locked CSVs:

- `validation_results.csv` — per-checkpoint validation MSE at O_train = 48.
- `locked_test_results.csv` — per-checkpoint test MSE at all O ∈ {24, 48, 96, 192, 336, 720}.
- `ablation_results.csv` — full ablation table (4 datasets × 4 modes × 6 horizons × 3 seeds).
- `parameter_invariance_audit.csv` — 96 rows of (dataset, mode, horizon, param_count).
- `selection_audit.csv` — 12 rows of pre-test selection decisions.
- `statistical_robustness.csv` — per-(dataset, horizon) paired differences, Cohen's d, p-values.

#### 23.2.3 Reports

Markdown reports in `results/final_audit/`:

- `FINAL_AUDIT_SUMMARY.md` — the umbrella document.
- `FINAL_CONTRIBUTION.md` — the 13-point defense statement.
- `parameter_invariance_report.md` — horizon-independence analysis.
- `selection_audit.md` — protocol verification.
- `statistical_robustness.md` — inference analysis.
- `etth2_long_horizon_analysis.md` — ETTh2 deep dive.
- `ettm1_failure_analysis.md` — ETTm1 explanation.
- `contribution_boundary.md` — novelty boundary.
- `numerical_consistency_report.md` — cross-csv consistency check.

### 23.3 How to reproduce a number

Suppose you want to verify "ETTh2 O=720 query_conditioned seed 42 gives MSE = 0.490".

1. **Load the checkpoint:**
   ```python
   ckpt = torch.load('results/phase6/checkpoints/etth2_query_conditioned_seed42.pt')
   model.load_state_dict(ckpt['state_dict'])
   ```

2. **Run on the ETTh2 test set with O=720:**
   ```python
   test_mse = evaluate(model, dataset='etth2', split='test', O=720, seed=42)
   ```

3. **Compare to the CSV:**
   ```python
   locked = pd.read_csv('locked_test_results.csv')
   row = locked[(locked.dataset=='etth2') &
                (locked.mode=='query_conditioned') &
                (locked.seed==42) &
                (locked.O==720)]
   assert abs(row.mse.values[0] - test_mse) < 1e-4
   ```

If this passes, the number is reproduced.

### 23.4 The protocol verification (`selection_audit.md`)

The selection audit reconstructs the **pre-test** decisions:

- Which checkpoint was selected as "best" for each (dataset, seed) pair.
- What its validation MSE was.
- That the selection happened **before** test-set evaluation.

This is critical because post-hoc selection on test data would be a protocol violation. The audit shows the selection decisions were made on validation data only.

### 23.5 The 12 (dataset, seed) pre-test selections

| Dataset | Seed | Selected epoch | Val MSE |
|---|---|---|---|
| ETTh1 | 42 | 7 | 0.456 |
| ETTh1 | 43 | 6 | 0.461 |
| ETTh1 | 44 | 8 | 0.453 |
| ETTh2 | 42 | 5 | 0.348 |
| ETTh2 | 43 | 6 | 0.351 |
| ETTh2 | 44 | 7 | 0.346 |
| ETTm1 | 42 | 4 | 0.420 |
| ETTm1 | 43 | 5 | 0.418 |
| ETTm1 | 44 | 4 | 0.422 |
| Exchange | 42 | 8 | 0.115 |
| Exchange | 43 | 9 | 0.118 |
| Exchange | 44 | 7 | 0.116 |

*Reference values from `selection_audit.csv`. These are the "best" checkpoints.*

For Phase 6's three additional modes, the same protocol was applied — 36 additional (dataset, mode, seed) selections.

### 23.6 The SHA-256 verification

Every checkpoint was hashed at training time and re-hashed at audit time:

```
audit_task1_reproducibility.py:
  for ckpt in checkpoints:
      sha = sha256(ckpt)
      assert sha == locked_sha[ckpt]
```

48/48 hashes matched. **No checkpoint was modified after training.**

### 23.7 The numerical consistency check (`numerical_consistency_report.md`)

The consistency check verified:

- All values in markdown tables match values in CSVs.
- All values in CSVs match values from re-evaluation.
- The only discrepancy found was a documentation typo (validation values were listed instead of test values in some markdown tables).

This typo:

- Does NOT affect any CSV value.
- Does NOT affect any checkpoint.
- Affects only one markdown table in `RESEARCH_FINDINGS.md` (specifically: 0.8037 / 0.23071 / 0.67232 / 0.11978 listed where the true test values are 0.56326 / 0.27554 / 0.45555 / 0.31498).

The lock values are in `locked_test_results.csv`. Any claim about specific MSE values should reference this CSV, not the markdown.

### 23.8 The "from scratch" reproducibility test

A from-scratch reproduction was also attempted:

1. Wipe all checkpoints.
2. Re-run all 48 training runs from scratch.
3. Compare new checkpoints' MSE to the locked ones.

The new checkpoints agree with the locked ones to within seed-noise (±2%). The slight discrepancy is expected because CUDA non-determinism can produce small differences in some operations. The protocol uses `torch.backends.cudnn.deterministic = True` to minimise this.

### 23.9 What reproducibility does NOT guarantee

Reproducibility does NOT guarantee:

- **Correctness of the contribution.** A reproducible wrong result is still wrong.
- **Statistical significance.** n = 3 is too small, regardless of how reproducible it is.
- **Generalisation beyond the four datasets.**

Reproducibility guarantees only:

- The numbers reported are the numbers produced.
- Anyone with the code can verify the numbers.

---

## 24. Final Audit

### 24.1 The audit's scope

The final audit has 10 tasks (Task 1–10). All are complete. The umbrella document is `FINAL_AUDIT_SUMMARY.md`.

### 24.2 Task-by-task summary

#### Task 1 — Data Integrity

- **What:** verify all 48 checkpoints, 288 evaluations, and CSV logs.
- **Result:** 48/48 SHA-256 checksums match. 0 omissions.

#### Task 2 — Horizon Independence

- **What:** verify `∂N_params/∂O ≡ 0` for all (dataset, mode, horizon) combinations.
- **Result:** 96/96 pass. Parameter count is 44,021 (c_in = 7) or 44,408 (c_in = 8) at every O.

#### Task 3 — Selection Protocol

- **What:** verify pre-test selection was on validation data only.
- **Result:** 12 (dataset, seed) selection decisions reconstructed; all from validation MSE at O=48.

#### Task 4 — Statistical Robustness

- **What:** compute standard errors, Cohen's d, paired differences, p-values.
- **Result:** ETTh2 O=720 Cohen's d = 1.67 (large); p = 0.102 (underpowered).

#### Task 5 — Central Claim

- **What:** evaluate the "temporal conditioning improves long-horizon forecasting" claim per dataset.
- **Result:** confirmed on ETTh2 (large effect), partial on ETTh1/Exchange (small effect), rejected on ETTm1 (negative).

#### Task 6 — ETTh2 Long Horizon

- **What:** verify monotonicity of gains with horizon on ETTh2.
- **Result:** gains grow from +0.58% (O=24) to +3.12% (O=720) — strictly monotonic.

#### Task 7 — ETTm1 Failure Analysis

- **What:** explain why τ-conditioning hurts ETTm1.
- **Result:** τ ≈ 0.18 over-sharpening due to 24-hour effective lookback vs 7.5-day horizon.

#### Task 8 — Novelty Boundary

- **What:** separate inherited from contributed work.
- **Result:** 9 contributions identified; novelty boundary explicit.

#### Task 9 — Numerical Consistency

- **What:** cross-check CSVs and reports.
- **Result:** 1 minor documentation typo logged. CSV values are uncorrupted.

#### Task 10 — Final Statement

- **What:** write the umbrella defense statement.
- **Result:** `FINAL_CONTRIBUTION.md` — 13 conservative claims.

### 24.3 The 13-point defense statement (preview)

From `FINAL_CONTRIBUTION.md`:

1. We provide a horizon-independent reconstruction of D2Vformer.
2. We add a learned temperature mechanism conditioned on Date2Vec.
3. We systematically ablate four temperature modes.
4. ETTh2 O=720 shows +3.12% MSE reduction.
5. All 3/3 seeds improve by > 2% on ETTh2 O=720.
6. Cohen's d = 1.67 (large effect).
7. n = 3 is underpowered for p < 0.05 (p ≈ 0.102).
8. ETTh1 and Exchange show directionally consistent small gains.
9. ETTm1 O=720 shows -10.34% (negative).
10. The negative result is mechanistically explained as τ over-sharpening.
11. The empirical boundary is identified: short effective lookback span.
12. All results are reproducible from the public artifacts.
13. We do not claim universal improvement across datasets.

### 24.4 The classification

`FINAL_AUDIT_SUMMARY.md` concludes with **Classification A**:

> Contribution is sufficiently validated; freeze technical work.

This means:
- No more experiments.
- No more phases.
- No "let me try one more thing."

The work is done.

### 24.5 What "Classification A" enables

With Classification A:

- The thesis can be written with confidence.
- The defense can be prepared against specific questions.
- The reproducibility package can be packaged for archival.

### 24.6 What "Classification A" forbids

- Phase 7 (any kind).
- Modifying checkpoints.
- Re-running experiments to "improve" numbers.
- Cherry-picking results post-hoc.

The freeze is permanent.

### 24.7 Audit deliverables summary

| Task | File | Status |
|---|---|---|
| 1 | scratch/audit_task1_reproducibility.py | ✓ |
| 2 | parameter_invariance_audit.csv, parameter_invariance_report.md | ✓ |
| 3 | selection_audit.csv, selection_audit.md | ✓ |
| 4 | statistical_robustness.csv, statistical_robustness.md | ✓ |
| 5 | FINAL_CONTRIBUTION.md (Task 5 section) | ✓ |
| 6 | etth2_long_horizon_analysis.md | ✓ |
| 7 | ettm1_failure_analysis.md | ✓ |
| 8 | contribution_boundary.md | ✓ |
| 9 | numerical_consistency_report.md | ✓ |
| 10 | FINAL_CONTRIBUTION.md | ✓ |

### 24.8 What the final audit does NOT cover

- Theoretical analysis (no proofs).
- Comparison to non-ETT/Exchange datasets (e.g., Weather, Electricity, Traffic).
- Long training (we train 10 epochs; longer schedules might give different results).
- Per-channel analysis.

These are out of scope. They could be future work but are not part of this thesis.

---

**End of Part VIII.** Continue with [TEXTBOOK_PART_IX.md](TEXTBOOK_PART_IX.md).