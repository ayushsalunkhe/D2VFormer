# D2Vformer to TCD2Vformer: A Complete Technical and Research Guide

> **Status:** Internal team textbook — BE Major Project, AY 2025–26.
> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md`. Every numerical value and every claim is cross-checked against that document.
> **Research status:** Phases 1–6 + Final Audit **FROZEN**. No Phase 7.
> **Audience:** A student who knows Python, ML, and DL — but has never seen D2Vformer.

---

# Table of Contents

- **PART I — Project Foundations** (this file)
  - §1 Project Overview
  - §2 Prerequisite Concepts
- **PART II — Original D2Vformer** (separate file: `TEXTBOOK_PART_II.md`)
- **PART III — Our Research Journey** (separate file: `TEXTBOOK_PART_III.md`)
- **PART IV — Our Main Contribution** (separate file: `TEXTBOOK_PART_IV.md`)
- **PART V — Experimental Methodology** (separate file: `TEXTBOOK_PART_V.md`)
- **PART VI — Important Findings** (separate file: `TEXTBOOK_PART_VI.md`)
- **PART VII — Contribution and Novelty** (separate file: `TEXTBOOK_PART_VII.md`)
- **PART VIII — Reproducibility and Audit** (separate file: `TEXTBOOK_PART_VIII.md`)
- **PART IX — Understanding the Code** (separate file: `TEXTBOOK_PART_IX.md`)
- **PART X — Defense Preparation** (separate file: `TEXTBOOK_PART_X.md`)
- **PART XI — Limitations and Honest Conclusion** (separate file: `TEXTBOOK_PART_XI.md`)

The combined final document is `PROJECT_TEXTBOOK.md`, assembled from these parts at the end.

---

# PART I — PROJECT FOUNDATIONS

## 1. Project Overview

### 1.1 What this project is, in one paragraph

> This BE project audits the base paper *D2Vformer* (Wang et al., *IEEE TNNLS*, arXiv:2409.11024) — a transformer-style model that uses *Date2Vec* time embeddings to forecast an arbitrary number of future steps from a single training run. We found that the official code was not truly horizon-independent (it had a hidden `Linear(d_model, O)` output head), reconstructed a parameter-free backbone (`PureD2Vformer`), and then discovered that cross-temporal Date2Vec attention is *diffuse* (near-uniform) across all four datasets we tested. We showed that a validation-selected scalar temperature τ regularises this attention and improves zero-shot multi-horizon forecasting. Finally, we replaced the discrete τ grid with an *input-conditioned temperature* `τ = f_θ(D_x, D_y)`, producing **TCD2Vformer** — Temporal-Conditioned Date2Vecformer — which adds only 305 trainable parameters regardless of forecast horizon, and yields a Cohen's *d* = 1.67 (+3.12% MSE) zero-shot improvement on ETTh2 at horizon O = 720, while staying exactly horizon-independent. We also documented, honestly, where this fails (ETTm1 long horizons).

This is the **project in one paragraph**. Every claim in it is sourced.

### 1.2 What is the project?

It is a **Bachelor of Engineering (Computer Engineering) Major Project** completed during AY 2025–26. It consists of:

1. A faithful **reconstruction** of an existing research model (`D2Vformer`) with a documented fix.
2. A **diagnostic study** of a component of that model (cross-temporal attention) that the original paper did not study.
3. A **methodological contribution** (validation-only temperature selection without test-set leakage).
4. An **architectural contribution** (`TCD2Vformer`) — dynamic, horizon-independent attention temperature conditioned on the same Date2Vec embeddings the original paper already uses.
5. A **final contribution audit** with a 10-task checklist that classifies the project as **Class A** ("Contribution sufficiently validated; freeze technical work").

The technical work is **frozen**. There is no Phase 7.

### 1.3 What problem does it address?

Three nested problems:

1. **Base architecture problem.** *D2Vformer* claims arbitrary-horizon forecasting with a *fixed* parameter count. Our Phase 1 audit discovered that the official code's parameter count was *not* fixed — it scaled with the forecast horizon O via a hidden output projection. So the headline claim of the base paper needed verification and correction.
2. **Mechanism problem.** Phase 2 diagnostics showed that even in the cleaned-up backbone, the cross-temporal attention distribution was nearly uniform across all lookback steps (H_norm ≈ 0.92–0.98). This is suspicious: a transformer whose attention is essentially an unweighted average has no mechanism for selective temporal grounding.
3. **Architectural problem.** Phases 3–5 showed that a manually chosen scalar temperature τ, selected by validation loss, improves performance. But manually picking τ is a hyperparameter search, not a model design. Phase 6 makes τ a *learned function of temporal phase embeddings* — turning a grid search into an architectural component.

### 1.4 What is time-series forecasting?

A time series is a sequence of measurements taken at successive points in time.

```
Time index:    t=1   t=2   t=3   ...   t=L   |   t=L+1   t=L+2   ...   t=L+O
Observed:      y_1   y_2   y_3   ...   y_L   |   ?
```

Given the past `L` values, we want to predict the next `O` values.

- `L` is the **lookback window**.
- `O` is the **forecast horizon**.
- `y_i` is often a vector (e.g. 7 sensor channels, or 8 currency exchange rates), not a single number — that is **multivariate** forecasting. When `y_i` is a single number, it is **univariate**.

**Short vs long horizon.** Forecasting 24 hours ahead is qualitatively different from forecasting 30 days ahead. Long-horizon forecasting compounds small errors, requires reasoning about seasonal/calendar structure, and exposes any temporal bias the model carries.

### 1.5 Why is long-horizon forecasting hard?

- **Error compounding.** A small per-step error multiplies over O steps. A model that drifts by 1% per step is 10% off at O = 10, and 50% off at O = 50.
- **Distribution shift.** The further into the future, the more likely the test data deviates from the training distribution (weather, demand, regime changes).
- **Temporal pattern mismatch.** Long-horizon forecasting depends on calendar cycles (hour-of-day, day-of-week, week-of-year). If the model does not encode these cycles, it has no way to project them forward.
- **Parameter cost.** A naive fix is to train a separate model per horizon. This is expensive at training time and impossible at deployment time when the horizon is not known in advance.

### 1.6 What makes temporal information important?

Most real time series have *calendar* structure. Electricity demand has a daily cycle. Stock trading volume has an intraday pattern. Weather has seasonal cycles. If the model cannot represent *"this is a Wednesday at 8 PM"* as a vector that is close to *"next Wednesday at 8 PM"* and far from *"this Monday at 3 AM"*, it will not be able to project forward.

This is the entire motivation for *time2vec*-style embeddings in modern forecasting models.

### 1.7 Why conventional architectures may struggle with temporal structure

A standard transformer encoder treats the input as a sequence of feature vectors. If you simply feed it `(feature_value_at_t1, feature_value_at_t2, ...)`, the model has to *infer* the calendar from the values. It can do this indirectly — but it loses fidelity, especially for low-frequency patterns (weekly, seasonal) that span many input steps.

The fundamental idea of D2Vformer (and of all *time2vec*-style architectures) is: **give the model the calendar directly**, in the form of a learnable temporal embedding. Then the attention mechanism can compare past times to future times by their embeddings.

### 1.8 What is D2Vformer?

D2Vformer is a transformer-style forecasting model whose key innovation is **continuous-time queries**:

- It encodes every historical timestamp `t` into a vector `D_x(t)` using Date2Vec.
- It encodes every future timestamp `τ` into a vector `D_y(τ)` using the same Date2Vec.
- It computes cross-temporal attention between these embeddings: how much should `D_y(τ)` attend to `D_x(t)`?
- It uses that attention to aggregate historical feature values into a forecast at time `τ`.

Because the future timestamps are supplied at *inference* time, the same trained model can forecast any horizon. This is the **flexible/zero-shot forecasting** claim.

### 1.9 What ultimately became our research contribution?

After reconstructing PureD2Vformer, diagnosing diffuse attention, and validating scalar temperature selection, we asked:

> *"Can we make τ a learned function of the temporal embeddings themselves, instead of picking it from a grid?"*

That yielded `TCD2Vformer`: `τ = τ_min + softplus(MLP(mean(D)))`. The MLP is tiny (305 parameters), horizon-independent, and produces a context-aware or query-aware temperature.

The final contribution is therefore:

> **Temporal-Conditioned Date2Vecformer (TCD2Vformer):** a Date2Vec-conditioned, horizon-independent dynamic attention temperature mechanism, validated by a 10-task final audit, achieving a Cohen's *d* = 1.67 (+3.12% MSE at O = 720 on ETTh2) under a strict zero-lookahead validation protocol.

---

## 2. Prerequisite Concepts

This section teaches every concept required to understand the rest of the textbook. Each concept has three layers:

- **Intuition** — what is it, in plain words.
- **Mathematical Explanation** — the formal version.
- **Tensor Shapes** — the dimensions in code.
- **Example** — a concrete case.
- **Why we use it** — its role in our project.

### 2.1 Time-Series Data

#### Observations

An **observation** is a single measurement. In a multivariate time series, an observation is a vector.

```
t=1:  y_1 = (y_1_1, y_1_2, ..., y_1_D)         # D = number of variables / features
t=2:  y_2 = (y_2_1, y_2_2, ..., y_2_D)
...
```

#### Timestamps

A **timestamp** is the calendar coordinate `t`. It is what humans use to label an observation ("3 PM on October 15"). A timestamp can be raw (`2026-10-15 15:00:00`) or decomposed into calendar features like `(year, month, day, weekday, hour, minute)`.

In our project, timestamps are represented as vectors of `M = 4` normalised calendar features: `[year_norm, month_sin, day_sin, weekday_sin]` (or similar).

#### Variables / features

The **variables** (also called **channels** or **features**) are the separate measurements at each time. For ETTh1/ETTh2/ETTm1 there are 7 (high-voltage transformer load, oil temperature, etc.). For Exchange Rate there are 8 (8 country currency pairs).

#### Lookback window

The **lookback window** is the slice of past time the model sees. We use `L = 96` steps.

```
[ y_1 ] [ y_2 ] ... [ y_L ]   ->   model   ->   forecast for t = L+1, ..., L+O
```

#### Forecast horizon

The **forecast horizon** is how many future steps the model predicts. We denote it `O`. The model is trained at `O_train = 48` but evaluated at `O ∈ {24, 48, 96, 192, 336, 720}`.

#### Univariate vs multivariate forecasting

- **Univariate:** predict one variable at a time.
- **Multivariate:** predict multiple variables jointly. Our project is multivariate.

#### Short vs long horizon

There is no universal cutoff. In this project:

| Horizon | Real-world span (hourly data) | Real-world span (15-min data) |
|---|---|---|
| 24 | 1 day | 6 hours |
| 48 | 2 days | 12 hours |
| 96 | 4 days | 1 day |
| 192 | 8 days | 2 days |
| 336 | 14 days | 3.5 days |
| 720 | 30 days | 7.5 days |

This physical-time interpretation matters for interpreting results — it is what makes the ETTm1 O = 720 result physically meaningful (or not).

#### Intuition box

> **Lookback vs horizon, intuitively.** You are predicting tomorrow's temperature using the last 4 days of hourly weather data. The lookback is 96 hourly readings; the forecast horizon is 24 (one day ahead, hourly). The model gets 4 days of history and emits 1 day of predictions.

### 2.2 Deep Learning for Time Series

#### Input tensors

A batch of time-series inputs is a tensor with four axes:

```
x [B, L, c_in]    # B = batch size, L = lookback, c_in = number of channels
```

In our project, `c_in = 7` for ETT datasets, `c_in = 8` for Exchange Rate.

#### Batches

A **batch** is a collection of independent samples processed in parallel. The model never sees batch elements as a sequence — they are stacked for compute efficiency.

#### Sequence length

**Sequence length** is the temporal axis. We have two: `L` (input lookback) and `O` (output forecast). The model accepts both because `O` is variable at inference time.

#### Model parameters

**Model parameters** are the trainable numbers in the network (weights and biases). For our TCD2Vformer, total parameters:

| Mode | c_in = 7 | c_in = 8 |
|---|---|---|
| `fixed` (baseline) | 44,021 | 44,408 |
| `learned_global` | 44,022 | 44,409 |
| `temporal_context` | 44,326 | 44,713 |
| `query_conditioned` | 44,326 | 44,713 |

#### Training / validation / test splits

We split each dataset **chronologically** (no shuffling):

```
[========== TRAIN (60%) ==========][==== VAL (20%) ====][==== TEST (20%) ====]
```

- **Train**: fit the model.
- **Validation**: choose hyperparameters, early stopping, τ selection. **Test set is forbidden here.**
- **Test**: evaluate the locked model **once**.

This chronological split is what makes "zero-lookahead" possible. A random split would leak future information into training.

### 2.3 Attention

This is the core mechanism. Spend time here.

#### Query, Key, Value

In standard attention:

- A **query** `q` is a vector that asks: "what should I look at?"
- A **key** `k` is a vector that says: "this is what I am."
- A **value** `v` is the actual content associated with each key.

Attention computes similarity between `q` and `k`, uses softmax to turn similarities into weights, and uses those weights to take a weighted sum of `v`.

#### Dot-product attention

The standard formula:

```
score_i = q · k_i                 # raw similarity
weight_i = exp(score_i) / sum_j exp(score_j)   # softmax
output = sum_i weight_i * v_i     # weighted sum
```

In matrix form:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

where `d_k` is the key dimensionality, and the `1/sqrt(d_k)` factor prevents the softmax from saturating.

#### Softmax

**Softmax** turns a vector of real numbers into a probability distribution (positive entries summing to 1).

```
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

Properties:

- Output is always positive.
- Output sums to 1.
- Larger inputs get relatively larger outputs.
- If one input is much larger than the others, the output concentrates on it.

#### Attention weights

The output of `softmax(Q K^T / sqrt(d_k))` is the **attention weight matrix** `A`. Row `A[i, :]` is the attention distribution that query `i` puts over all keys.

In cross-temporal attention:

- Each row `A[o, :]` is the probability distribution over past time steps `l ∈ {1, ..., L}` that future query step `o` attends to.
- The sum across `l` is 1.

#### Why attention can become diffuse

If the keys are nearly the same, all `score_i = q · k_i` will be similar, and softmax will produce a **nearly uniform** distribution — every past step gets nearly the same weight. This is what we observed: the attention distribution was 90–97% uniform across all past time steps.

This happens when:

- The key vectors have small variance.
- The query vector is similar in shape to all keys (i.e. not selective).

For Date2Vec embeddings, the inner products `q · k` are smooth trigonometric sums. Without a sharpness mechanism, they are too close together and the softmax cannot distinguish them.

#### Why attention sharpness matters

If attention is diffuse, the model is essentially computing a uniform average of past values — it has no mechanism to ground the forecast on specific historical moments. For long-horizon forecasting, you want the model to:

- **Soften** attention when the past is noisy (use an average to be robust).
- **Sharpen** attention when the past has clear patterns (focus on the relevant seasonal lag).

The **temperature** parameter is the knob that controls this.

### 2.4 Temperature in Softmax

The standard softmax has an implicit temperature of 1:

```
softmax(z) = exp(z / 1) / sum exp(z / 1)
```

We can generalise:

```
softmax(z / τ) = exp(z / τ) / sum_j exp(z_j / τ)
```

#### What happens when τ is small

```
τ → 0+:  exp(z_i / τ) dominates on the largest z_i.
         Output becomes one-hot at argmax.
         Sharpest possible focus.
```

#### What happens when τ is large

```
τ → ∞:   exp(z_i / τ) ≈ exp(0) for all i.
         Output becomes uniform 1/N.
         Sharpest possible smoothing.
```

#### τ = 1 is the standard baseline

```
τ = 1:   exp(z_i) / sum exp(z_j).
         Standard softmax.
         What you get when you do not think about temperature.
```

#### Connecting to attention

For our cross-temporal attention:

```
A[b, h, o, l] = exp(S[b, h, o, l] / τ) / sum_j exp(S[b, h, o, j] / τ)
```

- Small τ → attention focuses on the most similar past step.
- Large τ → attention spreads over all past steps.
- τ = 1 → the default, "vanilla" D2Vformer behaviour.

**Why we use it:** Date2Vec inner products have small variance (they are smooth trigonometric sums). A small τ *amplifies* the differences and lets the model selectively attend to specific seasonal phases. A large τ *suppresses* noise from weak attention and behaves like a smoother.

### 2.5 Entropy

#### Intuition

**Entropy** measures how "spread out" a probability distribution is. A distribution with one entry of 1 and the rest 0 has entropy 0 (concentrated). A uniform distribution over N entries has entropy log(N) (maximally spread).

For a distribution `p = (p_1, ..., p_N)`:

```
H(p) = -sum_i p_i log(p_i)
```

- All probability on one entry: H = 0.
- Uniform: H = log(N).
- Somewhere in between: H ∈ (0, log(N)).

#### Normalised entropy

To compare across different sequence lengths, we divide by log(N):

```
H_norm(p) = H(p) / log(N)
```

- H_norm = 1 means perfectly uniform.
- H_norm = 0 means perfectly concentrated.

#### Why we use it

We use H_norm to measure *how diffuse* the attention distribution is. For L = 96:

- H_norm ≈ 0.97 means attention is ~97% of the way to uniform. Each past step gets ~1/96 ≈ 1.04% of the weight, with tiny deviations.
- H_norm ≈ 0.5 means attention is meaningfully focused on half the lookback.

For our project, this is the diagnostic that told us attention was diffuse.

### 2.6 Effective Number of Attended Timestamps

#### Intuition

How many past time steps are *effectively* contributing to a forecast?

```
N_eff = exp(H(p))
```

For uniform attention over 96 steps, H = log(96) ≈ 4.56, so N_eff = 96.
For one-hot attention, H = 0, so N_eff = 1.

#### Why we use it

N_eff is the human-readable counterpart of H_norm. They encode the same information:

```
H_norm = ln(N_eff) / ln(L)         (equivalently)
N_eff  = L^(H_norm)
```

We report both in our tables.

### 2.7 Parameter Count

#### Why parameter count matters

For a forecasting model to be **horizon-independent**, the number of learnable parameters must not depend on the forecast horizon O. Otherwise:

- A model trained at O = 24 cannot be evaluated at O = 720 (different parameter count, different shape).
- Memory and compute scale with O, making "arbitrary-horizon forecasting" meaningless.

#### What `dN_params / dO = 0` means

Mathematically: the partial derivative of the parameter count with respect to the horizon O is identically zero across the entire evaluation range.

For all our TCD2Vformer modes:

```
dN_params / dO ≡ 0   for   O ∈ {24, 48, 96, 192, 336, 720}
```

The audit verified this for 96 / 96 (dataset × mode × horizon) combinations.

#### How the official D2Vformer violates this

The official repository has a final projection `Linear(d_model, O)`. That linear layer has `d_model × O` weights — directly scaling with O. From ~44,000 parameters at O = 24 to ~132,000 at O = 720.

This was the first thing we fixed.

### 2.8 Putting It Together — What the Project Actually Does

```
[ Past time series ]                [ Future time coordinates ]
       │                                       │
       │                                       │
   RevIN                                 (calendar features)
       │                                       │
       │                                       │
  Linear(c_in -> d_model)              Date2Vec (harmonic embedding)
       │                                       │
       │                                       │
       │                              D_y  [B, d_model, O, 17]
       │                                       │
       │                          ┌────────────┘
       │                          ▼
       │              S = D_y · D_x^T / sqrt(17)
       │                                       │
       │                          A = softmax(S / τ)   ← temperature here
       │                                       │
       ▼                          ┌────────────┘
       │                          ▼
       └────────────────────► Y_tilde = A · T^T
                                       │
                                       ▼
                            FFN: Linear(d_model -> d_ff -> c_in)
                                       │
                                       ▼
                                  RevIN denorm
                                       │
                                       ▼
                              Y_out  [B, O, c_in]
```

That is the entire model. Every chapter that follows either explains a component of this diagram, justifies a design choice, or audits the result.

### 2.9 The Big Picture — Concepts Mapped to Chapters

| Concept | Used in | Chapter |
|---|---|---|
| Time series + tensor shapes | everywhere | Part I |
| Attention (Q, K, V) | Phase 2 diagnostics, Phase 6 architecture | Part II §5, Part III §8 |
| Softmax + temperature | Phase 3–6 experiments | Part III §9, Part IV §11 |
| Entropy + N_eff | Phase 2 diagnostics | Part III §8 |
| Parameter count + horizon independence | Phase 1 + Phase 6 verification | Part III §7, Part IV §13 |
| Date2Vec | Original paper + our reuse | Part II §4 |

You now have the full prerequisite toolkit.

---

**End of Part I.** Continue with [TEXTBOOK_PART_II.md](TEXTBOOK_PART_II.md).