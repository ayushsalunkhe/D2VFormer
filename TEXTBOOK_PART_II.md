# D2Vformer to TCD2Vformer — Part II: Original D2Vformer

> **Source of truth:** `PROJECT_KNOWLEDGE_MAP.md` §3–§6 and `models/pure_d2vformer.py`.
> **What belongs to whom:** anything derived directly from arXiv:2409.11024 (Wang et al.) is **inherited**. Anything we did to fix or modify it is **our reconstruction**.

---

## 3. The Original Research Paper

### 3.1 Citation

- **Title:** *D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding*
- **Authors:** Wang, Hao, et al.
- **Venue:** *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)*
- **arXiv:** 2409.11024
- **Official repository:** `TeamofHaoWang/D2Vformer` on GitHub

The paper introduces three contributions: (1) a Date2Vec-style time-position embedding, (2) a fusion block that combines values and temporal representations, and (3) a flexible-forecasting claim enabled by querying at arbitrary future time coordinates.

### 3.2 Motivation in the original paper

The paper's motivation:

- Most forecasting models (PatchTST, iTransformer, FEDformer, etc.) are trained for a single fixed horizon O. Changing O at deployment requires retraining or maintaining multiple heads.
- Continuous-time embeddings let the model query at any future time τ. The same trained model can forecast 24 hours, 48 hours, or 30 days ahead without retraining.

This is a real and valuable property. The paper's main conceptual contribution is **enabling** this, not necessarily proving it optimal in isolation.

### 3.3 Problem statement in the original paper

Given:
- A historical series `x ∈ ℝ^{B × L × c_in}`.
- Historical timestamps `x_mark ∈ ℝ^{B × L × M}`.
- Future timestamps `y_mark ∈ ℝ^{B × O × M}` (variable O).

Predict:
- `ŷ ∈ ℝ^{B × O × c_in}`.

The model must produce `ŷ` for any `O`, using the same trained weights.

### 3.4 Architecture in the original paper (high level)

```
x_enc  ─► RevIN ─► TFE Linear ─► temporal features T ─┐
                                                       │
x_mark ─► Date2Vec ─► D_x ────────────────────────────┤
                                                       ├──► Cross-Temporal Attention ─► Y_tilde ─► FFN ─► Y_hat
y_mark ─► Date2Vec ─► D_y ────────────────────────────┘                                                  │
                                                                                                         ▼
                                                                                                  RevIN denorm ─► ŷ
```

This is conceptually what the original paper proposes. The *exact* implementation in the official repository is what our Phase 1 audit examined.

### 3.5 Date2Vec (in the original paper)

The Date2Vec layer takes a calendar coordinate `t` and produces a vector of dimension `k_freq + 1`:

```
Date2Vec(t) = concat(
    t,
    sin(ω_1 * t + φ_1), ..., sin(ω_k * t + φ_k),
    cos(ω_1 * t + φ_1), ..., cos(ω_k * t + φ_k)
)
```

The frequencies `ω_i` and phases `φ_i` are **learned** — not fixed like in the original Time2Vec paper (Kazemi et al., 2019). The D2Vformer paper's contribution is to make them learnable jointly with the rest of the model.

### 3.6 Cross-temporal attention (in the original paper)

The paper's central equation:

```
A[b, h, o, l] = exp( ⟨ D_x(b, h, l, :), D_y(b, h, o, :) ⟩ / √(k_freq + 1) )
              / sum_l exp( ⟨ D_x(b, h, l, :), D_y(b, h, o, :) ⟩ / √(k_freq + 1) )
```

That is, each future timestamp `o` attends to each past timestamp `l` proportionally to their Date2Vec similarity. The paper assumes `τ = 1` (no explicit temperature).

The output is then the attention-weighted average of temporal features:

```
Y_tilde[b, h, o] = sum_l A[b, h, o, l] * T[b, l, h]
```

### 3.7 Encoder / decoder flow

The paper's "fusion block" combines:
1. Temporal features from `x_enc`.
2. Date2Vec embeddings from `x_mark` and `y_mark`.

The output is passed through a position-wise FFN and projected to `c_in` channels. There is **no explicit decoder** in the transformer sense; the cross-temporal attention itself acts as the decoder.

### 3.8 The flexible-forecasting claim

Because the cross-temporal attention is computed between Date2Vec embeddings (and these are continuous functions of time), the same trained model can in principle be evaluated at any O. The paper demonstrates this on ETT datasets.

### 3.9 What belongs to the original paper — exactly

| Concept | Source |
|---|---|
| The Date2Vec formulation (harmonic embedding with learnable ω, φ) | Original paper, §3 |
| Cross-temporal attention similarity | Original paper, §3 |
| TFE Linear(c_in → d_model) | Original paper, §3 |
| RevIN normalisation | Kim et al. ICLR 2022 (inherited by the paper) |
| Output projection `Linear(d_model, O)` | Official repository implementation (NOT a paper claim) |
| Flexible / arbitrary-horizon forecasting claim | Original paper claim |

**Defense point:** we did not invent flexible forecasting. We fixed its parameter-independence flaw. (See `PROJECT_KNOWLEDGE_MAP.md` §1.3.)

---

## 4. Date2Vec Explained From Scratch

Date2Vec is the most important concept in this entire project. Take your time here.

### 4.1 Why timestamps are useful

Forecasting depends on **when** something is predicted, not just **what** came before. Two observations taken at 3 AM and 3 PM have different predictive meaning even if their values are identical — one precedes the morning peak, the other follows it.

To use this information, the model needs a vector representation of "3 AM on a Wednesday in October" that is:

- **Continuous** — small changes in time produce small changes in the vector.
- **Periodic** — equivalent times (3 AM today and 3 AM tomorrow) produce similar vectors.
- **Multi-scale** — daily, weekly, and seasonal cycles are all captured.

A raw timestamp `(2026, 10, 15, 3, 30)` does not satisfy any of these.

### 4.2 Why raw timestamps are insufficient

Consider encoding time as `[hour_of_day, day_of_week, day_of_year, month]`.

- `hour_of_day ∈ {0, ..., 23}` — not continuous; hour 23 and hour 0 are far apart numerically but adjacent in time.
- `day_of_week ∈ {0, ..., 6}` — same problem; Sunday and Monday are different categories, not periodic.
- `day_of_year ∈ {0, ..., 365}` — January 1 and December 31 are far apart numerically but conceptually adjacent in the annual cycle.

A neural network can learn nonlinear embeddings for these, but it would need to *rediscover* that `sin` and `cos` are the natural representation of periodic variables.

**Date2Vec skips this learning step by hard-coding the periodic structure via sin/cos.**

### 4.3 Periodicity

Many natural cycles are periodic:

- **Hour of day** — period 24 hours.
- **Day of week** — period 7 days.
- **Week of year** — period ~52 weeks.
- **Year cycle** — period ~365 days.

A periodic function `f(t)` satisfies `f(t + T) = f(t)`. The simplest non-trivial periodic functions are `sin(ω t)` and `cos(ω t)`, where `ω = 2π / T` is the angular frequency.

### 4.4 Harmonic representation

**A sum of harmonics with different frequencies can approximate any periodic function** (Fourier theorem). Date2Vec uses this:

```
f(t) ≈ a_0 + Σ_k (a_k * cos(ω_k * t) + b_k * sin(ω_k * t))
```

If we let the model *learn* the amplitudes `a_k, b_k` and the frequencies `ω_k`, the model can adaptively choose which periodicities matter for the data.

### 4.5 The Date2Vec equation

For a normalised timestamp `t ∈ ℝ`, Date2Vec with `k_freq` harmonics produces:

```
D(t) = [ t, sin(ω_1 t + φ_1), sin(ω_2 t + φ_2), ..., sin(ω_k t + φ_k),
              cos(ω_1 t + φ_1), cos(ω_2 t + φ_2), ..., cos(ω_k t + φ_k) ]
```

In our project, the implementation collapses sin and cos into a single dimension each (via the phase shift φ_k), giving:

```
D(t) = [ t, sin(ω_1 t + φ_1), sin(ω_2 t + φ_2), ..., sin(ω_k t + φ_k) ]
       ─┬─   ──────────────────────────┬──────────────────────────
        │                              │
   linear component            k_freq harmonic components
```

The output dimension is `k_freq + 1`. With `k_freq = 16`, the dimension is **17**.

### 4.6 Frequency, phase, amplitude

- **Frequency ω_k** — how fast this harmonic oscillates. Large ω_k = short period = high-frequency pattern. Small ω_k = long period = low-frequency pattern.
- **Phase φ_k** — where in the cycle this harmonic starts at t = 0.
- **Amplitude** — folded into the value of `Ω_S · T` (see §4.7). The "weight" of each frequency in the final representation.

The model has `k_freq = 16` frequencies, each with its own phase, allowing it to capture 16 different periodicities at once.

### 4.7 How the parameters are computed

In our code, the harmonics are produced by:

```
Ω_S = W_S · T + B_S         # [B, k_freq, d_model]
```

- `T` ∈ ℝ^{B × L × d_model} is the temporal feature projection of `x_enc`.
- `W_S` ∈ ℝ^{k_freq × L} is the harmonic frequency/phase parameter (learned).
- `B_S` ∈ ℝ^{k_freq × d_model} is the harmonic bias (learned).
- `Ω_S` ∈ ℝ^{B × k_freq × d_model} is the per-frequency "phase signal".

Then for any timestamp `t`:

```
D(b, k, h) = sin(Ω_S[b, k, h] * t + B_2[k, h])
```

where `B_2` is another learned phase parameter.

**Intuition:** the model learns which frequencies of `t` matter, then applies sin to them.

### 4.8 Examples

#### Example 1: Hour of day

If `t = hour_of_day / 24 ∈ [0, 1)`, then `sin(ω t + φ)` with `ω = 2π` gives a full daily cycle. The model can learn `ω ≈ 2π` and `φ ≈ 0`, capturing the 24-hour periodicity.

#### Example 2: Day of week

If `t = day_of_week / 7`, then `ω = 2π × 7 ≈ 14π` gives a weekly cycle. The model can learn this for weekly patterns.

#### Example 3: Multi-frequency

A single model simultaneously learns hourly, daily, weekly, and seasonal patterns because it has 16 different frequency parameters.

### 4.9 Why Date2Vec works for our project

Date2Vec is what makes zero-shot forecasting possible:

- The model is trained at O_train = 48 with timestamp queries up to 48 hours ahead.
- At inference, we query at O = 720 — 30 days ahead.
- Date2Vec's periodic embeddings extrapolate smoothly. A query at `t = 30 days` is represented in the same harmonic basis as `t = 1 day`, just shifted in phase.

If the model used *positional indices* (the i-th step) instead of timestamps, it would have no way to extrapolate.

### 4.10 The full mathematical formulation

For a single timestamp vector `x_mark = (year_norm, month_norm, day_norm, weekday_norm)`, the full Date2Vec computation is:

```
Linear part:     E_lin(b, h, l, m) = v_T(b, h) · x_mark(b, l, m) + b_1(h)
                 where v_T(b, h) = w_T · T(b, :, h) + b_T(h)
                 → shape [B, d_model, L, M]

Harmonic part:   E_har(b, k, h, l, m) = sin(Ω_S(b, k, h) · x_mark(b, l, m) + B_2(k, h))
                 → shape [B, k_freq, d_model, L, M]

Concatenation:   E(b, k+1, h, l, m) = concat(E_lin, E_har, dim=k)
                 → shape [B, k_freq + 1, d_model, L, M]

Aggregation:     D_x(b, k+1, h, l) = mean over m of E(b, k+1, h, l, m)
                 → shape [B, k_freq + 1, d_model, L]
```

The mean over `m` (the calendar features) collapses the time marker dimension. This is a design choice — the original paper averages them; we follow that.

### 4.11 D_x and D_y

- `D_x ∈ ℝ^{B × (k+1) × d_model × L}` — Date2Vec of every past timestamp.
- `D_y ∈ ℝ^{B × (k+1) × d_model × O}` — Date2Vec of every future timestamp.

These are the inputs to cross-temporal attention.

### 4.12 Intuition recap

> Date2Vec is a learned Fourier basis. Instead of fixing `ω_k = 2π / T_k` for known periods, the model learns the frequencies. The harmonic basis allows smooth, periodic extrapolation — the very property that makes zero-shot forecasting possible.

---

## 5. Original D2Vformer Architecture

This section traces one forward pass through the model.

### 5.1 Stage 1: Inputs

```
x_enc        ∈ ℝ^{B × L × c_in}       # historical values
x_mark_enc   ∈ ℝ^{B × L × M}           # historical calendar (M=4)
y_mark_dec   ∈ ℝ^{B × O × M}           # future calendar
```

#### Intuition

`x_enc` is the actual measurements. `x_mark_enc` and `y_mark_dec` are the calendar coordinates.

### 5.2 Stage 2: RevIN normalisation

```
x_norm = RevIN(x_enc, 'norm')        # [B, L, c_in]
```

RevIN = Reversible Instance Normalisation (Kim et al., ICLR 2022). It computes the mean and standard deviation over the lookback window per channel and normalises:

```
x_norm[b, l, d] = (x_enc[b, l, d] - mean_l(x_enc[b, :, d])) / std_l(x_enc[b, :, d])
```

The mean and std are stored so they can be inverted at the end.

#### Why

Time series have non-stationary scale. A model that sees raw values must learn to subtract the local mean. RevIN bakes this in.

#### Intuition

"Each input is rescaled to have zero mean and unit variance over the lookback window."

### 5.3 Stage 3: Temporal Feature Extraction (TFE)

```
T = Linear(c_in → d_model)(x_norm)   # [B, L, d_model]
```

A linear layer projects the normalised values into a `d_model = 128`-dimensional space.

#### Why

The model needs a high-dimensional representation of values. `d_model = 128` is large enough for expressive attention but small enough to be tractable.

### 5.4 Stage 4: Date2Vec

Using the formulas from §4.10:

```
D_x = Date2Vec(x_mark_enc) ∈ ℝ^{B × (k+1) × d_model × L}
D_y = Date2Vec(y_mark_dec) ∈ ℝ^{B × (k+1) × d_model × O}
```

These are the **past** and **future** temporal embeddings.

### 5.5 Stage 5: Cross-temporal similarity

Permute `D_x` and `D_y` to align the inner-product dimension:

```
D_x_tilde ∈ ℝ^{B × d_model × L × (k+1)}
D_y_tilde ∈ ℝ^{B × d_model × O × (k+1)}
```

Compute:

```
S = D_y_tilde · D_x_tilde^T / sqrt(k+1)   # [B, d_model, O, L]
```

That is, `S[b, h, o, l]` is the dot product between the Date2Vec of future step `o` and the Date2Vec of past step `l`, scaled by `1/sqrt(k+1) ≈ 0.24`.

#### Why the sqrt scaling

Dot products of high-dimensional vectors have large variance, which pushes softmax into saturation. Dividing by sqrt(d_k) normalises the variance to ~1, keeping softmax in a useful operating range.

#### Intuition

`S[b, h, o, l]` measures **how similar** the calendar of future step `o` is to the calendar of past step `l`. Large `S` means: *"this past step's calendar looks like the future step's calendar."*

### 5.6 Stage 6: Attention

```
A = softmax(S, dim=-1)   # [B, d_model, O, L]
```

Each row `A[b, h, o, :]` is a probability distribution over past time steps `l ∈ {1, ..., L}`.

In the TCD2D2Vformer extension, this becomes:

```
A = softmax(S / τ, dim=-1)
```

with `τ` either fixed (1.0), learned globally, or computed from the Date2Vec embeddings.

### 5.7 Stage 7: Temporal value aggregation

```
T_transposed ∈ ℝ^{B × d_model × L}
Y_tilde = einsum('bhol, bhl -> bho', A, T_transposed)   # [B, d_model, O]
```

For each future step `o`, take the attention-weighted average of past temporal features `T`.

#### Intuition

`Y_tilde[b, h, o]` is a weighted average of all past temporal features, where the weights are `A[b, h, o, :]`. Future step `o` gets a value that is the weighted sum of past values that "look like" `o` in calendar terms.

### 5.8 Stage 8: Feed-forward projection

```
Y_hat = FFN(Y_tilde.transpose(1, 2))   # [B, O, c_in]
```

where `FFN` is:

```
FFN(x) = Linear(d_ff → c_in)(GELU(Linear(d_model → d_ff)(x)))
```

The `d_model = 128` dimension is projected through `d_ff = 256` and back down to `c_in ∈ {7, 8}`.

#### Why a position-wise FFN

Standard transformer practice: attention captures temporal relationships, FFN adds pointwise nonlinearity.

### 5.9 Stage 9: RevIN denormalisation

```
Y_out = RevIN(Y_hat, 'denorm')   # [B, O, c_in]
```

Apply the inverse of the normalisation, using the stored mean and std.

### 5.10 The full data flow

```
x_enc ─► RevIN ─► x_norm ─► TFE ─► T ─┐
                                        │
x_mark ─► Date2Vec ─► D_x ──────────────┤
                                        ├──► S ─► A ─► Y_tilde ─► FFN ─► Y_hat ─► RevIN denorm ─► Y_out
y_mark ─► Date2Vec ─► D_y ──────────────┘
```

### 5.11 Tensor shapes summary

| Symbol | Shape | Source |
|---|---|---|
| `x_enc` | [B, L, c_in] | input |
| `x_norm` | [B, L, c_in] | RevIN |
| `T` | [B, L, d_model] | TFE |
| `D_x` | [B, k+1, d_model, L] | Date2Vec |
| `D_y` | [B, k+1, d_model, O] | Date2Vec |
| `S` | [B, d_model, O, L] | dot product |
| `A` | [B, d_model, O, L] | softmax |
| `Y_tilde` | [B, d_model, O] | aggregation |
| `Y_hat` | [B, O, c_in] | FFN |
| `Y_out` | [B, O, c_in] | RevIN denorm |

Where:
- `B` = batch size.
- `L = 96` = lookback.
- `O ∈ {24, 48, 96, 192, 336, 720}` = forecast horizon.
- `c_in = 7` (ETT) or `8` (Exchange).
- `d_model = 128`.
- `k_freq = 16`, so `k+1 = 17`.

### 5.12 What's special about this architecture

The unusual features:

1. **No encoder-decoder.** Attention itself plays the decoder role.
2. **The "values" being aggregated are the temporal features of past inputs**, not separate value embeddings. This is unusual compared to vanilla transformer attention.
3. **No causal mask.** Every future query can attend to every past step. The temporal structure comes from the Date2Vec similarity, not from masking.
4. **The output projection is the FFN, which projects across channels, not time.** No horizon-dependent layer exists.

### 5.13 What the original paper got wrong

The official repository had:

```python
self.project = nn.Linear(self.d_model, self.pred_len)   # pred_len = O
```

This is a `Linear(d_model, O)` projection. The output dimension depends on O. The number of parameters scales as `d_model × O`:

| O | Parameters in this layer alone |
|---|---|
| 24 | 24 × 128 = 3,072 |
| 48 | 48 × 128 = 6,144 |
| 96 | 96 × 128 = 12,288 |
| 192 | 192 × 128 = 24,576 |
| 336 | 336 × 128 = 43,008 |
| 720 | 720 × 128 = 92,160 |

Total model parameters:

| O | Total params |
|---|---|
| 24 | ~44,000 |
| 720 | ~132,000 |

This violates the "fixed parameter count" claim of the paper.

**This was Phase 1's discovery.**

---

## 6. The Zero-Shot Horizon Concept

### 6.1 The problem this concept solves

Most forecasting models train one model per horizon:

```
Model_O24:  forecast 24 steps ahead
Model_O48:  forecast 48 steps ahead
...
```

This is wasteful and inflexible. If you want to forecast 36 steps, you can't.

### 6.2 Training horizon vs evaluation horizon

- **Training horizon O_train:** the horizon the model is trained on. In our project: `O_train = 48`.
- **Evaluation horizon O_eval:** any horizon the model is tested on. In our project: `{24, 48, 96, 192, 336, 720}`.

A model is **zero-shot** for horizon O if it can be evaluated at O without retraining or fine-tuning.

### 6.3 Why horizon independence matters

If the parameter count depends on O, then:

- You cannot evaluate at an unseen O.
- Memory grows with O.
- Compute grows with O.
- The "fixed parameter count" claim is false.

Horizon independence means the model is truly a function of the timestamps, not of the horizon.

### 6.4 How the architecture enables zero-shot

Cross-temporal attention is computed between Date2Vec embeddings. These embeddings are continuous functions of time. Therefore:

```
forecast_at_time_τ_0 = model(x_past, τ_0)     for any τ_0 ∈ future
forecast_at_time_τ_1 = model(x_past, τ_1)     for any τ_1 ∈ future
```

The only thing that changes between evaluations is `y_mark_dec` (the future timestamps). All model weights stay the same.

### 6.5 The parameter-scaling issue discovered in the official implementation

The official `model/D2Vformer.py` (inherited from the base repo, *not* the audited chain) contained:

```python
self.trend_linear_decoder = nn.Linear(self.d_model, self.pred_len)
self.project = nn.Linear(self.d_model, self.pred_len)
```

These are `Linear(d_model, O)` projections. They violate horizon independence.

Our `models/pure_d2vformer.py` removes these and replaces the final aggregation with `Y = A · T^T` — a parameter-free operation.

### 6.6 Why this matters scientifically

If the parameter count grows with O, then:

- Comparison with a fixed-parameter baseline is unfair.
- Claims of memory/compute efficiency are invalid.
- Zero-shot evaluation is technically possible (you can call `forward` with new shapes), but you cannot re-use the trained weights without architectural modification.

A truly horizon-independent model:

- Has the same parameter count at every O.
- Can be trained once and deployed at any O.
- Has a clean `O` axis for theoretical analysis.

### 6.7 Intuition recap

> Zero-shot horizon means "train once, query at any future time." The official D2Vformer repository claimed this but had a hidden `Linear(d_model, O)` that scaled with O. Our reconstruction removed this dependency, restoring the claim's validity.

---

**End of Part II.** Continue with [TEXTBOOK_PART_III.md](TEXTBOOK_PART_III.md).