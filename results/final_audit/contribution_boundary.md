# Novelty and Contribution Boundary Audit

**Date of Audit:** 2026-09-23  
**Status:** **ACADEMICALLY BOUNDED & VERIFIED**  
**Document:** `results/final_audit/contribution_boundary.md`  

---

## 1. Delineation of Project Contributions

To ensure academic honesty during the BE thesis examination, every component of this codebase is explicitly categorized by provenance:

| Category | Component / Feature | Provenance | Detailed Description |
| :--- | :--- | :--- | :--- |
| **INHERITED FROM D2VFORMER** | Date2Vec Formulation | Wang et al. (IEEE TNNLS / arXiv:2409.11024) | Trigonometric frequency embedding of calendar timestamps using sin/cos functions. |
| | Flexible Forecasting Concept | Wang et al. (2024) | Conceptual goal of arbitrary-length forecasting using continuous temporal query coordinates. |
| | RevIN Normalization | Kim et al. (ICLR 2022) | Reversible instance normalization applied to time series input channels. |
| | Benchmark Datasets | Zhou et al. (AAAI 2021) | ETTh1, ETTh2, ETTm1, and Exchange Rate benchmarks. |
| **PROJECT RECONSTRUCTION** | Parameter-Free PureD2Vformer | **Our Reconstruction (Phase 1)** | Elimination of horizon-dependent linear projection layers (`Linear(d_model, O)`), achieving genuine mathematical parameter independence: $\partial N_{\text{params}} / \partial O \equiv 0$. |
| | Chronological Data Split Audit | **Our Audit (Phase 1)** | Identification and correction of data split leakage and inconsistent test windowing in open implementations. |
| **DIAGNOSTIC CONTRIBUTION** | Attention Entropy Audit | **Our Diagnostic (Phase 2)** | Discovery that cross-temporal Date2Vec attention operates near maximum entropy ($H_{\text{norm}} \approx 0.97$), behaving as an unweighted uniform smoother. |
| | Counterfactual Control Tests | **Our Diagnostic (Phase 2)** | Uniform substitution and position-shuffling controls quantifying the exact informational utility of cross-temporal attention. |
| **EMPIRICAL CONTRIBUTION** | Validation-Selected Temperature | **Our Contribution (Phases 3–4)** | Formal proof that temperature $\tau$ must be selected exclusively via validation loss at $O_{\text{train}}=48$, rejecting test-set lookahead and disproving universal $\tau=4.0$. |
| | Multi-Horizon Cross-Dataset Matrix | **Our Contribution (Phases 3–6)** | Pre-registered evaluation across 4 datasets, 3 seeds, and 6 horizons ($O \in [24, 720]$) with locked test evaluations and checksum verification. |
| **ARCHITECTURAL CONTRIBUTION** | TCD2Vformer Architecture | **Our Contribution (Phase 6)** | Introduction of dynamic, input-conditioned and query-conditioned temperature modulation: $\tau(X) = \text{MLP}(\bar{d}_x)$ and $\tau(t) = \text{MLP}(\bar{d}_{y, t})$. |
| | Horizon-Invariant Temperature MLP | **Our Contribution (Phase 6)** | Mathematical formulation ensuring dynamic temperature generation adds exactly 305 parameters regardless of forecast horizon. |
| **ENGINEERING / IMPLEMENTATION** | Reproducibility & Automation Suite | **Our Engineering (Phases 5–6)** | SHA-256 state-dict checksum verification, automated evaluation runners, interactive standalone browser demo (`demo/index.html`), and turnkey Google Colab GPU suite. |

---

## 2. Precise Formulation of the Core Contribution

### Inappropriate / Overstated Claims (DO NOT USE IN THESIS):
- ❌ *"We invented arbitrary-length zero-shot time-series forecasting."* (False: D2Vformer introduced the concept; our work fixed its parameter-dependence flaw).
- ❌ *"TCD2Vformer is the first-ever dynamic attention temperature mechanism in machine learning."* (False: Dynamic temperature has appeared in vision and NLP literature, e.g., temperature-scaled softmax in distillation and calibration).
- ❌ *"TCD2Vformer universally outperforms all baselines across all time series."* (False: As documented in Phase 6, high-frequency series like ETTm1 degrade at long horizons due to phase overfitting).

### Accurate / Defensible Formulation (USE IN THESIS & DEFENSE):
> *"We identify that D2Vformer's cross-temporal Date2Vec attention operates near uniform entropy, and that the original repository relied on horizon-dependent projection weights. We reconstruct a strictly horizon-independent PureD2Vformer ($\partial N_{\text{params}}/\partial O \equiv 0$) and propose **Temporal-Conditioned Date2Vecformer (TCD2Vformer)**, which dynamically modulates attention sharpness via lightweight temporal embeddings. We demonstrate that dynamic temperature modulation mitigates attention diffusion and achieves significant zero-shot forecasting gains at extended horizons (up to +3.12% MSE reduction at $O=720$ on ETTh2), while identifying the empirical boundary conditions of attention sharpening in high-frequency noise regimes."*

---

## 3. Claims Requiring Literature Verification Prior to Thesis Submission

Before finalizing Section 2 (Related Work) of the BE thesis, verify:
1. **Dynamic Attention Temperature Prior Art:** Cite prior temperature modulation papers in NLP (e.g., Learnable Temperature in Transformers, e.g., Vaswani et al. scaling, or adaptive softmax temperatures). Emphasize that our specific novelty lies in *conditioning temperature on continuous temporal phase embeddings (Date2Vec) under a horizon-invariant constraint*.
2. **Reconstruction Distinction:** Make clear that "PureD2Vformer" refers to our parameter-free reconstruction of Wang et al.'s concept, avoiding confusion with their official codebase release.
3. **Statistical Power Disclaimer:** Explicitly acknowledge that results are evaluated across $n=3$ seeds, providing descriptive effect sizes rather than formal asymptotic significance.
