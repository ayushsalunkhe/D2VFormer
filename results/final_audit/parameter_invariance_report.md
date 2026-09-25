# Parameter Invariance Audit Report

**Date of Audit:** 2026-09-23  
**Status:** **100% VERIFIED & PASSED**  
**Audit Artifact:** [`results/final_audit/parameter_invariance_audit.csv`](parameter_invariance_audit.csv)  

---

## 1. Executive Summary

This audit programmatically evaluates whether the parameter count of `TCD2Vformer` remains strictly invariant across all evaluated forecast horizons:
$$O \in \{24, 48, 96, 192, 336, 720\}$$

In the official D2Vformer implementation, an output projection layer `Linear(d_model, O)` or horizon-indexed projection was attached to the model, causing the total trainable parameter count to scale linearly with the forecast horizon $O$ (from ~44k parameters at $O=24$ up to ~132k parameters at $O=720$).

Our architectural reconstruction (`PureD2Vformer` and `TCD2Vformer`) completely eliminates horizon-dependent projection weights. This audit proves that across all 4 benchmark datasets, all 4 temperature modes, and all 6 evaluation horizons:
$$\frac{\partial N_{\text{params}}}{\partial O} \equiv 0$$

Every forward pass at any arbitrary horizon $O$ executes with an identical, fixed parameter footprint.

---

## 2. Programmatic Parameter Count Audit Table

| Dataset | Variable Count ($c_{\text{in}}$) | Mode | $O=24$ | $O=48$ | $O=96$ | $O=192$ | $O=336$ | $O=720$ | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | 7 | `fixed` ($\tau=1.0$) | 44,021 | 44,021 | 44,021 | 44,021 | 44,021 | 44,021 | **PASSED** |
| **ETTh1** | 7 | `learned_global` | 44,022 | 44,022 | 44,022 | 44,022 | 44,022 | 44,022 | **PASSED** |
| **ETTh1** | 7 | `temporal_context` | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | **PASSED** |
| **ETTh1** | 7 | `query_conditioned` | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | **PASSED** |
| **ETTh2** | 7 | `fixed` ($\tau=1.0$) | 44,021 | 44,021 | 44,021 | 44,021 | 44,021 | 44,021 | **PASSED** |
| **ETTh2** | 7 | `learned_global` | 44,022 | 44,022 | 44,022 | 44,022 | 44,022 | 44,022 | **PASSED** |
| **ETTh2** | 7 | `temporal_context` | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | **PASSED** |
| **ETTh2** | 7 | `query_conditioned` | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | **PASSED** |
| **ETTm1** | 7 | `fixed` ($\tau=1.0$) | 44,021 | 44,021 | 44,021 | 44,021 | 44,021 | 44,021 | **PASSED** |
| **ETTm1** | 7 | `learned_global` | 44,022 | 44,022 | 44,022 | 44,022 | 44,022 | 44,022 | **PASSED** |
| **ETTm1** | 7 | `temporal_context` | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | **PASSED** |
| **ETTm1** | 7 | `query_conditioned` | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | 44,326 | **PASSED** |
| **Exchange**| 8 | `fixed` ($\tau=1.0$) | 44,408 | 44,408 | 44,408 | 44,408 | 44,408 | 44,408 | **PASSED** |
| **Exchange**| 8 | `learned_global` | 44,409 | 44,409 | 44,409 | 44,409 | 44,409 | 44,409 | **PASSED** |
| **Exchange**| 8 | `temporal_context` | 44,713 | 44,713 | 44,713 | 44,713 | 44,713 | 44,713 | **PASSED** |
| **Exchange**| 8 | `query_conditioned` | 44,713 | 44,713 | 44,713 | 44,713 | 44,713 | 44,713 | **PASSED** |

Total audit records: 96 (4 datasets × 4 modes × 6 horizons).  
Pass rate: **96 / 96 (100.0%)**.

---

## 3. Structural & Architectural Inspection

### 3.1 Date2Vec Embeddings
The Date2Vec layer generates trigonometric frequency embeddings:
$$D = [ \sin(\omega_k t + \phi_k), \cos(\omega_k t + \phi_k) ]$$
The weight tensors $\omega \in \mathbb{R}^{d_{\text{date}} \times 4}$ and bias $\phi \in \mathbb{R}^{d_{\text{date}}}$ depend solely on calendar feature dimension (4) and embedding dimension $d_{\text{date}} = 16$. No tensor dimension depends on $O$.

### 3.2 Value Projection & Cross-Attention
The cross-temporal attention mechanism computes:
$$Q = D_y \in \mathbb{R}^{B \times H \times O \times d_k}$$
$$K = D_x \in \mathbb{R}^{B \times H \times L \times d_k}$$
$$V = W_v(X) \in \mathbb{R}^{B \times H \times L \times d_v}$$
The attention logits $S = Q K^\top / \sqrt{d_k} \in \mathbb{R}^{B \times H \times O \times L}$ are scaled by dynamic temperature $\tau$. The output is:
$$\text{Attn}(Q, K, V) = \text{Softmax}(S / \tau) \cdot V \in \mathbb{R}^{B \times H \times O \times d_v}$$
The projection weight $W_v$ has shape `[d_model, d_model]`, which is strictly independent of $O$.

### 3.3 Dynamic Temperature Network (`temp_mlp`)
The MLP for `temporal_context` and `query_conditioned` consists of:
- `Linear(17, 16)`: 17 input features (16 Date2Vec dimensions + 1 normalized step index $t_{\text{norm}}$) $\to$ 16 hidden units ($17 \times 16 + 16 = 288$ parameters).
- `GELU` activation.
- `Linear(16, 1)`: 16 hidden units $\to$ 1 scalar output ($16 \times 1 + 1 = 17$ parameters).
Total MLP parameters = $288 + 17 = 305$ parameters.
During forward execution for `query_conditioned`, the MLP is evaluated at each query step $o \in \{1, \dots, O\}$ via tensor broadcasting, producing $\tau \in \mathbb{R}^{B \times 1 \times O \times 1}$. The parameter count of the MLP remains fixed at **305** regardless of whether $O=24$ or $O=720$.

### 3.4 Decoder Output Layer
The final projection back to the multivariate time-series space uses:
$$\hat{Y} = W_{\text{out}}(\text{Dec\_out}) + b_{\text{out}}$$
where $W_{\text{out}} \in \mathbb{R}^{c_{\text{in}} \times d_{\text{model}}}$. It projects the hidden channel dimension $d_{\text{model}} = 128$ down to $c_{\text{in}} = 7$ (or 8), independently applied to each time step along the sequence dimension.
No linear layer projects across the time dimension $O$.

---

## 4. Conclusion
The audit confirms that `TCD2Vformer` is genuinely horizon-independent. Arbitrary prediction lengths can be requested at inference time without re-instantiating, re-initializing, or adding a single trainable parameter.
