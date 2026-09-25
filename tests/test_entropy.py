import torch
import math
import numpy as np

def compute_entropy_and_metrics(A: torch.Tensor, eps: float = 1e-12):
    """
    Computes Shannon entropy and related dispersion metrics across the historical dimension L.
    Input A: shape (..., L) where sum_l A[..., l] == 1.
    Returns:
        H: Shannon entropy in nats
        H_norm: Normalized Shannon entropy H / ln(L) in [0, 1]
        N_eff: Effective number of positions exp(H)
        N_eff_norm: N_eff / L
        max_val: max attention weight along L
        var_val: variance along L
    """
    L = A.shape[-1]
    # Verify sum to 1
    sums = torch.sum(A, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Attention does not sum to 1! max dev: {torch.max(torch.abs(sums - 1.0)).item()}"
    
    # H = -sum(A * ln(A + eps))
    H_tensor = -torch.sum(A * torch.log(A + eps), dim=-1)
    H = float(H_tensor.mean().item())
    
    log_L = math.log(L) if L > 1 else 1.0
    H_norm = H / log_L if L > 1 else 0.0
    N_eff = float(torch.exp(H_tensor).mean().item())
    N_eff_norm = N_eff / L if L > 1 else 1.0
    
    max_val = float(torch.max(A, dim=-1).values.mean().item())
    var_val = float(torch.var(A, dim=-1, unbiased=False).mean().item())
    
    return {
        'H': H,
        'H_norm': H_norm,
        'N_eff': N_eff,
        'N_eff_norm': N_eff_norm,
        'max_attention': max_val,
        'var_attention': var_val,
        'std_attention': math.sqrt(var_val),
        'L': L
    }

def test_entropy_implementation():
    L = 96
    B, H, O = 2, 4, 48
    print(f"============================================================")
    print(f"RUNNING DIAGNOSTIC 2: ENTROPY IMPLEMENTATION VERIFICATION")
    print(f"Historical Sequence Length L = {L}")
    print(f"============================================================")
    
    # 1. Perfectly Uniform Distribution
    A_uniform = torch.full((B, H, O, L), 1.0 / L, dtype=torch.float32)
    m_uni = compute_entropy_and_metrics(A_uniform)
    print(f"\n[Case 1: Perfectly Uniform (1/L)]")
    print(f"  H: {m_uni['H']:.6f} | ln(L): {math.log(L):.6f}")
    print(f"  H_norm: {m_uni['H_norm']:.6f} (Expected: 1.000000)")
    print(f"  N_eff: {m_uni['N_eff']:.2f} (Expected: {L})")
    print(f"  Max A: {m_uni['max_attention']:.6f} (Expected: {1/L:.6f})")
    assert abs(m_uni['H_norm'] - 1.0) < 1e-4, f"Uniform H_norm mismatch: {m_uni['H_norm']}"
    assert abs(m_uni['N_eff'] - L) < 1e-2, f"Uniform N_eff mismatch: {m_uni['N_eff']}"
    print("  [OK] PASS: Uniform distribution verified perfectly.")

    # 2. Perfectly One-Hot Distribution
    A_onehot = torch.zeros((B, H, O, L), dtype=torch.float32)
    A_onehot[..., 0] = 1.0
    m_one = compute_entropy_and_metrics(A_onehot)
    print(f"\n[Case 2: One-Hot (delta distribution)]")
    print(f"  H: {m_one['H']:.6f} (Expected: ~0.000000)")
    print(f"  H_norm: {m_one['H_norm']:.6f} (Expected: 0.000000)")
    print(f"  N_eff: {m_one['N_eff']:.2f} (Expected: 1.00)")
    print(f"  Max A: {m_one['max_attention']:.6f} (Expected: 1.000000)")
    assert abs(m_one['H_norm'] - 0.0) < 1e-4, f"One-hot H_norm mismatch: {m_one['H_norm']}"
    assert abs(m_one['N_eff'] - 1.0) < 1e-2, f"One-hot N_eff mismatch: {m_one['N_eff']}"
    print("  [OK] PASS: One-hot distribution verified perfectly.")

    # 3. Moderately Concentrated (e.g. Gaussian-like peak over ~5 positions)
    indices = torch.arange(L, dtype=torch.float32)
    center = L / 2.0
    sigma = 3.0 # ~99% within 3 sigma = 9 positions
    gauss = torch.exp(-0.5 * ((indices - center) / sigma) ** 2)
    gauss = gauss / gauss.sum()
    A_conc = gauss.view(1, 1, 1, L).repeat(B, H, O, 1)
    m_conc = compute_entropy_and_metrics(A_conc)
    print(f"\n[Case 3: Moderately Concentrated (Gaussian sigma=3.0)]")
    print(f"  H: {m_conc['H']:.4f}")
    print(f"  H_norm: {m_conc['H_norm']:.4f} (Expected: roughly 0.5 - 0.7)")
    print(f"  N_eff: {m_conc['N_eff']:.2f} positions")
    print(f"  Max A: {m_conc['max_attention']:.4f}")
    assert 0.0 < m_conc['H_norm'] < 1.0, f"Concentrated H_norm out of bounds: {m_conc['H_norm']}"
    assert 1.0 < m_conc['N_eff'] < L, f"Concentrated N_eff out of bounds: {m_conc['N_eff']}"
    print("  [OK] PASS: Concentrated distribution verified.")

    # 4. Random Softmax (Simulating typical uncalibrated logits)
    torch.manual_seed(42)
    logits = torch.randn(B, H, O, L) * 0.1 # Small variance logits -> near uniform
    A_soft_diffuse = torch.softmax(logits, dim=-1)
    m_soft_diffuse = compute_entropy_and_metrics(A_soft_diffuse)
    print(f"\n[Case 4A: Diffuse Random Softmax (std=0.1)]")
    print(f"  H_norm: {m_soft_diffuse['H_norm']:.4f}")
    print(f"  N_eff: {m_soft_diffuse['N_eff']:.2f} / {L}")

    logits_sharp = torch.randn(B, H, O, L) * 5.0 # High variance logits -> sharp
    A_soft_sharp = torch.softmax(logits_sharp, dim=-1)
    m_soft_sharp = compute_entropy_and_metrics(A_soft_sharp)
    print(f"\n[Case 4B: Sharp Random Softmax (std=5.0)]")
    print(f"  H_norm: {m_soft_sharp['H_norm']:.4f}")
    print(f"  N_eff: {m_soft_sharp['N_eff']:.2f} / {L}")
    print(f"  Max A: {m_soft_sharp['max_attention']:.4f}")
    assert m_soft_sharp['H_norm'] < m_soft_diffuse['H_norm'], "Sharp softmax should have lower entropy than diffuse!"
    print("  [OK] PASS: Random softmax scaling verified.")

    print("\n============================================================")
    print("ALL SYNTHETIC ENTROPY TESTS PASSED WITH STRICT MATHEMATICAL ACCURACY.")
    print("============================================================")

if __name__ == '__main__':
    test_entropy_implementation()
