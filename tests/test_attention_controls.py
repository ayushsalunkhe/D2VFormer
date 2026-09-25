import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pure_d2vformer import PureD2Vformer
from experiments.uniform_attention_control import forward_with_uniform_attention
from experiments.shuffled_attention_control import forward_with_shuffled_attention
from utils.reproducibility import compute_parameter_checksum

def test_attention_controls():
    print("============================================================")
    print("UNIT TESTING ATTENTION CONTROL OPERATORS")
    print("============================================================")
    
    B, L, D = 2, 96, 7
    O = 48
    H = 128
    
    model = PureD2Vformer(c_in=D, seq_len=L, d_model=H)
    model.eval()
    initial_hash = compute_parameter_checksum(model)
    
    x = torch.randn(B, L, D)
    xm = torch.randn(B, L, 4)
    ym = torch.randn(B, O, 4)
    
    # 1. Test Uniform Attention Operator
    out_uni, A_uni = forward_with_uniform_attention(model, x, ym)
    assert out_uni.shape == (B, O, D), f"Uniform out shape mismatch: {out_uni.shape}"
    assert A_uni.shape == (B, H, O, L), f"Uniform attention shape mismatch: {A_uni.shape}"
    assert torch.allclose(A_uni, torch.full_like(A_uni, 1.0 / L)), "Uniform attention values are not strictly 1/L!"
    print("  [OK] Uniform attention operator verified: shapes and 1/L values verified.")
    
    # 2. Test Shuffled Attention Operator
    out_orig, A_orig = model(x, xm, ym)
    out_shuf, A_shuf = forward_with_shuffled_attention(model, x, xm, ym, shuffle_seed=42)
    assert out_shuf.shape == (B, O, D), f"Shuffled out shape mismatch: {out_shuf.shape}"
    assert A_shuf.shape == (B, H, O, L), f"Shuffled attention shape mismatch: {A_shuf.shape}"
    
    # Verify that A_shuf is a true permutation of A_orig along the historical dimension L
    sorted_orig, _ = torch.sort(A_orig, dim=-1)
    sorted_shuf, _ = torch.sort(A_shuf, dim=-1)
    assert torch.allclose(sorted_orig, sorted_shuf, atol=1e-5), "Shuffled attention values do not match original distribution values!"
    assert not torch.allclose(A_orig, A_shuf), "Shuffled attention is identical to original (shuffle failed)!"
    print("  [OK] Shuffled attention operator verified: exact distribution permutation verified.")
    
    # 3. Verify Parameter Checksum Immutability
    final_hash = compute_parameter_checksum(model)
    assert initial_hash == final_hash, "Attention control operators mutated model weights!"
    print("  [OK] Parameter checksum immutable: no weights were modified during control operations.")
    
    print("\nALL ATTENTION CONTROL UNIT TESTS PASSED.")

if __name__ == '__main__':
    test_attention_controls()
