import os
import sys
import torch
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.temperature_d2vformer import TemperaturePureD2Vformer
from utils.reproducibility import compute_parameter_checksum

def test_temperature_horizon_independence():
    print("============================================================")
    print("UNIT TESTING TEMPERATURE PURED2VFORMER HORIZON INDEPENDENCE")
    print("============================================================")
    
    B, L, D = 2, 96, 7
    eval_horizons = [24, 48, 96, 192, 336, 720]
    
    for mode in ['fixed', 'learnable']:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        model = TemperaturePureD2Vformer(
            c_in=D,
            seq_len=L,
            d_model=128,
            d_ff=256,
            k_freq=16,
            temperature_mode=mode,
            initial_temperature=0.5 if mode == 'fixed' else 1.0
        )
        
        expected_params = 44021 if mode == 'fixed' else 44022
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert actual_params == expected_params, f"Parameter count mismatch in {mode}: got {actual_params}, expected {expected_params}"
        print(f"  [OK] Parameter count verified: {actual_params:,} parameters (strictly expected).")
        
        # Verify no parameter shape contains any horizon O
        for name, p in model.named_parameters():
            for O in [24, 48, 192, 336, 720]:
                assert O not in p.shape, f"Parameter {name} has shape {p.shape} dependent on horizon {O}!"
        print("  [OK] Parameter shapes verified: strictly independent of all horizons.")
        
        # Test zero-shot evaluation across all 6 horizons on SAME model instance
        model.eval()
        h0 = compute_parameter_checksum(model)
        
        x = torch.randn(B, L, D)
        xm = torch.randn(B, L, 4)
        
        for O in eval_horizons:
            ym = torch.randn(B, O, 4)
            with torch.no_grad():
                out, A = model(x, xm, ym)
            
            assert out.shape == (B, O, D), f"Output shape mismatch for O={O}: got {out.shape}"
            assert A.shape == (B, 128, O, L), f"Attention shape mismatch for O={O}: got {A.shape}"
            assert abs(float(A.sum(dim=-1).mean().item()) - 1.0) < 1e-5, f"Attention does not sum to 1 for O={O}!"
            assert compute_parameter_checksum(model) == h0, f"Weights mutated during inference for O={O}!"
            
        print(f"  [OK] Zero-shot evaluation across all horizons {eval_horizons} passed with immutable checksum.")
        
        # Backward pass gradient check for learnable mode
        if mode == 'learnable':
            model.train()
            ym = torch.randn(B, 48, 4)
            target = torch.randn(B, 48, D)
            out, _ = model(x, xm, ym)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            
            assert model.tau_raw.grad is not None, "tau_raw received no gradient during backward pass!"
            grad_val = float(model.tau_raw.grad.item())
            assert not math.isnan(grad_val) and not math.isinf(grad_val), f"tau_raw gradient is invalid: {grad_val}"
            print(f"  [OK] Learnable temperature gradient verified: dLoss/d(tau_raw) = {grad_val:+.6f}")
            
    print("\n============================================================")
    print("ALL TEMPERATURE HORIZON INDEPENDENCE TESTS PASSED.")
    print("============================================================")

if __name__ == '__main__':
    test_temperature_horizon_independence()
