import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pure_d2vformer import PureD2Vformer
from baselines.dlinear import DLinear
from baselines.persistence import PersistenceBaseline
from baselines.repository_d2vformer import create_repository_d2vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def run_pre_experiment_validation():
    print("=" * 80)
    print("PRE-EXPERIMENT RIGOROUS VALIDATION SUITE (TESTS A THROUGH I)")
    print("=" * 80)
    set_seed(42)
    device = 'cpu'
    
    # Test A: Import test
    print("\n--- Test A: Import Test ---")
    assert PureD2Vformer is not None
    assert DLinear is not None
    assert PersistenceBaseline is not None
    assert create_repository_d2vformer is not None
    assert get_data_loaders is not None
    print("[PASS] Test A: All classes and modules imported successfully.")
    
    # Test B: Dataset loading test
    print("\n--- Test B: Dataset Loading Test ---")
    tr_ett, _, _, meta_ett = get_data_loaders('ETTh1', seq_len=96, pred_len=48, batch_size=4)
    tr_exc, _, _, meta_exc = get_data_loaders('exchange', seq_len=96, pred_len=48, batch_size=4)
    assert meta_ett['num_variables'] == 7
    assert meta_exc['num_variables'] == 8
    print(f"[PASS] Test B: ETTh1 ({meta_ett['num_variables']} vars) and Exchange ({meta_exc['num_variables']} vars) loaded successfully.")
    
    # Test C: One-batch PureD2Vformer forward test
    print("\n--- Test C: One-Batch PureD2Vformer Forward Test ---")
    m_pure = PureD2Vformer(c_in=7, seq_len=96, d_model=128, d_ff=256, k_freq=16, dropout=0.05).to(device)
    bx, by, bxm, bym = next(iter(tr_ett))
    out, A = m_pure(bx, bxm, bym)
    assert out.shape == by.shape, f"Shape mismatch: {out.shape} vs {by.shape}"
    assert A.shape == (4, 128, 48, 96)
    print(f"[PASS] Test C: Forward pass succeeded. Output: {list(out.shape)}, Attention: {list(A.shape)}.")
    
    # Test D: Horizon test on same model instance
    print("\n--- Test D: Multi-Horizon Test (O in [24, 48, 96, 192, 336, 720]) ---")
    for O in [24, 48, 96, 192, 336, 720]:
        ym_dummy = torch.randn(4, O, 4)
        out_O, A_O = m_pure(bx, bxm, ym_dummy)
        assert out_O.shape == (4, O, 7), f"Mismatch for O={O}: {out_O.shape}"
        assert A_O.shape == (4, 128, O, 96), f"Attention mismatch for O={O}: {A_O.shape}"
        print(f"Horizon O={O:3d} -> Output: {list(out_O.shape)}, Attention: {list(A_O.shape)}")
    print("[PASS] Test D: All 6 horizons processed by the identical model instance.")
    
    # Test E: Backward pass test
    print("\n--- Test E: Backward Pass Test ---")
    m_pure.train()
    opt = torch.optim.Adam(m_pure.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = torch.nn.MSELoss()(out, by)
    loss.backward()
    opt.step()
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"[PASS] Test E: Backward pass and gradient step succeeded. Loss: {loss.item():.4f}")
    
    # Test F: Parameter count invariance test
    print("\n--- Test F: Parameter Count Invariance Test ---")
    p_count = sum(p.numel() for p in m_pure.parameters() if p.requires_grad)
    assert p_count == 44021, f"Parameter count mismatch: got {p_count}, expected 44,021"
    print(f"[PASS] Test F: Parameter count is exactly {p_count:,}.")
    
    # Test G & H: Checkpoint save/load and checksum invariance test
    print("\n--- Test G & H: Checkpoint Save/Load & Checksum Invariance Test ---")
    initial_hash = compute_parameter_checksum(m_pure)
    os.makedirs('scratch', exist_ok=True)
    temp_ckpt_path = 'scratch/val_test_ckpt.pt'
    torch.save({'model_state_dict': m_pure.state_dict(), 'checksum': initial_hash}, temp_ckpt_path)
    
    loaded_ckpt = torch.load(temp_ckpt_path, map_location='cpu')
    m_loaded = PureD2Vformer(c_in=7, seq_len=96, d_model=128, d_ff=256, k_freq=16, dropout=0.05)
    m_loaded.load_state_dict(loaded_ckpt['model_state_dict'])
    loaded_hash = compute_parameter_checksum(m_loaded)
    assert initial_hash == loaded_hash == loaded_ckpt['checksum'], "Checksum mismatch after save/load!"
    print(f"[PASS] Test G & H: Checksum verified identically ({initial_hash[:12]}...).")
    
    # Test I: One-batch baseline tests
    print("\n--- Test I: One-Batch Baseline Tests ---")
    # 1. DLinear
    dlinear_model = DLinear(seq_len=96, pred_len=48, c_in=7)
    dlinear_out = dlinear_model(bx)
    assert dlinear_out.shape == (4, 48, 7)
    print(f"[PASS] DLinear baseline output: {list(dlinear_out.shape)}.")
    
    # 2. Persistence
    persist_model = PersistenceBaseline()
    persist_out = persist_model(bx, pred_len=48)
    assert persist_out.shape == (4, 48, 7)
    print(f"[PASS] Persistence baseline output: {list(persist_out.shape)}.")
    
    # 3. Repository D2Vformer
    repo_model = create_repository_d2vformer(c_in=7, seq_len=96, pred_len=48, mask_spectrum=torch.tensor([0, 1]), d_model=128)
    repo_out = repo_model(bx, bxm, by, bym)
    assert repo_out.shape == (4, 48, 7)
    print(f"[PASS] Repository D2Vformer baseline output: {list(repo_out.shape)}.")
    
    print("\n" + "=" * 80)
    print("ALL VALIDATION TESTS A THROUGH I PASSED WITH 100% SUCCESS.")
    print("=" * 80)

if __name__ == '__main__':
    run_pre_experiment_validation()
