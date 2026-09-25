import unittest
import torch
import torch.nn as nn
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.tcd2vformer import TCD2Vformer
from utils.reproducibility import compute_parameter_checksum, verify_parameter_invariance

class TestTCD2VformerPhase6(unittest.TestCase):
    def setUp(self):
        self.c_in = 7
        self.seq_len = 96
        self.d_model = 128
        self.d_ff = 256
        self.k_freq = 16
        self.horizons = [24, 48, 96, 192, 336, 720]

    def test_modes_parameter_counts(self):
        """Verifies exact parameter counts for all modes."""
        # 1. Fixed
        m_fixed = TCD2Vformer(c_in=self.c_in, temperature_mode='fixed')
        self.assertEqual(m_fixed.get_parameter_count(), 44021)

        # 2. Learned Global (1 extra param)
        m_learned = TCD2Vformer(c_in=self.c_in, temperature_mode='learned_global')
        self.assertEqual(m_learned.get_parameter_count(), 44022)

        # 3. Temporal Context (17*16 + 16 + 16*1 + 1 = 305 extra params)
        m_ctx = TCD2Vformer(c_in=self.c_in, temperature_mode='temporal_context')
        self.assertEqual(m_ctx.get_parameter_count(), 44326)

        # 4. Query Conditioned (305 extra params)
        m_qry = TCD2Vformer(c_in=self.c_in, temperature_mode='query_conditioned')
        self.assertEqual(m_qry.get_parameter_count(), 44326)

    def test_horizon_independence(self):
        """Strictly tests that parameter count does NOT change with forecast horizon O."""
        for mode in ['fixed', 'learned_global', 'temporal_context', 'query_conditioned']:
            with self.subTest(mode=mode):
                model = TCD2Vformer(c_in=self.c_in, temperature_mode=mode)
                counts = model.assert_horizon_independence(eval_horizons=self.horizons)
                baseline = counts[self.horizons[0]]
                for O, count in counts.items():
                    self.assertEqual(count, baseline, f"Mode {mode}: Horizon {O} has count {count} != {baseline}")

    def test_forward_and_backward_gradient_flow(self):
        """Verifies forward pass shapes, attention normalization, and gradient propagation."""
        B, L, O = 2, 96, 48
        x = torch.randn(B, L, self.c_in)
        xm = torch.randn(B, L, 4)
        ym = torch.randn(B, O, 4)
        target = torch.randn(B, O, self.c_in)

        for mode in ['fixed', 'learned_global', 'temporal_context', 'query_conditioned']:
            with self.subTest(mode=mode):
                model = TCD2Vformer(c_in=self.c_in, temperature_mode=mode)
                model.train()
                out, A, tau = model(x, xm, ym, return_attention=True)

                self.assertEqual(out.shape, (B, O, self.c_in))
                self.assertEqual(A.shape, (B, self.d_model, O, L))

                # Check attention sums to 1.0 along L
                sums = A.sum(dim=-1)
                self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

                # Temperature must be strictly positive
                self.assertTrue(torch.all(tau > 0.0), f"Temperature in mode {mode} must be > 0")

                # Loss & backward
                loss = nn.MSELoss()(out, target)
                loss.backward()

                # Verify gradient flow into temperature parameters
                if mode == 'learned_global':
                    self.assertIsNotNone(model.tau_raw.grad)
                    self.assertNotEqual(model.tau_raw.grad.item(), 0.0)
                elif mode in ['temporal_context', 'query_conditioned']:
                    for param in model.temp_mlp.parameters():
                        self.assertIsNotNone(param.grad)
                        self.assertTrue(torch.any(param.grad != 0.0))

    def test_checkpoint_loading_and_checksum(self):
        """Verifies checkpoint saving, restoration, and SHA-256 parameter invariance."""
        model = TCD2Vformer(c_in=self.c_in, temperature_mode='query_conditioned')
        orig_checksum = compute_parameter_checksum(model)

        ckpt_dict = {
            'model_state_dict': model.state_dict(),
            'checksum': orig_checksum,
            'temperature_mode': 'query_conditioned'
        }

        # New instance
        loaded_model = TCD2Vformer(c_in=self.c_in, temperature_mode='query_conditioned')
        loaded_model.load_state_dict(ckpt_dict['model_state_dict'])

        self.assertTrue(verify_parameter_invariance(loaded_model, orig_checksum))

if __name__ == '__main__':
    unittest.main()
