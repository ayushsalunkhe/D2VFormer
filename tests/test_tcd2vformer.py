import os
import sys
import torch
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import TCD2Vformer
from utils.reproducibility import compute_parameter_checksum

class TestTCD2VformerHorizonIndependence(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.L = 96
        self.D = 7
        self.eval_horizons = [24, 48, 96, 192, 336, 720]
        self.model = TCD2Vformer(
            c_in=self.D,
            seq_len=self.L,
            d_model=128,
            d_ff=256,
            k_freq=16,
            dropout=0.05,
            temperature_mode='fixed',
            initial_temperature=2.0
        )
        self.model.eval()

    def test_parameter_count(self):
        """Verify model has exactly 44,021 parameters, independent of O."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_params, 44021)

    def test_horizon_independent_parameter_shapes(self):
        """Verify no parameter shape depends on the forecast horizon O."""
        # For horizons not equal to lookback length L=96
        for name, param in self.model.named_parameters():
            for O in [24, 48, 192, 336, 720]:
                self.assertNotIn(O, param.shape, f"Parameter {name} shape {param.shape} depends on horizon {O}")
        
        # Test with alternative lookback L=104 to confirm O=96 is never in shapes
        alt_model = TCD2Vformer(c_in=self.D, seq_len=104, d_model=128, d_ff=256, k_freq=16)
        for name, param in alt_model.named_parameters():
            for O in [24, 48, 96, 192, 336, 720]:
                self.assertNotIn(O, param.shape, f"Parameter {name} shape {param.shape} depends on horizon {O}")

    def test_zero_shot_multi_horizon_invariance(self):
        """Verify same model instance forecasts all horizons with unchanged parameter checksum."""
        checksum_initial = compute_parameter_checksum(self.model)
        x = torch.randn(self.B, self.L, self.D)
        xm = torch.randn(self.B, self.L, 4)

        for O in self.eval_horizons:
            ym = torch.randn(self.B, O, 4)
            with torch.no_grad():
                y_pred, A = self.model(x, xm, ym)
            
            # Check shapes
            self.assertEqual(y_pred.shape, (self.B, O, self.D))
            self.assertEqual(A.shape, (self.B, 128, O, self.L))
            
            # Check attention normalization
            sum_A = A.sum(dim=-1)
            self.assertTrue(torch.allclose(sum_A, torch.ones_like(sum_A), atol=1e-5))
            
            # Check parameter checksum remains strictly immutable
            checksum_current = compute_parameter_checksum(self.model)
            self.assertEqual(checksum_initial, checksum_current)

if __name__ == '__main__':
    unittest.main()
