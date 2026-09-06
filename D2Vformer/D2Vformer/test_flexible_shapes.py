#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flexible Forecasting Shape Test
Tests whether D2Vformer_simple_flexible can handle variable pred_len at inference.

This is a SMOKE TEST — no real data, no trained weights.
Just verifies tensor shapes flow correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse

# Import the flexible variant
from model.D2Vformer_simple_flexible import D2Vformer_simple_flexible

def test_shape(pred_len, seq_len=96, label_len=48, d_feature=7, batch_size=2):
    """
    Test if model can process inputs with given pred_len and produce correct output shape.
    """
    print(f"\n{'='*60}")
    print(f"Testing pred_len={pred_len}")
    print(f"{'='*60}")

    # Create dummy config
    args = argparse.Namespace(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=96,  # Training pred_len (default)
        d_feature=d_feature,
        c_out=d_feature,
        d_model=512,
        d_ff=1024,
        T2V_outmodel=36,
        mark_index=[0, 1, 2, 3],
        d_mark=27,
        dropout=0.1,
        patch_len=16,
        stride=8,
        n_heads=3,
        save_path='./test_output',
        output_path='./test_output'
    )

    # Create model (untrained, just for shape testing)
    model = D2Vformer_simple_flexible(args)
    model.eval()

    # Create dummy inputs
    x_enc = torch.randn(batch_size, seq_len, d_feature)
    x_mark_enc = torch.randn(batch_size, seq_len, 27)

    # CRITICAL: x_dec and x_mark_dec must have shape [B, label_len + pred_len, ...]
    x_dec = torch.randn(batch_size, label_len + pred_len, d_feature)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 27)

    print(f"Input shapes:")
    print(f"  x_enc:      {tuple(x_enc.shape)}")
    print(f"  x_mark_enc: {tuple(x_mark_enc.shape)}")
    print(f"  x_dec:      {tuple(x_dec.shape)}")
    print(f"  x_mark_dec: {tuple(x_mark_dec.shape)}")

    try:
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mode='test', pred_len=pred_len)

        expected_shape = (batch_size, pred_len, d_feature)
        actual_shape = tuple(output.shape)

        print(f"\nOutput shape: {actual_shape}")
        print(f"Expected:     {expected_shape}")

        if actual_shape == expected_shape:
            print(f"[PASS]: Shape matches!")
            return True
        else:
            print(f"[FAIL]: Shape mismatch!")
            return False

    except Exception as e:
        print(f"[FAIL]: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("D2Vformer Flexible Forecasting — Shape Test")
    print("="*60)
    print("\nThis test verifies that D2Vformer_simple_flexible can")
    print("handle variable pred_len values at inference time.")
    print("\nNo trained weights loaded — this is a pure shape test.")

    # Test multiple prediction lengths
    test_cases = [48, 72, 96, 192, 336]
    results = {}

    for pred_len in test_cases:
        success = test_shape(pred_len)
        results[pred_len] = success

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for pred_len, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"pred_len={pred_len:3d}h: {status}")

    all_pass = all(results.values())

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
        print("\nNext step: Test with real trained checkpoint")
    else:
        print("SOME TESTS FAILED")
        print("\nDo NOT proceed to checkpoint testing until shapes work.")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
