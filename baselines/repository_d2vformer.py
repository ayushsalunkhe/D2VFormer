import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.D2Vformer import D2Vformer

def create_repository_d2vformer(c_in: int, seq_len: int, pred_len: int, mask_spectrum, d_model: int = 128, patch_len: int = 4, stride: int = 2, d_mark: int = 4, T2V_outmodel: int = 16, dropout: float = 0.05):
    """
    Factory function for the official repository D2Vformer (with line 77 bug fixed).
    Because it contains horizon-dependent components (project, trend_linear_decoder, fusion.linear_out),
    a separate model must be instantiated and retrained for each target horizon pred_len.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', default=seq_len, type=int)
    parser.add_argument('--label_len', default=seq_len // 2, type=int)
    parser.add_argument('--pred_len', default=pred_len, type=int)
    parser.add_argument('--d_feature', default=c_in, type=int)
    parser.add_argument('--d_model', default=d_model, type=int)
    parser.add_argument('--T2V_outmodel', default=T2V_outmodel, type=int)
    parser.add_argument('--d_mark', default=d_mark, type=int)
    parser.add_argument('--patch_len', default=patch_len, type=int)
    parser.add_argument('--stride', default=stride, type=int)
    parser.add_argument('--dropout', default=dropout, type=float)
    parser.add_argument('--mask_spectrum', default=mask_spectrum)
    args = parser.parse_args([])
    
    return D2Vformer(args)
