import torch
import torch.nn as nn
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from baselines.repository_d2vformer import create_repository_d2vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def compute_mask_spectrum(train_loader, alpha: float = 0.2, device: str = 'cpu'):
    """Calculates top-k FFT amplitude spectrum indices as required by FourierFilter."""
    amps = 0.0
    for batch_x, _, _, _ in train_loader:
        lookback = batch_x.to(device)
        amps += torch.abs(torch.fft.rfft(lookback, dim=1)).mean(dim=0).mean(dim=1)
    k = max(1, int(amps.shape[0] * alpha))
    return amps.topk(k).indices

def train_repository_d2vformer(dataset_name: str = 'ETTh1',
                              seq_len: int = 96,
                              pred_len: int = 48,
                              d_model: int = 128,
                              d_mark: int = 4,
                              patch_len: int = 4,
                              stride: int = 2,
                              lr: float = 1e-3,
                              epochs: int = 10,
                              patience: int = 3,
                              batch_size: int = 64,
                              seed: int = 42,
                              device: str = 'cpu',
                              checkpoint_dir: str = 'results/checkpoints'):
    """
    Trains the repaired official repository D2Vformer for a specific prediction horizon pred_len.
    """
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_loader, val_loader, _, metadata = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size
    )
    c_in = metadata['num_variables']
    mask_spectrum = compute_mask_spectrum(train_loader, alpha=0.2, device=device)
    
    model = create_repository_d2vformer(
        c_in=c_in,
        seq_len=seq_len,
        pred_len=pred_len,
        mask_spectrum=mask_spectrum,
        d_model=d_model,
        d_mark=d_mark,
        patch_len=patch_len,
        stride=stride
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoint_dir, f"repo_d2vformer_{dataset_name}_O{pred_len}_seed{seed}.pt")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y, batch_xm, batch_ym in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_xm = batch_xm.to(device)
            batch_ym = batch_ym.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x, batch_xm, batch_y, batch_ym)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
        train_loss /= max(1, n_batches)
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_xm, batch_ym in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_xm = batch_xm.to(device)
                batch_ym = batch_ym.to(device)
                out = model(batch_x, batch_xm, batch_y, batch_ym)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            checksum = compute_parameter_checksum(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'checksum': checksum,
                'param_count': param_count,
                'c_in': c_in,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'seed': seed,
                'dataset': dataset_name,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    training_time = time.time() - start_time
    return checkpoint_path, training_time, param_count
