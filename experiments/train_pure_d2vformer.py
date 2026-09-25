import torch
import torch.nn as nn
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.pure_d2vformer import PureD2Vformer
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def train_pure_d2vformer(dataset_name: str = 'ETTh1',
                         seq_len: int = 96,
                         train_horizon: int = 48,
                         d_model: int = 128,
                         d_ff: int = 256,
                         k_freq: int = 16,
                         dropout: float = 0.05,
                         lr: float = 1e-3,
                         epochs: int = 10,
                         patience: int = 3,
                         batch_size: int = 64,
                         seed: int = 42,
                         device: str = 'cpu',
                         checkpoint_dir: str = 'results/checkpoints'):
    """
    Trains PureD2Vformer ONCE strictly at train_horizon = 48.
    The resulting checkpoint is saved and used for zero-shot evaluation across all horizons.
    """
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_loader, val_loader, _, metadata = get_data_loaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=train_horizon,
        batch_size=batch_size
    )
    c_in = metadata['num_variables']
    
    model = PureD2Vformer(
        c_in=c_in,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        k_freq=k_freq,
        dropout=dropout
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoint_dir, f"pured2vformer_{dataset_name}_seed{seed}.pt")
    
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
            out, _ = model(batch_x, batch_xm, batch_ym)
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
                out, _ = model(batch_x, batch_xm, batch_ym)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)
        
        print(f"[PureD2Vformer | {dataset_name} | Seed {seed}] Epoch {epoch:2d}/{epochs:2d} - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint with deterministic parameter checksum
            checksum = compute_parameter_checksum(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'checksum': checksum,
                'param_count': param_count,
                'c_in': c_in,
                'seq_len': seq_len,
                'd_model': d_model,
                'd_ff': d_ff,
                'k_freq': k_freq,
                'dropout': dropout,
                'train_horizon': train_horizon,
                'seed': seed,
                'dataset': dataset_name,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val MSE: {best_val_loss:.4f} at epoch {best_epoch}.")
                break
                
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s. Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path, training_time, param_count

if __name__ == '__main__':
    train_pure_d2vformer(dataset_name='ETTh1', epochs=2, batch_size=64)
