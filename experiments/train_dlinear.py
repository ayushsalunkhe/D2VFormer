import torch
import torch.nn as nn
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from baselines.dlinear import DLinear
from utils.data import get_data_loaders
from utils.reproducibility import set_seed, compute_parameter_checksum

def train_dlinear(dataset_name: str = 'ETTh1',
                  seq_len: int = 96,
                  pred_len: int = 48,
                  moving_avg_kernel: int = 25,
                  individual: bool = False,
                  lr: float = 1e-3,
                  epochs: int = 10,
                  patience: int = 3,
                  batch_size: int = 64,
                  seed: int = 42,
                  device: str = 'cpu',
                  checkpoint_dir: str = 'results/checkpoints'):
    """
    Trains DLinear for a specific target prediction horizon pred_len.
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
    
    model = DLinear(
        seq_len=seq_len,
        pred_len=pred_len,
        c_in=c_in,
        moving_avg_kernel=moving_avg_kernel,
        individual=individual
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoint_dir, f"dlinear_{dataset_name}_O{pred_len}_seed{seed}.pt")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
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
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
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
