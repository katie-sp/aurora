import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import pickle

from config import *
from oracles import ORACLE_REGISTRY

def extract_dataset_arrays(dataset, device):
    """
    Extract X and y arrays from any Dataset, handling both:
    - Datasets with pre-computed embeddings (ESM_Dataset)
    - Datasets returning raw sequences (Raw_Dataset)
    """
    print(f"Extracting data from dataset with {len(dataset)} samples...")
    
    # Check if dataset has pre-computed embeddings
    if hasattr(dataset, 'embeddings') and dataset.embeddings is not None:
        print("  Using pre-computed embeddings")
        X = dataset.embeddings.cpu().numpy()
        y = dataset.scores.cpu().numpy()
    else:
        print("  Extracting data on-the-fly from __getitem__")
        X_list, y_list = [], []
        for i in range(len(dataset)):
            x, y_val = dataset[i]
            
            # Handle different x types
            if torch.is_tensor(x):
                X_list.append(x.cpu().numpy())
            elif isinstance(x, str):
                # Raw sequence - needs embedding by Oracle
                # For now, convert to placeholder; Oracle will handle encoding
                X_list.append(x)
            else:
                X_list.append(np.array(x))
            
            # Handle y
            if torch.is_tensor(y_val):
                y_list.append(y_val.item())
            else:
                y_list.append(float(y_val))
        
        # Check if X contains sequences (strings)
        if isinstance(X_list[0], str):
            print("  WARNING: Dataset returns raw sequences. Oracle must handle encoding.")
            # Return as-is; will need special handling in training loop
            X = X_list
        else:
            X = np.stack(X_list)
        
        y = np.array(y_list)
    
    print(f"  Data extracted: X type={type(X)}, y shape={y.shape}")
    return X, y


def train(n_epochs=30, n_splits=5, batch_size=64, lr=1e-3):
    Oracle = ORACLE_REGISTRY[ORACLE_NAME][0]
    Dataset = ORACLE_REGISTRY[ORACLE_NAME][1]()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Extract data from dataset
    X, y = extract_dataset_arrays(Dataset, device)
    
    # Handle string sequences (for Raw_Dataset)
    sequences_mode = isinstance(X, list) and isinstance(X[0], str)
    
    if sequences_mode:
        print("\n⚠️  Raw sequence mode detected. Oracle must have encode() method.")
        # Keep X as list of strings, will encode per-fold
    else:
        print(f"Embedding mode: X shape = {X.shape}")
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    best_model = None
    best_score = -np.inf
    
    # Get indices (works for both arrays and lists)
    indices = np.arange(len(y))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        # Initialize fresh model for this fold
        model = Oracle().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare data for this fold
        if sequences_mode:
            # Raw sequences: need to encode them
            train_seqs = [X[i] for i in train_idx]
            test_seqs = [X[i] for i in test_idx]
            
            # Check if Oracle has encode method
            if not hasattr(model, 'encode'):
                raise AttributeError(
                    f"Oracle {ORACLE_NAME} must have encode(sequences) method "
                    "to work with Raw_Dataset"
                )
            
            print("  Encoding train sequences...")
            train_X_encoded = model.encode(train_seqs)  # Should return tensor
            print("  Encoding test sequences...")
            test_X_encoded = model.encode(test_seqs)
            
            train_X = train_X_encoded
            test_X = test_X_encoded
        else:
            # Pre-computed embeddings
            train_X = torch.tensor(X[train_idx], dtype=torch.float32)
            test_X = torch.tensor(X[test_idx], dtype=torch.float32)
        
        # Targets
        train_y = torch.tensor(y[train_idx], dtype=torch.float32).to(device)
        test_y = torch.tensor(y[test_idx], dtype=torch.float32).to(device)
        
        # Move to device
        train_X = train_X.to(device)
        test_X = test_X.to(device)
        
        # Create dataloaders
        train_dataset_fold = TensorDataset(train_X, train_y)
        test_dataset_fold = TensorDataset(test_X, test_y)
        
        train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset_fold, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            
            epoch_loss /= len(train_loader.dataset)
            
            # Print every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {epoch_loss:.4f}")
        
        # Evaluation
        model.eval()
        
        # Get predictions
        def get_predictions(loader):
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    preds = model(batch_x).squeeze()
                    all_preds.append(preds.cpu())
                    all_targets.append(batch_y.cpu())
            return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()
        
        train_preds, train_targets = get_predictions(train_loader)
        test_preds, test_targets = get_predictions(test_loader)
        
        # Compute metrics
        test_mse = mean_squared_error(test_targets, test_preds)
        test_r2 = r2_score(test_targets, test_preds)
        train_spearman, _ = spearmanr(train_targets, train_preds)
        test_spearman, _ = spearmanr(test_targets, test_preds)
        
        print(f"\n  Fold {fold+1} Results:")
        print(f"    Test MSE:       {test_mse:.4f}")
        print(f"    Test R²:        {test_r2:.4f}")
        print(f"    Train Spearman: {train_spearman:.4f}")
        print(f"    Test Spearman:  {test_spearman:.4f}")
        
        fold_metrics.append({
            "fold": fold + 1,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "train_spearman": train_spearman,
            "test_spearman": test_spearman
        })
        
        # Track best model based on test Spearman
        if test_spearman > best_score:
            best_score = test_spearman
            best_model = model.state_dict().copy()
            print(f"    ✓ New best model (Test Spearman: {test_spearman:.4f})")
    
    # Compute summary statistics
    metrics = {
        "fold_metrics": fold_metrics,
        "mean_test_mse": np.mean([m["test_mse"] for m in fold_metrics]),
        "std_test_mse": np.std([m["test_mse"] for m in fold_metrics]),
        "mean_test_r2": np.mean([m["test_r2"] for m in fold_metrics]),
        "std_test_r2": np.std([m["test_r2"] for m in fold_metrics]),
        "mean_train_spearman": np.mean([m["train_spearman"] for m in fold_metrics]),
        "std_train_spearman": np.std([m["train_spearman"] for m in fold_metrics]),
        "mean_test_spearman": np.mean([m["test_spearman"] for m in fold_metrics]),
        "std_test_spearman": np.std([m["test_spearman"] for m in fold_metrics])
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Test MSE:       {metrics['mean_test_mse']:.4f} ± {metrics['std_test_mse']:.4f}")
    print(f"Test R²:        {metrics['mean_test_r2']:.4f} ± {metrics['std_test_r2']:.4f}")
    print(f"Train Spearman: {metrics['mean_train_spearman']:.4f} ± {metrics['std_train_spearman']:.4f}")
    print(f"Test Spearman:  {metrics['mean_test_spearman']:.4f} ± {metrics['std_test_spearman']:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    with open(f"{ORACLE_DIR}/oracle_metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Metrics saved to {ORACLE_DIR}/oracle_metrics.pkl")
    
    # Save best model
    torch.save(best_model, f'{ORACLE_DIR}/oracle.state_dict')
    print(f"✓ Best model saved to {ORACLE_DIR}/oracle.state_dict")
    print(f"  (Best Test Spearman: {best_score:.4f})\n")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Oracle Training: {ORACLE_NAME}")
    print(f"Output directory: {ORACLE_DIR}/")
    print(f"{'='*60}\n")
    train()