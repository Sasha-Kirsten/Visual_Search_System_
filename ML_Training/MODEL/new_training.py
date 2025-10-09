# # hparam_search.py
# import torch
# import torch.nn as nn
# from torchvision import models
# from lightly.loss import NTXentLoss
# from lightly.models.modules import SimCLRProjectionHead
# from lightly.transforms import SimCLRTransform
# from lightly.data import LightlyDataset
# from torch.utils.data import DataLoader
# import os
# import json
# from datetime import datetime
# import itertools  # For generating parameter grids

# # --- 1. MODEL DEFINITION ---
# class SimCLR(nn.Module):
#     def __init__(self, backbone, out_dim=128):
#         super().__init__()
#         self.backbone = backbone
#         self.projection_head = SimCLRProjectionHead(512, 512, out_dim)

#     def forward(self, x):
#         x = self.backbone(x).flatten(start_dim=1)
#         z = self.projection_head(x)
#         return z

# # --- 2. HYPERPARAMETER CONFIGURATION ---
# # Define a LARGE set of hyperparameter combinations to try
# hyperparameter_sets = []

# # Define the ranges for each parameter we want to test
# search_space = {
#     'batch_size': [256, 512],         # Limited by GPU memory
#     'lr': [0.01, 0.03, 0.06, 0.1],    # Learning rate for SGD
#     'temperature': [0.07, 0.1, 0.5],  # NT-Xent temperature
#     'optimizer': ['SGD', 'AdamW'],
#     'weight_decay': [1e-4, 1e-5],     # Regularization strength
#     'out_dim': [64, 128, 256],        # Projection head output size
#     'momentum': [0.9],                # For SGD
#     'lr_adam': [1e-4, 5e-4, 1e-3]     # Learning rate for Adam (separate range)
# }

# # Generate combinations: for SGD and AdamW separately due to different LR ranges
# for params in itertools.product(search_space['batch_size'], 
#                                 search_space['lr'], 
#                                 search_space['temperature'], 
#                                 search_space['weight_decay'], 
#                                 search_space['out_dim'], 
#                                 search_space['momentum']):
#     batch_size, lr, temp, wd, out_dim, momentum = params
#     hyperparameter_sets.append({
#         'name': f'sgd_bs{batch_size}_lr{lr}_temp{temp}_wd{wd}_dim{out_dim}',
#         'batch_size': batch_size,
#         'lr': lr,
#         'temperature': temp,
#         'optimizer': 'SGD',
#         'momentum': momentum,
#         'weight_decay': wd,
#         'out_dim': out_dim,
#     })

# for params in itertools.product(search_space['batch_size'], 
#                                 search_space['lr_adam'], 
#                                 search_space['temperature'], 
#                                 search_space['weight_decay'], 
#                                 search_space['out_dim']):
#     batch_size, lr, temp, wd, out_dim = params
#     hyperparameter_sets.append({
#         'name': f'adam_bs{batch_size}_lr{lr}_temp{temp}_wd{wd}_dim{out_dim}',
#         'batch_size': batch_size,
#         'lr': lr,
#         'temperature': temp,
#         'optimizer': 'AdamW',
#         'weight_decay': wd,
#         'out_dim': out_dim,
#         'momentum': 0.9,  # Not used for Adam, but included for consistency
#     })

# print(f"Generated {len(hyperparameter_sets)} hyperparameter combinations to test!")

# # --- 3. TRAINING FUNCTION ---
# def train_one_config(config, run_name, base_results_dir="hparam_search_results"):
#     """Train model with one specific hyperparameter configuration"""
    
#     # Create a unique directory for this run
#     run_dir = os.path.join(base_results_dir, run_name)
#     os.makedirs(run_dir, exist_ok=True)
#     print(f"\n=== Starting Run: {run_name} ===")
#     print(f"Parameters: {json.dumps(config, indent=2)}")
    
#     # Save the config for this run
#     with open(os.path.join(run_dir, 'config.json'), 'w') as f:
#         json.dump(config, f, indent=2)
    
#     # Setup device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Data loading
#     transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
#     train_dataset = LightlyDataset(r"C:\Users\Besitzer\Desktop\Image_Dataset_Split\train", transform=transform)
#     val_dataset = LightlyDataset(r"C:\Users\Besitzer\Desktop\Image_Dataset_Split\val", transform=transform)

#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         drop_last=True,
#         num_workers=4,  # Reduced for stability during hyperparameter search
#         persistent_workers=False
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=4
#     )
    
#     # Model
#     resnet18 = models.resnet18()
#     backbone = nn.Sequential(*list(resnet18.children())[:-1])
#     model = SimCLR(backbone, out_dim=config['out_dim']).to(device)
    
#     # Loss function
#     criterion = NTXentLoss(temperature=config['temperature'])
    
#     # Optimizer
#     if config['optimizer'].lower() == 'sgd':
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=config['lr'],
#             momentum=config['momentum'],
#             weight_decay=config['weight_decay']
#         )
#     elif config['optimizer'].lower() == 'adamw':
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=config['lr'],
#             weight_decay=config['weight_decay']
#         )
    
#     # Training setup
#     num_epochs = 25  # Reduced for hyperparameter search
#     best_val_loss = float('inf')
#     history = {'train_loss': [], 'val_loss': []}
    
#     # Training loop
#     for epoch in range(num_epochs):
#         # Training
#         model.train()
#         total_train_loss = 0
#         for batch in train_dataloader:
#             x0, x1 = batch[0]
#             x0, x1 = x0.to(device), x1.to(device)
            
#             optimizer.zero_grad()
#             z0 = model(x0)
#             z1 = model(x1)
#             loss = criterion(z0, z1)
#             loss.backward()
#             optimizer.step()
            
#             total_train_loss += loss.detach().item()
        
#         avg_train_loss = total_train_loss / len(train_dataloader)
#         history['train_loss'].append(avg_train_loss)
        
#         # Validation
#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for batch in val_dataloader:
#                 x0, x1 = batch[0]
#                 x0, x1 = x0.to(device), x1.to(device)
#                 z0 = model(x0)
#                 z1 = model(x1)
#                 loss = criterion(z0, z1)
#                 total_val_loss += loss.detach().item()
        
#         avg_val_loss = total_val_loss / len(val_dataloader)
#         history['val_loss'].append(avg_val_loss)
        
#         print(f"Epoch {epoch:02d}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
#         # Save best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'config': config,
#                 'epoch': epoch,
#                 'loss': best_val_loss,
#             }, os.path.join(run_dir, 'best_model.pth'))
    
#     # Save training history
#     torch.save(history, os.path.join(run_dir, 'training_history.pth'))
    
#     print(f"=== Finished Run: {run_name}. Best Val Loss: {best_val_loss:.4f} ===\n")
#     return best_val_loss, history

# # --- 4. MAIN EXECUTION ---
# if __name__ == '__main__':
#     # Create main results directory with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = f"hparam_search_results_{timestamp}"
#     os.makedirs(results_dir, exist_ok=True)
    
#     # Save the complete search space for reference
#     with open(os.path.join(results_dir, 'search_space.json'), 'w') as f:
#         json.dump(hyperparameter_sets, f, indent=2)
    
#     results = {}
    
#     # Run training for each configuration
#     for i, config in enumerate(hyperparameter_sets):
#         try:
#             best_val_loss, history = train_one_config(config, f"run_{i:03d}", results_dir)
            
#             results[f"run_{i:03d}"] = {
#                 'best_val_loss': best_val_loss,
#                 'config': config,
#                 'name': config['name']
#             }
            
#             # Save interim results after each run
#             with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
#                 json.dump(results, f, indent=2)
                
#         except Exception as e:
#             print(f"!!! Run {i} failed with error: {e}")
#             results[f"run_{i:03d}"] = {
#                 'error': str(e),
#                 'config': config,
#                 'name': config['name']
#             }
    
#     # Print final summary
#     print("\n" + "="*50)
#     print("HYPERPARAMETER SEARCH COMPLETE")
#     print("="*50)
    
#     # Filter out failed runs and sort by best validation loss
#     successful_runs = {k: v for k, v in results.items() if 'best_val_loss' in v}
#     sorted_results = sorted(successful_runs.items(), key=lambda x: x[1]['best_val_loss'])
    
#     print("\nTOP 10 CONFIGURATIONS:")
#     for i, (run_name, result) in enumerate(sorted_results[:10]):
#         print(f"{i+1:2d}. {result['name']}: {result['best_val_loss']:.4f}")
    
#     print(f"\nComplete results saved to: {results_dir}")





import torch
import torch.nn as nn
from torchvision import models
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform
from lightly.data import LightlyDataset
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
import optuna  # Import optuna
from optuna.trial import TrialState

from torch.cuda import amp

# Inside your training function, after device setup
scaler  = torch.amp.GradScaler('cuda')

# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"GPU device: {torch.cuda.get_device_name(0)}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
# else:
#     print("CUDA is not available. Falling back to CPU.")

# --- 1. MODEL DEFINITION ---
class SimCLR(nn.Module):
    def __init__(self, backbone, out_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, out_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

# --- 2. TRAINING FUNCTION (Modified for Optuna) ---
def train_one_config(trial, base_results_dir="hparam_search_results_optuna"):
    """Train model with hyperparameters suggested by Optuna"""
    
    # 1. Let Optuna suggest hyperparameters
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
        'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'AdamW']),
        'temperature': trial.suggest_float('temperature', 0.05, 1.0),  # Wider range
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),  # Log scale
        'out_dim': trial.suggest_categorical('out_dim', [64, 128, 256]),
    }
    
    # Conditional parameters based on optimizer choice
    if config['optimizer'] == 'SGD':
        config['lr'] = trial.suggest_float('lr_sgd', 0.01, 0.2, log=True)  # Log scale for LR
        config['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)  # Range for momentum
    else:  # AdamW
        config['lr'] = trial.suggest_float('lr_adam', 1e-5, 1e-2, log=True)  # Different range for Adam
    
    run_name = f"trial_{trial.number:03d}"
    
    # Create a unique directory for this run
    run_dir = os.path.join(base_results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== Starting Trial {trial.number}: {run_name} ===")
    print(f"Parameters: {json.dumps(config, indent=2)}")
    
    # Save the config for this run
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data loading
    transform = SimCLRTransform(input_size=112, gaussian_blur=0.2)
    train_dataset = LightlyDataset(r"C:\Users\Besitzer\Desktop\Image_Dataset_Split\train", transform=transform)
    val_dataset = LightlyDataset(r"C:\Users\Besitzer\Desktop\Image_Dataset_Split\val", transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=8,  # Increase workers
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Model
    resnet18 = models.resnet18()
    backbone = nn.Sequential(*list(resnet18.children())[:-1])
    model = SimCLR(backbone, out_dim=config['out_dim']).to(device)
    
    # Loss function
    criterion = NTXentLoss(temperature=config['temperature'])
    
    # Optimizer
    if config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    
    # Training setup - reduced epochs for faster search
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            
            optimizer.zero_grad()
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.detach().item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                x0, x1 = batch[0]
                x0, x1 = x0.to(device), x1.to(device)
                z0 = model(x0)
                z1 = model(x1)
                loss = criterion(z0, z1)
                total_val_loss += loss.detach().item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch:02d}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Report intermediate value to Optuna (optional but helpful)
        trial.report(avg_val_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'loss': best_val_loss,
            }, os.path.join(run_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"=== Finished Trial {trial.number}. Best Val Loss: {best_val_loss:.4f} ===\n")
    return best_val_loss

# --- 3. OPTUNA OPTIMIZATION FUNCTION ---
def objective(trial):
    """Objective function for Optuna to minimize"""
    try:
        best_val_loss = train_one_config(trial)
        return best_val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a large value to indicate failure
        return float('inf')

# --- 4. MAIN EXECUTION WITH OPTUNA ---
if __name__ == '__main__':
    # Create main results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hparam_search_results_optuna_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,      # Can prune after 1 epoch
    max_resource=20,     # Maximum epochs
    reduction_factor=3   # How aggressively to prune
)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',  # We want to minimize validation loss
        study_name='simclr_hparam_search',
        storage=f"sqlite:///{os.path.join(results_dir, 'optuna_study.db')}",  # Save progress to DB
        load_if_exists=False,
        pruner=pruner,  # Use the more aggressive pruner  # Prune bad trials early
    )
    
    # Run optimization
    print("Starting Bayesian hyperparameter optimization with Optuna...")
    print(f"Study name: {study.study_name}")
    print(f"Results will be saved to: {results_dir}")
    
    # Run for 50 trials (much fewer than 255!)
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*60)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*60)
    
    # Get completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    # Best trial
    trial = study.best_trial
    print(f"\nBest trial (#{trial.number}):")
    print(f"  Value (Best Validation Loss): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save full study results
    study.trials_dataframe().to_csv(os.path.join(results_dir, 'optuna_study_results.csv'))
    
    # Visualize results (requires plotly)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(results_dir, 'optimization_history.html'))
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(results_dir, 'param_importances.html'))
        
        print("Visualizations saved to HTML files.")
    except ImportError:
        print("Plotly not installed. Skipping visualizations.")
    
    print(f"\nComplete results saved to: {results_dir}")