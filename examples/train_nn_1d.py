"""
Example script for training the 1D Neural Network on radial data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.nn_1d.model import RadialNN
from models.nn_1d.train import RadialNNTrainer
from utils.visualization import plot_training_history


def generate_synthetic_data(n_samples=1000, input_size=64, output_size=64):
    """
    Generate synthetic radial data for demonstration.
    
    In practice, replace this with real data loading.
    """
    # Generate synthetic radial profiles
    x = np.linspace(0, 1, input_size)
    
    X_train = []
    y_train = []
    
    for _ in range(n_samples):
        # Create synthetic radial profiles with different characteristics
        amplitude = np.random.uniform(0.5, 2.0)
        width = np.random.uniform(0.1, 0.5)
        center = np.random.uniform(0.3, 0.7)
        
        # Input profile (e.g., temperature or density)
        profile_in = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        profile_in += np.random.normal(0, 0.05, input_size)
        
        # Output profile (transformed or predicted quantity)
        profile_out = amplitude * 1.2 * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        profile_out += np.random.normal(0, 0.03, output_size)
        
        X_train.append(profile_in)
        y_train.append(profile_out)
    
    return np.array(X_train), np.array(y_train)


def main():
    # Load configuration
    config_path = 'configs/nn_1d_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Config file not found at {config_path}, using defaults")
        config = {
            "model": {"input_size": 64, "hidden_sizes": [128, 256, 128], 
                     "output_size": 64, "dropout_rate": 0.2},
            "training": {"batch_size": 32, "learning_rate": 0.001, 
                        "epochs": 100, "train_split": 0.8},
            "checkpoint": {"checkpoint_dir": "checkpoints/nn_1d"}
        }
    
    print("=" * 60)
    print("Training 1D Radial Neural Network")
    print("=" * 60)
    
    # Generate or load data
    print("\n1. Loading/Generating data...")
    n_samples = 2000
    X_train, y_train = generate_synthetic_data(
        n_samples=n_samples,
        input_size=config['model']['input_size'],
        output_size=config['model']['output_size']
    )
    print(f"   Generated {n_samples} training samples")
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    
    # Create datasets and data loaders
    print("\n2. Creating data loaders...")
    train_size = int(len(X_train) * config['training']['train_split'])
    val_size = len(X_train) - train_size
    
    indices = np.random.permutation(len(X_train))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train[train_indices]),
        torch.FloatTensor(y_train[train_indices])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_train[val_indices]),
        torch.FloatTensor(y_train[val_indices])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")
    
    # Initialize model
    print("\n3. Initializing model...")
    model = RadialNN(
        input_size=config['model']['input_size'],
        hidden_sizes=config['model']['hidden_sizes'],
        output_size=config['model']['output_size'],
        dropout_rate=config['model']['dropout_rate']
    )
    print(f"   Model parameters: {model.get_num_parameters():,}")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    trainer = RadialNNTrainer(
        model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        checkpoint_dir=config['checkpoint']['checkpoint_dir']
    )
    
    # Train model
    print("\n4. Training model...")
    print("-" * 60)
    trainer.train(
        train_loader,
        val_loader,
        epochs=config['training']['epochs']
    )
    
    # Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(trainer.history)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {config['checkpoint']['checkpoint_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
