"""
Example script for training the 2D UNet on SOLPS data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.unet_2d.model import UNet2D
from models.unet_2d.train import UNet2DTrainer
from utils.visualization import plot_training_history


def generate_synthetic_2d_data(n_samples=100, channels=1, height=256, width=256):
    """
    Generate synthetic 2D SOLPS-like data for demonstration.
    
    In practice, replace this with real SOLPS data loading.
    """
    X_train = []
    y_train = []
    
    for _ in range(n_samples):
        # Create synthetic 2D plasma distribution
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create a radial pattern typical in plasma physics
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Input field (e.g., electron density)
        amplitude = np.random.uniform(0.5, 2.0)
        width_param = np.random.uniform(0.3, 0.7)
        field_in = amplitude * np.exp(-r**2 / width_param**2)
        
        # Add some angular variation
        field_in *= (1 + 0.2 * np.sin(3 * theta))
        field_in += np.random.normal(0, 0.05, (height, width))
        
        # Output field (e.g., temperature or derived quantity)
        field_out = amplitude * 1.3 * np.exp(-r**2 / width_param**2)
        field_out *= (1 + 0.15 * np.sin(3 * theta))
        field_out += np.random.normal(0, 0.03, (height, width))
        
        X_train.append(field_in.reshape(channels, height, width))
        y_train.append(field_out.reshape(channels, height, width))
    
    return np.array(X_train), np.array(y_train)


def main():
    # Load configuration
    config_path = 'configs/unet_2d_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Config file not found at {config_path}, using defaults")
        config = {
            "model": {"n_channels": 1, "n_classes": 1, "bilinear": True, 
                     "base_channels": 64},
            "training": {"batch_size": 4, "learning_rate": 0.0001, 
                        "epochs": 50, "train_split": 0.8},
            "data": {"image_size": [256, 256]},
            "checkpoint": {"checkpoint_dir": "checkpoints/unet_2d"}
        }
    
    print("=" * 60)
    print("Training 2D UNet for SOLPS Data")
    print("=" * 60)
    
    # Generate or load data
    print("\n1. Loading/Generating data...")
    n_samples = 200
    height, width = config['data']['image_size']
    X_train, y_train = generate_synthetic_2d_data(
        n_samples=n_samples,
        channels=config['model']['n_channels'],
        height=height,
        width=width
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
    model = UNet2D(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes'],
        bilinear=config['model']['bilinear'],
        base_channels=config['model']['base_channels']
    )
    print(f"   Model parameters: {model.get_num_parameters():,}")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    trainer = UNet2DTrainer(
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
