#!/usr/bin/env python
"""
Quick test script to verify both models work correctly.
"""

import torch
import numpy as np

print("=" * 60)
print("Testing APP-FPP ML Models")
print("=" * 60)

# Test 1D NN Model
print("\n1. Testing 1D Neural Network for Radial Data")
print("-" * 60)
from models.nn_1d.model import RadialNN

model_1d = RadialNN(input_size=64, hidden_sizes=[128, 256, 128], output_size=64)
print(f"✓ Model created successfully")
print(f"  Parameters: {model_1d.get_num_parameters():,}")

# Test forward pass
batch_size = 4
x = torch.randn(batch_size, 64)
y = model_1d(x)
print(f"✓ Forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {y.shape}")

# Test 2D UNet Model
print("\n2. Testing 2D UNet for SOLPS Data")
print("-" * 60)
from models.unet_2d.model import UNet2D

model_2d = UNet2D(n_channels=1, n_classes=1, base_channels=64)
print(f"✓ Model created successfully")
print(f"  Parameters: {model_2d.get_num_parameters():,}")

# Test forward pass
batch_size = 2
height, width = 256, 256
x = torch.randn(batch_size, 1, height, width)
y = model_2d(x)
print(f"✓ Forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {y.shape}")

# Test utilities
print("\n3. Testing Utility Functions")
print("-" * 60)

# Test data loaders (with synthetic data)
print("Creating synthetic data for testing...")
X_train = np.random.randn(100, 64)
y_train = np.random.randn(100, 64)
np.save('/tmp/test_radial_data.npy', np.stack([X_train, y_train], axis=1))

from utils.data_loader import RadialDataLoader, create_data_loaders

print("✓ Testing 1D data loader...")
dataset_1d = RadialDataLoader('/tmp/test_radial_data.npy', normalize=False)
print(f"  Loaded {len(dataset_1d)} samples")

train_loader, val_loader = create_data_loaders(dataset_1d, batch_size=16)
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# Test 2D data loader
X_train_2d = np.random.randn(50, 1, 128, 128)
y_train_2d = np.random.randn(50, 1, 128, 128)
np.save('/tmp/test_solps_2d_data.npy', np.stack([X_train_2d, y_train_2d], axis=1))

from utils.data_loader import SOLPS2DDataLoader

print("✓ Testing 2D data loader...")
dataset_2d = SOLPS2DDataLoader('/tmp/test_solps_2d_data.npy', normalize=False)
print(f"  Loaded {len(dataset_2d)} samples")

# Summary
print("\n" + "=" * 60)
print("All tests passed successfully! ✓")
print("=" * 60)
print("\nYou can now:")
print("  1. Train the 1D NN model: python examples/train_nn_1d.py")
print("  2. Train the 2D UNet model: python examples/train_unet_2d.py")
print("  3. Run inference after training")
print("=" * 60)
