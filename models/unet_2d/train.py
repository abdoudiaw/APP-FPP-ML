"""
Training script for 2D UNet model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import json

from .model import UNet2D


class UNet2DTrainer:
    """
    Trainer class for UNet2D model.
    
    Args:
        model (UNet2D): The UNet model
        device (str): Device to train on ('cuda' or 'cpu')
        learning_rate (float): Learning rate for optimizer
        checkpoint_dir (str): Directory to save checkpoints
    """
    
    def __init__(self, model, device='cpu', learning_rate=1e-4, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader=None, epochs=100):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            epochs (int): Number of epochs to train
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}", end="")
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                print(f" - Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
            else:
                print()
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, filename):
        """
        Save model checkpoint.
        
        Args:
            filename (str): Checkpoint filename
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """
        Load model checkpoint.
        
        Args:
            filename (str): Checkpoint filename
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded: {path}")


if __name__ == "__main__":
    # Example usage
    print("Creating sample training data...")
    
    # Generate synthetic 2D data for demonstration
    n_samples = 100
    channels = 1
    height, width = 256, 256
    
    X_train = torch.randn(n_samples, channels, height, width)
    y_train = torch.randn(n_samples, channels, height, width)
    
    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Initialize model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = UNet2D(n_channels=channels, n_classes=channels)
    trainer = UNet2DTrainer(model, device=device, learning_rate=1e-4)
    
    print("Starting training...")
    trainer.train(train_loader, epochs=5)
