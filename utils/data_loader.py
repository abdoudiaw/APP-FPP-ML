"""
Data loading utilities for radial and 2D SOLPS data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os


class RadialDataLoader(Dataset):
    """
    Dataset class for loading 1D radial data.
    
    This class handles loading and preprocessing of radial profile data
    from various file formats (HDF5, NPY, text files).
    
    Args:
        data_path (str): Path to data file or directory
        file_format (str): Format of data files ('hdf5', 'npy', 'txt')
        normalize (bool): Whether to normalize the data
        transform (callable, optional): Optional transform to apply to data
    """
    
    def __init__(self, data_path, file_format='npy', normalize=True, transform=None):
        self.data_path = data_path
        self.file_format = file_format
        self.normalize = normalize
        self.transform = transform
        
        self.data = []
        self.targets = []
        
        self.load_data()
        
        if self.normalize:
            self.normalize_data()
    
    def load_data(self):
        """Load data from file."""
        if self.file_format == 'npy':
            data = np.load(self.data_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                # Assume data is structured as [inputs, targets]
                if len(data.shape) == 3:  # (n_samples, 2, n_points)
                    self.data = data[:, 0, :]
                    self.targets = data[:, 1, :]
                else:  # (n_samples, n_points)
                    self.data = data
                    self.targets = data  # For autoencoder-like tasks
            else:
                self.data = data.item()['inputs']
                self.targets = data.item()['targets']
        
        elif self.file_format == 'hdf5':
            with h5py.File(self.data_path, 'r') as f:
                self.data = f['inputs'][:]
                self.targets = f['targets'][:]
        
        elif self.file_format == 'txt':
            data = np.loadtxt(self.data_path)
            self.data = data
            self.targets = data
        
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        print(f"Loaded {len(self.data)} samples from {self.data_path}")
    
    def normalize_data(self):
        """Normalize data to zero mean and unit variance."""
        self.data_mean = np.mean(self.data, axis=0)
        self.data_std = np.std(self.data, axis=0) + 1e-8
        self.data = (self.data - self.data_mean) / self.data_std
        
        self.target_mean = np.mean(self.targets, axis=0)
        self.target_std = np.std(self.targets, axis=0) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std
    
    def denormalize(self, data, is_target=False):
        """
        Denormalize data back to original scale.
        
        Args:
            data (np.ndarray): Normalized data
            is_target (bool): Whether data is target (True) or input (False)
        
        Returns:
            np.ndarray: Denormalized data
        """
        if not self.normalize:
            return data
        
        if is_target:
            return data * self.target_std + self.target_mean
        else:
            return data * self.data_std + self.data_mean
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.targets[idx])
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class SOLPS2DDataLoader(Dataset):
    """
    Dataset class for loading 2D SOLPS data.
    
    This class handles loading and preprocessing of 2D plasma simulation data
    from SOLPS simulations.
    
    Args:
        data_path (str): Path to data file or directory
        file_format (str): Format of data files ('hdf5', 'npy')
        normalize (bool): Whether to normalize the data
        transform (callable, optional): Optional transform to apply to data
    """
    
    def __init__(self, data_path, file_format='npy', normalize=True, transform=None):
        self.data_path = data_path
        self.file_format = file_format
        self.normalize = normalize
        self.transform = transform
        
        self.data = []
        self.targets = []
        
        self.load_data()
        
        if self.normalize:
            self.normalize_data()
    
    def load_data(self):
        """Load 2D data from file."""
        if self.file_format == 'npy':
            data = np.load(self.data_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                # Assume data shape: (n_samples, 2, channels, H, W) or (n_samples, channels, H, W)
                if len(data.shape) == 5:  # (n_samples, 2, channels, H, W)
                    self.data = data[:, 0, :, :, :]
                    self.targets = data[:, 1, :, :, :]
                else:  # (n_samples, channels, H, W)
                    self.data = data
                    self.targets = data  # For autoencoder-like tasks
            else:
                self.data = data.item()['inputs']
                self.targets = data.item()['targets']
        
        elif self.file_format == 'hdf5':
            with h5py.File(self.data_path, 'r') as f:
                self.data = f['inputs'][:]
                self.targets = f['targets'][:]
        
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        print(f"Loaded {len(self.data)} samples from {self.data_path}")
        print(f"Data shape: {self.data.shape}")
    
    def normalize_data(self):
        """Normalize data to zero mean and unit variance."""
        self.data_mean = np.mean(self.data)
        self.data_std = np.std(self.data) + 1e-8
        self.data = (self.data - self.data_mean) / self.data_std
        
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std
    
    def denormalize(self, data, is_target=False):
        """
        Denormalize data back to original scale.
        
        Args:
            data (np.ndarray): Normalized data
            is_target (bool): Whether data is target (True) or input (False)
        
        Returns:
            np.ndarray: Denormalized data
        """
        if not self.normalize:
            return data
        
        if is_target:
            return data * self.target_std + self.target_mean
        else:
            return data * self.data_std + self.data_mean
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.targets[idx])
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_data_loaders(dataset, batch_size=32, train_split=0.8, shuffle=True):
    """
    Create train and validation data loaders from a dataset.
    
    Args:
        dataset (Dataset): PyTorch dataset
        batch_size (int): Batch size for data loaders
        train_split (float): Fraction of data to use for training
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Data loader utilities")
    print("\nExample usage for 1D radial data:")
    print("dataset = RadialDataLoader('data/radial_data.npy')")
    print("train_loader, val_loader = create_data_loaders(dataset)")
    print("\nExample usage for 2D SOLPS data:")
    print("dataset = SOLPS2DDataLoader('data/solps_2d_data.npy')")
    print("train_loader, val_loader = create_data_loaders(dataset)")
