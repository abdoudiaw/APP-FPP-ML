"""
1D Neural Network model for processing radial data from APP-FPP project.
This model is designed to learn patterns in 1D radial profiles.
"""

import torch
import torch.nn as nn


class RadialNN(nn.Module):
    """
    Neural Network for 1D radial data processing.
    
    This model uses fully connected layers with batch normalization and dropout
    to learn mappings from input radial profiles to target outputs.
    
    Args:
        input_size (int): Size of input radial data points
        hidden_sizes (list): List of hidden layer sizes
        output_size (int): Size of output
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, input_size=64, hidden_sizes=None, 
                 output_size=64, dropout_rate=0.2):
        super(RadialNN, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 256, 128]
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)
    
    def get_num_parameters(self):
        """
        Get the total number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = RadialNN(input_size=64, hidden_sizes=[128, 256, 128], output_size=64)
    print(f"Model architecture:\n{model}")
    print(f"\nTotal trainable parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 64)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
