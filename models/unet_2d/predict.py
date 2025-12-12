"""
Inference/prediction script for 2D UNet model.
"""

import torch
import numpy as np
from .model import UNet2D


class UNet2DPredictor:
    """
    Predictor class for UNet2D model inference.
    
    Args:
        model_path (str): Path to saved model checkpoint
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path (str): Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model architecture info from state dict
        state_dict = checkpoint['model_state_dict']
        n_channels = state_dict['inc.double_conv.0.weight'].shape[1]
        n_classes = state_dict['outc.conv.weight'].shape[0]
        
        # Initialize model
        self.model = UNet2D(n_channels=n_channels, n_classes=n_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Input channels: {n_channels}, Output channels: {n_classes}")
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x (np.ndarray or torch.Tensor): Input data of shape (n_samples, C, H, W),
                                            (C, H, W), or (H, W) for single channel
        
        Returns:
            np.ndarray: Predictions of same shape as input
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Add channel dimension if needed
        original_shape = x.shape
        if x.dim() == 2:  # (H, W)
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            single_sample = True
            single_channel = True
        elif x.dim() == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dim
            single_sample = True
            single_channel = False
        else:  # (N, C, H, W)
            single_sample = False
            single_channel = False
        
        # Move to device
        x = x.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(x)
        
        # Convert to numpy
        output = output.cpu().numpy()
        
        # Restore original shape
        if single_sample and single_channel:
            output = output.squeeze(0).squeeze(0)
        elif single_sample:
            output = output.squeeze(0)
        
        return output
    
    def predict_batch(self, x_list):
        """
        Make predictions on a list of input samples.
        
        Args:
            x_list (list): List of input samples
        
        Returns:
            list: List of predictions
        """
        return [self.predict(x) for x in x_list]
    
    def predict_with_tiling(self, x, tile_size=256, overlap=32):
        """
        Make predictions on large images using tiling strategy.
        
        This is useful for images larger than what fits in memory or
        for maintaining consistent predictions across the image.
        
        Args:
            x (np.ndarray or torch.Tensor): Input image
            tile_size (int): Size of each tile
            overlap (int): Overlap between tiles
        
        Returns:
            np.ndarray: Full prediction
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure proper dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add channel dim
        
        C, H, W = x.shape
        stride = tile_size - overlap
        
        # Calculate number of tiles needed
        n_tiles_h = (H - overlap) // stride + 1
        n_tiles_w = (W - overlap) // stride + 1
        
        # Prepare output
        output = torch.zeros((C, H, W))
        counts = torch.zeros((H, W))
        
        # Process each tile
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries
                h_start = i * stride
                h_end = min(h_start + tile_size, H)
                w_start = j * stride
                w_end = min(w_start + tile_size, W)
                
                # Extract tile
                tile = x[:, h_start:h_end, w_start:w_end]
                
                # Pad if necessary
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    pad_h = tile_size - tile.shape[1]
                    pad_w = tile_size - tile.shape[2]
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))
                
                # Predict
                tile_pred = self.predict(tile)
                tile_pred = torch.from_numpy(tile_pred) if isinstance(tile_pred, np.ndarray) else tile_pred
                
                # Remove padding
                tile_pred = tile_pred[:, :h_end-h_start, :w_end-w_start]
                
                # Add to output
                output[:, h_start:h_end, w_start:w_end] += tile_pred
                counts[h_start:h_end, w_start:w_end] += 1
        
        # Average overlapping predictions
        output = output / counts.unsqueeze(0)
        
        return output.numpy()


if __name__ == "__main__":
    # Example usage
    print("Predictor example usage:")
    print("Note: This requires a trained model checkpoint.")
    print("\nTo use:")
    print("1. Train a model using train.py")
    print("2. Load the checkpoint and make predictions:")
    print()
    print("predictor = UNet2DPredictor('checkpoints/best_model.pth')")
    print("x = np.random.randn(1, 256, 256)  # Single 2D image")
    print("prediction = predictor.predict(x)")
    print("print(f'Prediction shape: {prediction.shape}')")
