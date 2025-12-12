"""
Inference/prediction script for 1D Radial Neural Network.
"""

import torch
import numpy as np
from .model import RadialNN


class RadialNNPredictor:
    """
    Predictor class for RadialNN model inference.
    
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
        input_size = state_dict['network.0.weight'].shape[1]
        output_size = state_dict[list(state_dict.keys())[-1]].shape[0]
        
        # Initialize model
        self.model = RadialNN(input_size=input_size, output_size=output_size)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Input size: {input_size}, Output size: {output_size}")
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x (np.ndarray or torch.Tensor): Input data of shape (n_samples, input_size)
                                            or (input_size,) for single sample
        
        Returns:
            np.ndarray: Predictions of shape (n_samples, output_size) or (output_size,)
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Add batch dimension if single sample
        single_sample = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        
        # Move to device
        x = x.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(x)
        
        # Convert to numpy
        output = output.cpu().numpy()
        
        # Remove batch dimension if single sample
        if single_sample:
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


if __name__ == "__main__":
    # Example usage
    print("Predictor example usage:")
    print("Note: This requires a trained model checkpoint.")
    print("\nTo use:")
    print("1. Train a model using train.py")
    print("2. Load the checkpoint and make predictions:")
    print()
    print("predictor = RadialNNPredictor('checkpoints/best_model.pth')")
    print("x = np.random.randn(64)  # Single radial profile")
    print("prediction = predictor.predict(x)")
    print("print(f'Prediction shape: {prediction.shape}')")
