"""
Example script for inference using the trained 2D UNet.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from models.unet_2d.predict import UNet2DPredictor
from utils.visualization import plot_2d_data, compare_predictions


def main():
    print("=" * 60)
    print("2D UNet - Inference Example")
    print("=" * 60)
    
    # Path to trained model
    model_path = 'checkpoints/unet_2d/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using train_unet_2d.py")
        return
    
    # Load predictor
    print(f"\nLoading model from {model_path}...")
    predictor = UNet2DPredictor(model_path, device='cpu')
    
    # Generate sample 2D data
    print("\nGenerating sample 2D SOLPS-like data...")
    height, width = 256, 256
    
    # Create synthetic 2D field
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Sample field (e.g., plasma density distribution)
    sample_field = 1.8 * np.exp(-r**2 / 0.5**2)
    sample_field *= (1 + 0.2 * np.sin(3 * theta))
    sample_field += np.random.normal(0, 0.05, (height, width))
    
    # Make prediction
    print("Making prediction...")
    prediction = predictor.predict(sample_field)
    
    print(f"\nInput shape: {sample_field.shape}")
    print(f"Prediction shape: {prediction.shape}")
    
    # Visualize results
    print("\nVisualizing results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    vmin = min(sample_field.min(), prediction.min())
    vmax = max(sample_field.max(), prediction.max())
    
    im1 = axes[0].imshow(sample_field, cmap='viridis', vmin=vmin, vmax=vmax, 
                         aspect='auto', interpolation='bilinear')
    axes[0].set_title('Input 2D Field', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(prediction, cmap='viridis', vmin=vmin, vmax=vmax,
                         aspect='auto', interpolation='bilinear')
    axes[1].set_title('Model Prediction', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('prediction_unet_2d.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'prediction_unet_2d.png'")
    plt.show()
    
    # Show statistics
    print("\nPrediction statistics:")
    print(f"  Min value: {prediction.min():.4f}")
    print(f"  Max value: {prediction.max():.4f}")
    print(f"  Mean value: {prediction.mean():.4f}")
    print(f"  Std value: {prediction.std():.4f}")
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
