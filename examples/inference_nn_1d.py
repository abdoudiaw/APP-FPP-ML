"""
Example script for inference using the trained 1D Neural Network.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from models.nn_1d.predict import RadialNNPredictor
from utils.visualization import plot_radial_data


def main():
    print("=" * 60)
    print("1D Radial NN - Inference Example")
    print("=" * 60)
    
    # Path to trained model
    model_path = 'checkpoints/nn_1d/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using train_nn_1d.py")
        return
    
    # Load predictor
    print(f"\nLoading model from {model_path}...")
    predictor = RadialNNPredictor(model_path, device='cpu')
    
    # Generate sample input data
    print("\nGenerating sample radial profile...")
    input_size = 64
    x = np.linspace(0, 1, input_size)
    
    # Create a sample radial profile
    sample_profile = 1.5 * np.exp(-((x - 0.5) ** 2) / (2 * 0.3 ** 2))
    sample_profile += np.random.normal(0, 0.05, input_size)
    
    # Make prediction
    print("Making prediction...")
    prediction = predictor.predict(sample_profile)
    
    print(f"\nInput shape: {sample_profile.shape}")
    print(f"Prediction shape: {prediction.shape}")
    
    # Visualize results
    print("\nVisualizing results...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, sample_profile, 'b-', linewidth=2, label='Input Profile')
    plt.xlabel('Radial Position', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Input Radial Profile', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, prediction, 'r-', linewidth=2, label='Predicted Profile')
    plt.xlabel('Radial Position', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Model Prediction', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_nn_1d.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'prediction_nn_1d.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
