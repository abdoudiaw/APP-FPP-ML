"""
Visualization utilities for plotting data and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_radial_data(data, labels=None, title='Radial Profile', 
                     xlabel='Radial Position', ylabel='Value', 
                     save_path=None, figsize=(10, 6)):
    """
    Plot 1D radial data profiles.
    
    Args:
        data (np.ndarray or list): Radial data to plot, shape (n_points,) or (n_profiles, n_points)
        labels (list, optional): Labels for each profile
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    if isinstance(data, list):
        for i, profile in enumerate(data):
            label = labels[i] if labels else f'Profile {i+1}'
            plt.plot(profile, label=label, linewidth=2)
    else:
        if data.ndim == 1:
            plt.plot(data, linewidth=2)
        else:
            for i in range(data.shape[0]):
                label = labels[i] if labels else f'Profile {i+1}'
                plt.plot(data[i], label=label, linewidth=2)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if labels or (isinstance(data, np.ndarray) and data.ndim > 1):
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_2d_data(data, title='2D Data', cmap='viridis', 
                 vmin=None, vmax=None, save_path=None, 
                 figsize=(10, 8), colorbar=True):
    """
    Plot 2D data as a heatmap.
    
    Args:
        data (np.ndarray): 2D data to plot, shape (H, W)
        title (str): Plot title
        cmap (str): Colormap to use
        vmin (float, optional): Minimum value for colormap
        vmax (float, optional): Maximum value for colormap
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
        colorbar (bool): Whether to show colorbar
    """
    plt.figure(figsize=figsize)
    
    if data.ndim == 3:
        # If data has channels, take first channel
        data = data[0]
    
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                    aspect='auto', interpolation='bilinear')
    
    plt.title(title, fontsize=14)
    
    if colorbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(history, metrics=['loss'], save_path=None, 
                          figsize=(12, 6)):
    """
    Plot training history metrics.
    
    Args:
        history (dict): Dictionary containing training history
                       Keys should be 'train_loss', 'val_loss', etc.
        metrics (list): List of metrics to plot
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-', linewidth=2, 
                   label=f'Training {metric}')
        
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-', linewidth=2, 
                   label=f'Validation {metric}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Training History - {metric.capitalize()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def compare_predictions(input_data, target_data, predicted_data, 
                       title='Prediction Comparison', save_path=None,
                       figsize=(15, 5)):
    """
    Compare input, target, and predicted data side by side.
    
    Args:
        input_data (np.ndarray): Input data
        target_data (np.ndarray): Ground truth target data
        predicted_data (np.ndarray): Model predictions
        title (str): Overall title
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # For 2D data
    if input_data.ndim >= 2:
        if input_data.ndim == 3:
            input_data = input_data[0]
            target_data = target_data[0] if target_data.ndim == 3 else target_data
            predicted_data = predicted_data[0] if predicted_data.ndim == 3 else predicted_data
        
        vmin = min(input_data.min(), target_data.min(), predicted_data.min())
        vmax = max(input_data.max(), target_data.max(), predicted_data.max())
        
        im1 = axes[0].imshow(input_data, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Input')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(target_data, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Target')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        im3 = axes[2].imshow(predicted_data, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # For 1D data
    else:
        axes[0].plot(input_data, linewidth=2)
        axes[0].set_title('Input')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(target_data, linewidth=2)
        axes[1].set_title('Target')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(predicted_data, linewidth=2)
        axes[2].set_title('Prediction')
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities")
    print("\nExample usage:")
    print("# Plot radial data")
    print("plot_radial_data(data, labels=['Profile 1', 'Profile 2'])")
    print("\n# Plot 2D data")
    print("plot_2d_data(data_2d, title='SOLPS Data')")
    print("\n# Plot training history")
    print("plot_training_history(history)")
