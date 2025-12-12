# APP-FPP-ML

ML/AI models developed for the APP-FPP project. This repository contains neural network models for processing plasma physics data from fusion simulations.

## Models

This repository includes two main models:

### 1. 1D Neural Network for Radial Data (`models/nn_1d/`)
A fully connected neural network designed to process 1D radial profiles from plasma simulations. This model learns mappings from input radial profiles to output predictions using deep fully connected layers with batch normalization and dropout for regularization.

**Features:**
- Configurable architecture with multiple hidden layers
- Batch normalization for stable training
- Dropout for regularization
- Efficient training and inference
- Support for various radial profile sizes

### 2. 2D UNet for SOLPS Data (`models/unet_2d/`)
A UNet architecture for processing 2D plasma simulation data from SOLPS (Scrape-Off Layer Plasma Simulation). This model uses an encoder-decoder structure with skip connections to learn complex 2D spatial patterns.

**Features:**
- Standard UNet architecture with skip connections
- Configurable depth and channel sizes
- Bilinear or transposed convolution upsampling
- Support for multi-channel inputs/outputs
- Tiling strategy for large images

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
APP-FPP-ML/
├── models/
│   ├── nn_1d/              # 1D Neural Network for radial data
│   │   ├── __init__.py
│   │   ├── model.py        # Model architecture
│   │   ├── train.py        # Training script
│   │   └── predict.py      # Inference script
│   └── unet_2d/            # 2D UNet for SOLPS data
│       ├── __init__.py
│       ├── model.py        # UNet architecture
│       ├── train.py        # Training script
│       └── predict.py      # Inference script
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   └── visualization.py    # Plotting and visualization
├── configs/
│   ├── nn_1d_config.json   # Configuration for 1D NN
│   └── unet_2d_config.json # Configuration for UNet
├── examples/
│   ├── train_nn_1d.py      # Example training script for 1D NN
│   ├── train_unet_2d.py    # Example training script for UNet
│   ├── inference_nn_1d.py  # Example inference script for 1D NN
│   └── inference_unet_2d.py # Example inference script for UNet
├── data/                    # Data directory (not included in repo)
├── checkpoints/             # Model checkpoints (created during training)
└── requirements.txt         # Python dependencies
```

## Quick Start

### Training the 1D Neural Network

```python
import torch
from models.nn_1d.model import RadialNN
from models.nn_1d.train import RadialNNTrainer

# Initialize model
model = RadialNN(input_size=64, hidden_sizes=[128, 256, 128], output_size=64)

# Initialize trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = RadialNNTrainer(model, device=device, learning_rate=1e-3)

# Train (requires data loaders)
trainer.train(train_loader, val_loader, epochs=100)
```

Or use the example script:

```bash
python examples/train_nn_1d.py
```

### Training the 2D UNet

```python
import torch
from models.unet_2d.model import UNet2D
from models.unet_2d.train import UNet2DTrainer

# Initialize model
model = UNet2D(n_channels=1, n_classes=1, base_channels=64)

# Initialize trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = UNet2DTrainer(model, device=device, learning_rate=1e-4)

# Train (requires data loaders)
trainer.train(train_loader, val_loader, epochs=100)
```

Or use the example script:

```bash
python examples/train_unet_2d.py
```

### Inference

#### 1D Neural Network

```python
from models.nn_1d.predict import RadialNNPredictor
import numpy as np

# Load trained model
predictor = RadialNNPredictor('checkpoints/nn_1d/best_model.pth')

# Make prediction
input_profile = np.random.randn(64)
prediction = predictor.predict(input_profile)
```

Or use the example script:

```bash
python examples/inference_nn_1d.py
```

#### 2D UNet

```python
from models.unet_2d.predict import UNet2DPredictor
import numpy as np

# Load trained model
predictor = UNet2DPredictor('checkpoints/unet_2d/best_model.pth')

# Make prediction
input_field = np.random.randn(256, 256)
prediction = predictor.predict(input_field)
```

Or use the example script:

```bash
python examples/inference_unet_2d.py
```

## Data Format

### 1D Radial Data
- Format: NumPy arrays (.npy), HDF5 (.h5), or text files
- Shape: `(n_samples, n_points)` for inputs and targets
- The data should represent radial profiles from plasma simulations

### 2D SOLPS Data
- Format: NumPy arrays (.npy) or HDF5 (.h5)
- Shape: `(n_samples, channels, height, width)`
- The data should represent 2D spatial distributions from SOLPS simulations

## Configuration

Model configurations are stored in JSON files in the `configs/` directory. You can modify these files to adjust:
- Model architecture (layer sizes, channels, etc.)
- Training parameters (learning rate, batch size, epochs)
- Data paths and preprocessing options
- Checkpoint settings

## Model Testing

Test the models by running them directly:

```bash
# Test 1D NN model
python -m models.nn_1d.model

# Test UNet model
python -m models.unet_2d.model
```

## Visualization

The `utils/visualization.py` module provides functions for:
- Plotting radial profiles
- Visualizing 2D fields
- Comparing predictions with ground truth
- Plotting training history

Example:

```python
from utils.visualization import plot_radial_data, plot_2d_data

# Plot radial profile
plot_radial_data(profile_data, title='Radial Profile')

# Plot 2D field
plot_2d_data(field_data, title='SOLPS Data')
```

## Hardware Requirements

- **CPU**: Modern multi-core processor
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for 2D UNet)
- **Storage**: Depends on dataset size

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is part of the APP-FPP research project.

## Citation

If you use this code in your research, please cite the APP-FPP project.

## Contact

For questions or issues, please open an issue on GitHub.
