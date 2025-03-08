# Semantic Segmentation Models Repository

A flexible and extensible repository for training, testing, and deploying multiple semantic segmentation models with a unified interface.

## Features

- **Modular Design**: Easy to add new models, datasets, and functionalities
- **Config-Driven**: All parameters controlled via simple YAML configuration files
- **Unified Interface**: Train and test any model with the same commands
- **Model Registry**: Automatically register and retrieve models by name
- **Extensible**: Easily add custom models, losses, metrics, and datasets
- **Visualizations**: Automatic visualization of predictions and performance metrics
- **Metrics Tracking**: Track multiple performance metrics during training and evaluation

## Repository Structure

```
segmentation-models/
├── models/              # Model implementations
├── datasets/            # Dataset loading and preprocessing
├── utils/               # Utility functions and helpers
├── configs/             # Configuration files
├── train.py             # Unified training script
├── test.py              # Unified testing script
└── README.md            # This file
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/segmentation-models.git
cd segmentation-models
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
python train.py --config configs/unet_cityscapes.yaml
```

### Testing a Model

```bash
python test.py --config configs/unet_cityscapes.yaml --checkpoint path/to/checkpoint.pth --save-predictions
```

### Listing Available Models

```bash
python train.py --list-models
```

## Adding a New Model

1. Create a new file in the `models/` directory
2. Implement your model class
3. Add the `@register_model("model_name")` decorator to register it
4. Create a configuration file in `configs/`

Example:

```python
# models/my_model.py
import torch.nn as nn
from . import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # Implement your model here
        
    def forward(self, x):
        # Implement forward pass
        return output
```

## Adding a New Dataset

1. Create a new dataset class or use the generic one
2. Add the `@register_dataset("dataset_name")` decorator to register it
3. Update your configuration file

Example:

```python
# datasets/my_dataset.py
from torch.utils.data import Dataset
from .dataset_loader import register_dataset

@register_dataset("my_dataset")
class MyDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        # Initialize your dataset
        
    def __len__(self):
        # Return dataset length
        
    def __getitem__(self, idx):
        # Return a sample
```

## Configuration Files

Configuration files use YAML format and control all aspects of training and testing:

```yaml
# Model configuration
model:
  name: 'model_name'
  params:
    # Model-specific parameters

# Dataset configuration
dataset:
  train:
    name: 'dataset_name'
    params:
      # Dataset parameters
    loader:
      # DataLoader parameters

# Loss function configuration
loss:
  type: 'loss_name'
  params:
    # Loss function parameters

# Optimizer configuration
optimizer:
  type: 'optimizer_name'
  lr: 0.001
  # Other optimizer parameters

# Training configuration
training:
  epochs: 100
  save_frequency: 10
  # Other training parameters
```

## Available Models

- **UNet**: Classic encoder-decoder architecture with skip connections
- Add more models here...

## Available Datasets

- **GenericSegmentationDataset**: Flexible dataset for common segmentation formats
- Add more datasets here...

## Available Loss Functions

- **CrossEntropyLoss**: Standard cross-entropy loss for segmentation
- **DiceLoss**: Dice coefficient loss
- **CombinedLoss**: Weighted combination of multiple losses

## Available Metrics

- **Pixel Accuracy**: Percentage of correctly classified pixels
- **IoU (Jaccard Index)**: Intersection over Union
- **Dice Coefficient**: F1 score for segmentation

## License

[MIT License](LICENSE)

## Acknowledgements

- List any libraries, papers, or projects that inspired this repository
