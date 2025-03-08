from typing import Dict, Type, Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import cv2
import yaml

# Dictionary to store all registered datasets
_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}

def register_dataset(name: str):
    """
    Decorator for registering dataset classes
    """
    def decorator(cls):
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name: str, **kwargs) -> Dataset:
    """
    Get a dataset instance by name with provided parameters
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(_DATASET_REGISTRY.keys())}")
    
    dataset_cls = _DATASET_REGISTRY[name]
    return dataset_cls(**kwargs)

def list_datasets() -> list:
    """
    List all available datasets
    """
    return list(_DATASET_REGISTRY.keys())

def get_default_transforms(image_size: Tuple[int, int] = (512, 512), 
                           augment: bool = True) -> Dict[str, A.Compose]:
    """
    Get default transforms for training and validation
    """
    if augment:
        train_transform = A.Compose([
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }

@register_dataset("generic_segmentation")
class GenericSegmentationDataset(Dataset):
    """
    Generic dataset for semantic segmentation
    """
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 transforms: Optional[A.Compose] = None,
                 image_dir: str = 'images',
                 mask_dir: str = 'masks',
                 image_suffix: str = '.jpg',
                 mask_suffix: str = '.png',
                 augment: bool = True):
        """
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Size of the output images and masks
            transforms: Albumentations transforms
            image_dir: Directory name for images
            mask_dir: Directory name for masks
            image_suffix: File suffix for images
            mask_suffix: File suffix for masks
            augment: Whether to apply augmentations (for training split)
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.image_dir = os.path.join(root_dir, image_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        
        # Use default transforms if none provided
        if transforms is None:
            transforms_dict = get_default_transforms(image_size, augment and split == 'train')
            self.transform = transforms_dict[split]
        else:
            self.transform = transforms
        
        # Get file names
        split_file = os.path.join(root_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_names = [line.strip() for line in f]
        else:
            # If no split file exists, use all files in image_dir
            self.file_names = [f[:-len(image_suffix)] for f in os.listdir(self.image_dir) 
                               if f.endswith(image_suffix)]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # Read image
        img_path = os.path.join(self.image_dir, file_name + self.image_suffix)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_path = os.path.join(self.mask_dir, file_name + self.mask_suffix)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert to tensor if not already
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'file_name': file_name
        }

def create_dataloader(dataset_config: dict, split: str) -> DataLoader:
    """
    Create DataLoader from config
    """
    # Extract dataset parameters
    dataset_name = dataset_config.get('name', 'generic_segmentation')
    params = dataset_config.get('params', {})
    
    # Create dataset
    params['split'] = split
    dataset = get_dataset(dataset_name, **params)
    
    # Create dataloader
    loader_params = dataset_config.get('loader', {})
    batch_size = loader_params.get('batch_size', 8 if split == 'train' else 1)
    num_workers = loader_params.get('num_workers', 4)
    shuffle = loader_params.get('shuffle', split == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def load_dataloaders_from_config(config_path: str) -> Dict[str, DataLoader]:
    """
    Load all dataloaders from config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset', {})
    
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        if split in dataset_config:
            dataloaders[split] = create_dataloader(dataset_config[split], split)
    
    return dataloaders
