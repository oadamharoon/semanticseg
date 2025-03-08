import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type, Callable, Optional, List

# Dictionary to store all registered loss functions
_LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_loss(name: str):
    """
    Decorator for registering loss functions
    """
    def decorator(cls):
        if name in _LOSS_REGISTRY:
            raise ValueError(f"Loss function '{name}' already registered")
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name with provided parameters
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Loss function '{name}' not found. Available loss functions: {list(_LOSS_REGISTRY.keys())}")
    
    loss_cls = _LOSS_REGISTRY[name]
    return loss_cls(**kwargs)

def list_losses() -> list:
    """
    List all available loss functions
    """
    return list(_LOSS_REGISTRY.keys())

@register_loss("dice_loss")
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot encoding
        if targets.dim() == 3:
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate Dice coefficient for each class
        dice = 0
        for cls in range(num_classes):
            if cls == self.ignore_index:
                continue
                
            pred_cls = probs[:, cls, ...]
            target_cls = targets_one_hot[:, cls, ...]
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            dice += (2.0 * intersection + self.smooth) / (union + self.smooth)
            
        # Average over number of classes (excluding ignore_index)
        effective_num_classes = num_classes - (1 if self.ignore_index != -100 else 0)
        dice = dice / effective_num_classes
        
        return 1.0 - dice

@register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[List[float]] = None, ignore_index: int = -100):
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits, targets)

@register_loss("combined_loss")
class CombinedLoss(nn.Module):
    def __init__(self, losses: List[Dict], weights: Optional[List[float]] = None):
        """
        Combine multiple loss functions with weights
        
        Args:
            losses: List of dictionaries with 'name' and 'params' keys
            weights: List of weights for each loss function
        """
        super(CombinedLoss, self).__init__()
        self.losses = []
        
        for loss_dict in losses:
            name = loss_dict['name']
            params = loss_dict.get('params', {})
            self.losses.append(get_loss(name, **params))
        
        if weights is None:
            self.weights = [1.0] * len(self.losses)
        else:
            assert len(weights) == len(self.losses), "Number of weights must match number of losses"
            self.weights = weights
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for i, loss_fn in enumerate(self.losses):
            total_loss += self.weights[i] * loss_fn(logits, targets)
        return total_loss

def get_loss_from_config(loss_config: dict) -> nn.Module:
    """
    Create loss function from config
    """
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'combined_loss':
        losses = loss_config.get('losses', [])
        weights = loss_config.get('weights', None)
        return CombinedLoss(losses, weights)
    else:
        params = loss_config.get('params', {})
        return get_loss(loss_type, **params)
