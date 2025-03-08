import torch
import numpy as np
from typing import Dict, Callable, List, Union, Tuple

# Dictionary to store all registered metrics
_METRIC_REGISTRY: Dict[str, Callable] = {}

def register_metric(name: str):
    """
    Decorator for registering metrics
    """
    def decorator(func):
        if name in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' already registered")
        _METRIC_REGISTRY[name] = func
        return func
    return decorator

def get_metric(name: str) -> Callable:
    """
    Get a metric function by name
    """
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' not found. Available metrics: {list(_METRIC_REGISTRY.keys())}")
    
    return _METRIC_REGISTRY[name]

def list_metrics() -> list:
    """
    List all available metrics
    """
    return list(_METRIC_REGISTRY.keys())

@register_metric("pixel_accuracy")
def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Calculate pixel accuracy
    
    Args:
        pred: Prediction tensor of shape (N, C, H, W) with logits
        target: Target tensor of shape (N, H, W) with class indices
        ignore_index: Index to ignore in the evaluation
        
    Returns:
        Pixel accuracy as a float
    """
    pred = pred.argmax(dim=1)
    mask = target != ignore_index
    correct = (pred == target) & mask
    return correct.sum().float() / mask.sum().float()

@register_metric("iou")
def iou(pred: torch.Tensor, target: torch.Tensor, 
        num_classes: int, ignore_index: int = -100) -> Tuple[float, List[float]]:
    """
    Calculate IoU (Jaccard index)
    
    Args:
        pred: Prediction tensor of shape (N, C, H, W) with logits
        target: Target tensor of shape (N, H, W) with class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in the evaluation
        
    Returns:
        Tuple of (mean_iou, class_ious)
    """
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    # Initialize arrays for intersection and union
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection[cls] = np.logical_and(pred_cls, target_cls).sum()
        union[cls] = np.logical_or(pred_cls, target_cls).sum()
    
    # Calculate IoU for each class
    class_ious = np.zeros(num_classes)
    valid_classes = 0
    
    for cls in range(num_classes):
        if cls == ignore_index or union[cls] == 0:
            class_ious[cls] = float('nan')
        else:
            class_ious[cls] = intersection[cls] / union[cls]
            valid_classes += 1
    
    # Calculate mean IoU over valid classes
    mean_iou = np.nanmean(class_ious)
    
    return mean_iou, class_ious.tolist()

@register_metric("dice_coefficient")
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                    num_classes: int, ignore_index: int = -100) -> Tuple[float, List[float]]:
    """
    Calculate Dice coefficient
    
    Args:
        pred: Prediction tensor of shape (N, C, H, W) with logits
        target: Target tensor of shape (N, H, W) with class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in the evaluation
        
    Returns:
        Tuple of (mean_dice, class_dices)
    """
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    # Initialize arrays for intersection and areas
    intersection = np.zeros(num_classes)
    pred_area = np.zeros(num_classes)
    target_area = np.zeros(num_classes)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection[cls] = np.logical_and(pred_cls, target_cls).sum()
        pred_area[cls] = pred_cls.sum()
        target_area[cls] = target_cls.sum()
    
    # Calculate Dice for each class
    class_dices = np.zeros(num_classes)
    valid_classes = 0
    
    for cls in range(num_classes):
        if cls == ignore_index or (pred_area[cls] + target_area[cls]) == 0:
            class_dices[cls] = float('nan')
        else:
            class_dices[cls] = 2.0 * intersection[cls] / (pred_area[cls] + target_area[cls])
            valid_classes += 1
    
    # Calculate mean Dice over valid classes
    mean_dice = np.nanmean(class_dices)
    
    return mean_dice, class_dices.tolist()

class MetricTracker:
    """
    Track and compute multiple metrics during training/evaluation
    """
    def __init__(self, metric_names: List[str], num_classes: int, ignore_index: int = -100):
        """
        Initialize metric tracker
        
        Args:
            metric_names: List of metric names to track
            num_classes: Number of classes in the segmentation task
            ignore_index: Index to ignore in evaluation
        """
        self.metric_functions = {}
        for name in metric_names:
            self.metric_functions[name] = get_metric(name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """
        Reset all tracked metrics
        """
        self.metrics = {name: [] for name in self.metric_functions.keys()}
        self.class_metrics = {}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new batch
        
        Args:
            pred: Prediction tensor of shape (N, C, H, W) with logits
            target: Target tensor of shape (N, H, W) with class indices
        """
        with torch.no_grad():
            for name, metric_fn in self.metric_functions.items():
                if name in ["iou", "dice_coefficient"]:
                    mean_value, class_values = metric_fn(
                        pred, target, self.num_classes, self.ignore_index
                    )
                    self.metrics[name].append(mean_value)
                    
                    if name not in self.class_metrics:
                        self.class_metrics[name] = [[] for _ in range(self.num_classes)]
                    
                    for cls in range(self.num_classes):
                        if not np.isnan(class_values[cls]):
                            self.class_metrics[name][cls].append(class_values[cls])
                else:
                    value = metric_fn(pred, target, self.ignore_index)
                    self.metrics[name].append(value.item())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute averaged metrics
        
        Returns:
            Dictionary of metric name to averaged value
        """
        result = {}
        
        # Compute mean for each metric
        for name, values in self.metrics.items():
            if values:
                result[name] = np.mean(values)
            else:
                result[name] = 0.0
        
        # Compute class-wise metrics
        for name, class_values in self.class_metrics.items():
            for cls in range(self.num_classes):
                if class_values[cls]:
                    result[f"{name}_class_{cls}"] = np.mean(class_values[cls])
                else:
                    result[f"{name}_class_{cls}"] = 0.0
        
        return result
