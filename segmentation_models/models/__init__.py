from typing import Dict, Type, Any, Optional
import torch.nn as nn

# Dictionary to store all registered models
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """
    Decorator for registering model classes
    """
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model with name '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Get a model instance by name with provided parameters
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(_MODEL_REGISTRY.keys())}")
    
    model_cls = _MODEL_REGISTRY[name]
    return model_cls(**kwargs)

def list_models() -> list:
    """
    List all available models
    """
    return list(_MODEL_REGISTRY.keys())
