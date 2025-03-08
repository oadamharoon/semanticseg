#!/usr/bin/env python
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random

# Import local modules
from models import get_model, list_models
from datasets.dataset_loader import load_dataloaders_from_config
from utils.losses import get_loss_from_config
from utils.metrics import MetricTracker
from utils.visualization import visualize_predictions

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    return parser.parse_args()

def create_experiment_dir(config):
    """Create experiment directory based on config"""
    # Create a unique experiment name
    model_name = config['model']['name']
    dataset_name = config['dataset']['train']['name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_name = f"{model_name}_{dataset_name}_{timestamp}"
    exp_dir = os.path.join(config['training']['output_dir'], exp_name)
    
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'visualizations'), exist_ok=True)
    
    # Save config to experiment directory
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    return exp_dir

def train_one_epoch(model, train_loader, criterion, optimizer, device, metric_tracker, epoch):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metric_tracker.update(outputs, masks)
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Compute average loss and metrics
    epoch_loss /= len(train_loader)
    metrics = metric_tracker.compute()
    metric_tracker.reset()
    
    metrics['loss'] = epoch_loss
    return metrics

def validate(model, val_loader, criterion, device, metric_tracker, config, epoch):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0.0
    
    # For visualization
    vis_config = config.get('visualization', {})
    num_vis = vis_config.get('num_samples', 4)
    vis_samples = {'images': [], 'masks': [], 'preds': []}
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            metric_tracker.update(outputs, masks)
            val_loss += loss.item()
            
            # Store samples for visualization
            if batch_idx < num_vis:
                for i in range(min(images.size(0), 1)):  # Store only one sample per batch
                    vis_samples['images'].append(images[i].cpu())
                    vis_samples['masks'].append(masks[i].cpu())
                    vis_samples['preds'].append(outputs[i].cpu())
    
    # Compute average loss and metrics
    val_loss /= len(val_loader)
    metrics = metric_tracker.compute()
    metric_tracker.reset()
    
    metrics['loss'] = val_loss
    
    # Generate visualization if enabled
    if vis_config.get('enabled', False) and epoch % vis_config.get('frequency', 5) == 0:
        vis_dir = os.path.join(config['exp_dir'], 'visualizations', f'epoch_{epoch}')
        os.makedirs(vis_dir, exist_ok=True)
        visualize_predictions(vis_samples, vis_dir, config)
    
    return metrics

def main():
    args = parse_args()
    
    # List available models and exit if requested
    if args.list_models:
        print("Available models:")
        for model_name in list_models():
            print(f"  - {model_name}")
        return
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config)
    config['exp_dir'] = exp_dir
    print(f"Experiment directory: {exp_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = load_dataloaders_from_config(args.config)
    if 'train' not in dataloaders or 'val' not in dataloaders:
        raise ValueError("Config must include both train and validation datasets")
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Create model
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config.get('params', {})
    
    model = get_model(model_name, **model_params)
    model = model.to(device)
    
    # Create loss function
    loss_config = config.get('loss', {'type': 'cross_entropy'})
    criterion = get_loss_from_config(loss_config)
    
    # Create optimizer
    optim_config = config.get('optimizer', {'type': 'adam', 'lr': 0.001})
    optim_type = optim_config.get('type', 'adam').lower()
    lr = optim_config.get('lr', 0.001)
    
    if optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type == 'sgd':
        momentum = optim_config.get('momentum', 0.9)
        weight_decay = optim_config.get('weight_decay', 0.0001)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")
    
    # Create learning rate scheduler
    scheduler_config = config.get('scheduler', {'type': 'none'})
    scheduler_type = scheduler_config.get('type', 'none').lower()
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', config['training']['epochs'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    # Create metric tracker
    num_classes = model_params.get('num_classes', 1)
    ignore_index = loss_config.get('params', {}).get('ignore_index', -100)
    
    metric_names = config.get('metrics', ['pixel_accuracy', 'iou', 'dice_coefficient'])
    metric_tracker = MetricTracker(metric_names, num_classes, ignore_index)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    train_config = config['training']
    num_epochs = train_config.get('epochs', 100)
    save_freq = train_config.get('save_frequency', 10)
    
    best_metric = 0.0
    best_epoch = 0
    
    main_metric = train_config.get('main_metric', 'iou')
    minimize_metric = train_config.get('minimize_metric', False)
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, metric_tracker, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, metric_tracker, config, epoch
        )
        
        # Log metrics
        print(f"Epoch {epoch}:")
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        
        for metric in metric_names:
            if metric in train_metrics and metric in val_metrics:
                print(f"  Train {metric}: {train_metrics[metric]:.4f}")
                print(f"  Val {metric}: {val_metrics[metric]:.4f}")
                
                writer.add_scalar(f'Train/{metric}', train_metrics[metric], epoch)
                writer.add_scalar(f'Val/{metric}', val_metrics[metric], epoch)
        
        writer.add_scalar('Train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('Val/loss', val_metrics['loss'], epoch)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
        
        # Save best model
        current_metric = val_metrics[main_metric]
        is_best = (minimize_metric and current_metric < best_metric) or \
                 (not minimize_metric and current_metric > best_metric)
        
        if is_best or epoch == 0:
            best_metric = current_metric
            best_epoch = epoch
            
            best_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, best_path)
            
            print(f"  Best model saved (epoch {epoch}, {main_metric}: {best_metric:.4f})")
    
    # Save final model
    final_path = os.path.join(exp_dir, 'checkpoints', 'final_model.pth')
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }, final_path)
    
    print(f"Training completed. Best {main_metric}: {best_metric:.4f} at epoch {best_epoch}")
    print(f"Model saved to {exp_dir}")

if __name__ == '__main__':
    main()
