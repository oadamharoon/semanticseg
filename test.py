#!/usr/bin/env python
import os
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
import pandas as pd

# Import local modules
from models import get_model
from datasets.dataset_loader import load_dataloaders_from_config
from utils.metrics import MetricTracker
from utils.visualization import visualize_predictions, create_overlay

def parse_args():
    parser = argparse.ArgumentParser(description='Test semantic segmentation models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true', help='Save model predictions')
    parser.add_argument('--save-overlays', action='store_true', help='Save prediction overlays on input images')
    parser.add_argument('--save-metrics', action='store_true', help='Save metrics to CSV')
    parser.add_argument('--no-evaluation', action='store_true', help='Skip evaluation, only generate predictions')
    return parser.parse_args()

def get_color_map(num_classes):
    """Generate a color map for visualization"""
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    
    for i in range(num_classes):
        # Generate a unique color for each class
        r = (i * 100) % 255
        g = (i * 40) % 255
        b = (i * 90) % 255
        color_map[i] = [r, g, b]
    
    return color_map

def save_predictions(images, masks, predictions, filenames, output_dir, color_map):
    """Save prediction masks and overlays"""
    # Create output directories
    mask_dir = os.path.join(output_dir, 'masks')
    overlay_dir = os.path.join(output_dir, 'overlays')
    
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    for i, (image, mask, pred, filename) in enumerate(zip(images, masks, predictions, filenames)):
        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).numpy()
        # Denormalize image
        image_np = ((image_np * np.array([0.229, 0.224, 0.225])) + 
                    np.array([0.485, 0.456, 0.406])) * 255.0
        image_np = image_np.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Get class predictions
        pred_mask = pred.argmax(dim=0).numpy().astype(np.uint8)
        
        # Save prediction mask
        cv2.imwrite(os.path.join(mask_dir, f"{filename}.png"), pred_mask)
        
        # Create and save overlay
        overlay = create_overlay(image_np, pred_mask, color_map)
        cv2.imwrite(os.path.join(overlay_dir, f"{filename}.png"), overlay)

def test(model, test_loader, device, metric_tracker, config, args):
    """Test model on test set"""
    model.eval()
    
    results = {
        'images': [],
        'masks': [],
        'preds': [],
        'filenames': []
    }
    
    # For per-image metrics
    per_image_metrics = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['file_name']
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            if not args.no_evaluation:
                metric_tracker.update(outputs, masks)
            
            # Store results for saving predictions
            if args.save_predictions or args.save_overlays:
                for i in range(images.size(0)):
                    results['images'].append(images[i].cpu())
                    results['masks'].append(masks[i].cpu())
                    results['preds'].append(outputs[i].cpu())
                    results['filenames'].append(filenames[i])
            
            # Calculate per-image metrics if required
            if args.save_metrics and not args.no_evaluation:
                for i in range(images.size(0)):
                    img_metrics = {}
                    img_metrics['filename'] = filenames[i]
                    
                    # Create a temporary metric tracker for this single image
                    temp_tracker = MetricTracker(
                        metric_tracker.metric_functions.keys(),
                        metric_tracker.num_classes,
                        metric_tracker.ignore_index
                    )
                    
                    # Update with single image
                    temp_tracker.update(
                        outputs[i:i+1], masks[i:i+1]
                    )
                    
                    # Get metrics
                    img_metrics.update(temp_tracker.compute())
                    per_image_metrics.append(img_metrics)
    
    # Compute overall metrics
    metrics = {}
    if not args.no_evaluation:
        metrics = metric_tracker.compute()
        
        print("Test metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    
    # Save predictions if requested
    if (args.save_predictions or args.save_overlays) and results['images']:
        num_classes = config['model']['params'].get('num_classes', 1)
        color_map = get_color_map(num_classes)
        
        save_predictions(
            results['images'],
            results['masks'],
            results['preds'],
            results['filenames'],
            args.output_dir,
            color_map
        )
    
    # Save metrics to CSV if requested
    if args.save_metrics and not args.no_evaluation:
        # Save overall metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
        
        # Save per-image metrics
        if per_image_metrics:
            per_image_df = pd.DataFrame(per_image_metrics)
            per_image_df.to_csv(os.path.join(args.output_dir, 'per_image_metrics.csv'), index=False)
    
    return metrics

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set output directory
    if args.output_dir is None:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_dir = checkpoint_path.parent
        args.output_dir = os.path.join(checkpoint_dir, 'test_results')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloader
    dataloaders = load_dataloaders_from_config(args.config)
    if 'test' not in dataloaders:
        if 'val' in dataloaders:
            print("No test dataloader found, using validation dataloader instead")
            test_loader = dataloaders['val']
        else:
            raise ValueError("Config must include either test or validation dataset")
    else:
        test_loader = dataloaders['test']
    
    # Create model
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config.get('params', {})
    
    model = get_model(model_name, **model_params)
    model = model.to(device)
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"No checkpoint found at {args.checkpoint}")
    
    # Create metric tracker
    metric_names = config.get('metrics', ['pixel_accuracy', 'iou', 'dice_coefficient'])
    num_classes = model_params.get('num_classes', 1)
    ignore_index = config.get('loss', {}).get('params', {}).get('ignore_index', -100)
    
    metric_tracker = MetricTracker(metric_names, num_classes, ignore_index)
    
    # Test model
    metrics = test(model, test_loader, device, metric_tracker, config, args)
    
    print(f"Testing completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
