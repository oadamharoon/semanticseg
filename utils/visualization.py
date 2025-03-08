import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import cv2

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

def apply_color_map(mask, color_map):
    """Apply color map to segmentation mask"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(color_map):
        rgb_mask[mask == class_idx] = color
    
    return rgb_mask

def create_overlay(image, mask, color_map, alpha=0.5):
    """Create transparent overlay of mask on image"""
    # Create colored mask
    colored_mask = apply_color_map(mask, color_map)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    return overlay

def visualize_predictions(samples, output_dir, config):
    """Visualize model predictions"""
    # Extract samples
    images = samples['images']
    masks = samples['masks']
    predictions = samples['preds']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of classes
    num_classes = config['model']['params'].get('num_classes', 1)
    
    # Generate color map
    color_map = get_color_map(num_classes)
    
    for i, (image, mask, pred) in enumerate(zip(images, masks, predictions)):
        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).numpy()
        # Denormalize image
        image_np = ((image_np * np.array([0.229, 0.224, 0.225])) + 
                   np.array([0.485, 0.456, 0.406])) * 255.0
        image_np = image_np.astype(np.uint8)
        
        # Get ground truth mask
        if mask.ndim == 3:  # One-hot encoded masks
            mask_np = mask.argmax(dim=0).numpy()
        else:  # Class indices
            mask_np = mask.numpy()
        
        # Get prediction mask
        pred_np = pred.argmax(dim=0).numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot image
        axes[0].imshow(image_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        colored_mask = apply_color_map(mask_np, color_map)
        axes[1].imshow(colored_mask)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction mask
        colored_pred = apply_color_map(pred_np, color_map)
        axes[2].imshow(colored_pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Add class legend
        if num_classes <= 10:  # Only show legend for small number of classes
            legend_elements = []
            for cls in range(num_classes):
                color = [c/255.0 for c in color_map[cls]]
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                              markerfacecolor=color, markersize=10,
                                              label=f'Class {cls}'))
            
            fig.legend(handles=legend_elements, loc='lower center', ncol=min(5, num_classes))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'sample_{i}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also save overlay images
        overlay_gt = create_overlay(image_np, mask_np, color_map)
        overlay_pred = create_overlay(image_np, pred_np, color_map)
        
        # Convert RGB to BGR for OpenCV
        overlay_gt = cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR)
        overlay_pred = cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(output_dir, f'overlay_gt_{i}.png'), overlay_gt)
        cv2.imwrite(os.path.join(output_dir, f'overlay_pred_{i}.png'), overlay_pred)

def generate_confusion_matrix(predictions, targets, num_classes, output_dir, normalize=True):
    """Generate confusion matrix for segmentation results"""
    # Flatten predictions and targets
    preds_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Create confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for i in range(len(preds_flat)):
        conf_matrix[targets_flat[i], preds_flat[i]] += 1
    
    # Normalize if requested
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = conf_matrix / (row_sums + 1e-7)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    
    # Add numbers to plot
    for i in range(num_classes):
        for j in range(num_classes):
            value = conf_matrix[i, j]
            if normalize:
                text = f'{value:.2f}'
            else:
                text = str(value)
