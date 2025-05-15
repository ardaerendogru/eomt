import yaml
import os
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import numpy as np
import importlib
import warnings
from tqdm import tqdm
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import csv
import json
from torchmetrics.detection import PanopticQuality
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _Color,
    _get_color_areas,
    _calculate_iou,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Panoptic segmentation inference script')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (ignored if --weights_file is provided)')
    parser.add_argument('--weights_file', type=str, default=None, 
                        help='Path to a text file with experiment_name,weight_path pairs')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--device', type=int, default=0, help='GPU device id to use')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
    parser.add_argument('--save_mode', type=str, default='both', choices=['pred', 'target', 'both'], 
                        help='What to save: predictions, targets, or both')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process (default: all)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--with_legend', action='store_true', help='Include class legend in output')
    parser.add_argument('--class_names_file', type=str, default=None, 
                        help='Path to file containing class names (if None, generic class names will be used)')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--num_gpus', type=int, default=1, 
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--distributed', action='store_true',
                        help='Use torch.nn.parallel.DistributedDataParallel instead of DataParallel')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip metrics calculation entirely for faster visualization')
    return parser.parse_args()

# ADE20K class names - can be loaded from a file if needed
ADE20K_CLASS_NAMES = [
    "background", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "window", 
    "grass", "cabinet", "sidewalk", "person", "ground", "door", "table", "mountain", "plant", "curtain",
    "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "carpet",
    "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing",
    "cushion", "pedestal", "box", "pillar", "sign", "dresser", "counter", "sand", "sink", "skyscraper",
    "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "showcase", "pool-table", "pillow", "screen-door",
    "stairway", "river", "bridge", "bookcase", "blind", "coffee-table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen-island", "computer", "swivel-chair", "boat", "bar", "arcade-machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television", "airplane", "dirt-track", "clothes", "pole", "soil", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer-belt", "canopy", "washing-machine", "toy",
    "swimming-pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "motorbike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade-name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic-light", "tray", "trash-can",
    "fan", "pier", "crt-screen", "plate", "monitor", "bulletin-board", "shower", "radiator", "glass", "clock", "flag"
]

def load_class_names(file_path=None):
    """Load class names from file or use default ADE20K class names"""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return [line.strip().split('\t')[-1] for line in f.readlines()]
    return ADE20K_CLASS_NAMES

def get_short_class_name(class_name):
    """Get the first part of the class name (before any comma)"""
    return class_name.split(',')[0].strip()

def create_mapping(images, ignore_index=-1):
    """Create a consistent color mapping for segmentation classes"""
    unique_ids = np.unique(np.concatenate([np.unique(img) for img in images]))
    valid_ids = unique_ids[unique_ids != ignore_index]
    # Use a fixed colormap for consistency between runs
    np.random.seed(42)  # Fixed seed for color consistency
    colors = np.array([plt.cm.hsv(i / len(valid_ids))[:3] for i in range(len(valid_ids))])
    np.random.seed(None)  # Reset seed
    
    mapping = {cid: colors[i] for i, cid in enumerate(valid_ids)}
    mapping[ignore_index] = np.array([0, 0, 0])
    return mapping

def draw_black_border(sem, inst, mapping):
    """Draw black borders between different instances - optimized version"""
    h, w = sem.shape
    out = np.zeros((h, w, 3))
    
    # Vectorized color assignment for better performance
    unique_sem_values = np.unique(sem)
    for s in unique_sem_values:
        if s in mapping:
            mask = sem == s
            out[mask] = mapping[s]
        else:
            # Use black for any class not in mapping (like -1)
            mask = sem == s
            out[mask] = np.array([0, 0, 0])

    # Combine semantic and instance IDs to identify object boundaries
    # Use int32 instead of int64 when possible to reduce memory usage
    dtype = np.int32 if np.max(sem) * 100000 + np.max(inst) < 2**31 else np.int64
    combined = sem.astype(dtype) * 100000 + inst.astype(dtype)
    
    # Compute boundaries more efficiently
    border = np.zeros((h, w), dtype=bool)
    
    # Horizontal boundaries
    border[:-1, :] |= combined[:-1, :] != combined[1:, :]
    # Already included in the previous operation: border[1:, :] |= combined[1:, :] != combined[:-1, :]
    
    # Vertical boundaries
    border[:, :-1] |= combined[:, :-1] != combined[:, 1:]
    # Already included in the previous operation: border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    
    # Apply border in one step
    out[border] = 0
    return out

def infer_panoptic(model, img, target, device):
    """Run panoptic segmentation inference"""
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]

        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)

        preds = model.to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            model.mask_thresh,
            model.overlap_thresh,
        )[0].cpu()

    pred = preds.numpy()
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]

    target_seg = model.to_per_pixel_targets_panoptic([target])[0].cpu().numpy()
    sem_target, inst_target = target_seg[..., 0], target_seg[..., 1]

    # Also return is_crowd information for PQ calculation
    is_crowd = target.get("is_crowd", torch.zeros(len(target["labels"]), dtype=torch.bool)).cpu()
    return sem_pred, inst_pred, sem_target, inst_target, is_crowd

def create_class_legend(class_ids, class_colors, class_names):
    """Create a separate legend image with class names and colors"""
    num_classes = len(class_ids)
    
    # Calculate size and layout
    fig_height = min(15, max(5, num_classes * 0.25))  # Smaller height per item
    cols = max(1, min(4, num_classes // 15 + 1))
    rows = (num_classes + cols - 1) // cols
    
    # Create figure and axis properly
    fig = plt.figure(figsize=(cols * 4, fig_height), dpi=150)
    ax = fig.add_subplot(111)
    
    # Create legend elements
    legend_elements = []
    for i, class_id in enumerate(class_ids):
        if 0 <= class_id < len(class_names):
            full_name = class_names[class_id]
            short_name = get_short_class_name(full_name)
            class_name = f"{class_id}: {short_name}"
        else:
            class_name = f"Class {class_id}"
        legend_elements.append(Patch(facecolor=class_colors[i], edgecolor='black', label=class_name))
    
    # Create the legend with smaller font
    ax.legend(handles=legend_elements, loc='center', ncol=cols, frameon=True, 
              fontsize='x-small', bbox_to_anchor=(0.5, 0.5))
    ax.axis('off')
    
    return fig

def adjust_text_position(cx, cy, img_width, img_height, padding=10):
    """Adjust text position to ensure it stays within image boundaries"""
    # Add padding to keep text away from edges
    cx = max(padding, min(img_width - padding, cx))
    cy = max(padding, min(img_height - padding, cy))
    return cx, cy

def save_panoptic_results(img, sem_pred, inst_pred, sem_target, inst_target, 
                         model_num_classes, filename, save_mode, output_dir, 
                         with_legend=False, class_names=None, exp_name=None, is_first_exp=False):
    """Save panoptic segmentation results as images and return instance counts for pred and target."""
    # Create output directory at the beginning
    os.makedirs(output_dir, exist_ok=True)
    
    # Add experiment name suffix if provided
    suffix = f"_{exp_name}" if exp_name else ""
    
    # Create a consistent color mapping - only compute once if possible
    all_ids = np.union1d(np.unique(sem_pred), np.unique(sem_target))
    
    # Make sure to include -1 (background or ignore index) in the mapping
    # Use a faster approach to create mapping
    np.random.seed(42)  # Fixed seed for color consistency
    mapping = {}
    for i, s in enumerate(all_ids):
        if s == -1:
            mapping[s] = np.array([0, 0, 0])
        # Don't add class 150 (model_num_classes) to the mapping
        elif s != model_num_classes:
            mapping[s] = plt.cm.hsv(i / max(1, len(all_ids)))[:3]
    np.random.seed(None)  # Reset seed
    
    # Ensure -1 is always in the mapping
    if -1 not in mapping:
        mapping[-1] = np.array([0, 0, 0])
    
    # Prep instance counts dicts
    pred_instance_counts = None
    target_instance_counts = None
    
    # Save input image (only once if we have multiple experiments)
    if is_first_exp:
        img_np = img.cpu().numpy().transpose(1, 2, 0) if img.dim() == 3 else img.cpu().numpy()
        
        # Normalize image for proper display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_input.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # Prepare instance counting
    if save_mode in ['pred', 'both']:
        # Count instances for predictions
        pred_instance_counts = {}
        for cls in np.unique(sem_pred):
            if cls < 0 or cls >= len(class_names):
                continue
            mask = sem_pred == cls
            instance_ids = np.unique(inst_pred[mask])
            valid_instance_ids = [iid for iid in instance_ids if iid >= 0]
            class_name = class_names[int(cls)]
            pred_instance_counts[class_name] = len(valid_instance_ids)
    
    if save_mode in ['target', 'both'] and is_first_exp:
        # Count instances for targets
        target_instance_counts = {}
        for cls in np.unique(sem_target):
            if cls < 0 or cls >= len(class_names):
                continue
            mask = sem_target == cls
            instance_ids = np.unique(inst_target[mask])
            valid_instance_ids = [iid for iid in instance_ids if iid >= 0]
            class_name = class_names[int(cls)]
            target_instance_counts[class_name] = len(valid_instance_ids)
    
    # Helper function to add class labels to an image efficiently
    def add_class_labels(sem_img, inst_img, class_names, h, w, ax):
        # Get unique semantic classes in the image
        unique_classes = np.unique(sem_img)
        from scipy import ndimage
        
        # Process all classes at once where possible
        for cls in unique_classes:
            if cls >= 0 and cls != model_num_classes:  # Skip background, invalid classes, and class 150
                mask = sem_img == cls
                area = np.sum(mask)
                labeled, num = ndimage.label(mask)
                
                if num > 0:
                    # Process each connected component separately
                    for component_id in range(1, num+1):
                        component_mask = labeled == component_id
                        component_area = np.sum(component_mask)
                        
                        # Skip very small regions
                        if component_area < 50:
                            continue
                        
                        # Find bounding box of component to place text more accurately
                        # This helps ensure text is inside the region
                        y_indices, x_indices = np.where(component_mask)
                        if len(y_indices) == 0 or len(x_indices) == 0:
                            continue
                            
                        # Find center coordinates inside the mask
                        cy, cx = ndimage.center_of_mass(component_mask)
                        
                        # For extremely thin or small regions, use midpoint of bounding box instead
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        
                        # If region is very elongated, use middle of bounding box
                        aspect_ratio = max((y_max - y_min + 1) / max(1, (x_max - x_min + 1)),
                                          (x_max - x_min + 1) / max(1, (y_max - y_min + 1)))
                        if aspect_ratio > 3 or component_area < 300:
                            cy = (y_min + y_max) / 2
                            cx = (x_min + x_max) / 2
                        
                        # Adjust to ensure we're inside the mask
                        # Sample a few points near center to find a good placement
                        radius = min(10, int(min(component_area**0.5 / 4, 20)))
                        best_cx, best_cy = cx, cy
                        
                        # Only do point sampling for medium to large regions
                        if component_area > 500:
                            offset_points = [(0, 0)]  # Start with center
                            for r in range(1, radius+1, 2):
                                for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r)]:
                                    offset_points.append((dx, dy))
                            
                            # Try points until we find one inside the mask
                            for dx, dy in offset_points:
                                test_x, test_y = int(cx + dx), int(cy + dy)
                                if (0 <= test_y < h and 0 <= test_x < w and component_mask[test_y, test_x]):
                                    best_cx, best_cy = test_x, test_y
                                    break
                                    
                        # Stay inside image boundaries
                        cx, cy = adjust_text_position(best_cx, best_cy, w, h)
                        
                        if cls < len(class_names):
                            full_name = class_names[cls]
                            cls_name = get_short_class_name(full_name)
                        else:
                            cls_name = ""
                            
                        # Adaptive font size: larger for big regions, smaller for small
                        font_size = 6
                        if component_area > 10000:
                            font_size = 10
                        elif component_area > 3000:
                            font_size = 8
                        elif component_area < 500:
                            font_size = 5
                            
                        text = ax.text(cx, cy, f"{cls_name}", 
                                       color='white', fontsize=font_size, ha='center', va='center')
                        text.set_path_effects([
                            path_effects.Stroke(linewidth=1.5, foreground='black'),
                            path_effects.Normal()
                        ])
    
    # Save prediction if requested
    if save_mode in ['pred', 'both']:
        vis_pred = draw_black_border(sem_pred, inst_pred, mapping)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(vis_pred)
        
        # Get image dimensions
        h, w = sem_pred.shape
        
        # Add class labels - always do this regardless of with_legend flag
        add_class_labels(sem_pred, inst_pred, class_names, h, w, ax)
                
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_pred{suffix}.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
    
    # Save target if requested (only once if we have multiple experiments)
    if (save_mode in ['target', 'both']) and is_first_exp:
        vis_target = draw_black_border(sem_target, inst_target, mapping)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(vis_target)
        
        # Get image dimensions
        h, w = sem_target.shape
        
        # Add class labels to ground truth as well
        add_class_labels(sem_target, inst_target, class_names, h, w, ax)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_target.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
    
    # Create and save a separate legend image only if explicitly requested
    if with_legend:
        # Extract class colors for legend
        legend_class_ids = []
        legend_class_colors = []
        
        # For visualization, we want only classes that appear in the image
        classes_in_image = set()
        if save_mode in ['pred', 'both']:
            classes_in_image.update(np.unique(sem_pred))
        if save_mode in ['target', 'both']:
            classes_in_image.update(np.unique(sem_target))
        
        # Remove background class if present
        if -1 in classes_in_image:
            classes_in_image.remove(-1)
        
        # Sort class IDs for a consistent legend
        for cls_id in sorted(classes_in_image):
            if cls_id >= 0 and cls_id < model_num_classes:  # Skip background and invalid classes
                legend_class_ids.append(cls_id)
                legend_class_colors.append(mapping.get(cls_id, [0, 0, 0]))
        
        if legend_class_ids:
            legend_fig = create_class_legend(legend_class_ids, legend_class_colors, class_names)
            legend_fig.savefig(os.path.join(output_dir, f"{filename}_legend{suffix}.png"), 
                             bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close(legend_fig)

    return pred_instance_counts, target_instance_counts

def load_model(config, device, weights_path, data):
    """Load the model with specified weights"""
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=data.img_size, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
    }
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=data.num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name.replace("ADE20K",""))
    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items() if k != "network"
    }
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

        # Initialize model
    model = (
        lit_cls(
            img_size=data.img_size,
            num_classes=data.num_classes,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    # Load model weights
    try:
        obj = torch.load(weights_path, map_location=device)
        if hasattr(obj, 'state_dict'):
            model.load_state_dict(obj.state_dict())
        elif isinstance(obj, dict) and 'state_dict' in obj:
            model.load_state_dict(obj['state_dict'])
        else:
            model.load_state_dict(obj)
        print(f"Loaded model weights from {weights_path}")
        return model
    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        return None

def load_weights_from_file(weights_file):
    """Load experiment names and weights paths from a text file"""
    experiments = []
    try:
        with open(weights_file, 'r') as f:
            # First try CSV format with comma delimiter
            try:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and not row[0].startswith('#'):
                        exp_name = row[0].strip()
                        weight_path = row[1].strip()
                        experiments.append((exp_name, weight_path))
            except:
                # If CSV parsing fails, try simple line-by-line parsing
                f.seek(0)  # Reset file position
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',', 1)
                        if len(parts) >= 2:
                            exp_name = parts[0].strip()
                            weight_path = parts[1].strip()
                            experiments.append((exp_name, weight_path))
        
        if experiments:
            print(f"Loaded {len(experiments)} experiment(s) from {weights_file}")
        else:
            print(f"No valid experiment entries found in {weights_file}")
    except Exception as e:
        print(f"Error loading weights file {weights_file}: {e}")
    
    return experiments

def infer_panoptic_batch(model, batch_imgs, batch_targets, device):
    """Run panoptic segmentation inference on a batch of images"""
    batch_results = []
    
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        # Move all images to device at once
        batch_imgs = [img.to(device) for img in batch_imgs]
        img_sizes = [img.shape[-2:] for img in batch_imgs]

        # Use module if available (for DataParallel model)
        model_module = model.module if hasattr(model, 'module') else model
        
        # Create a batch tensor - stack the images
        transformed_imgs = torch.stack([
            model_module.resize_and_pad_imgs_instance_panoptic([img])[0] 
            for img in batch_imgs
        ])
        
        # Process the batch - this will be distributed across GPUs with DataParallel
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        
        # Get the last layer logits
        last_mask_logits = mask_logits_per_layer[-1]
        last_class_logits = class_logits_per_layer[-1]
        
        # Process each image in the batch - keep everything on GPU until the end
        for i in range(len(batch_imgs)):
            # Get mask logits for this image and interpolate
            single_mask_logits = last_mask_logits[i:i+1]
            img_size = model_module.img_size
            mask_logits = F.interpolate(single_mask_logits, img_size, mode="bilinear")
            mask_logits = model_module.revert_resize_and_pad_logits_instance_panoptic(mask_logits, [img_sizes[i]])
            
            # Generate prediction
            pred = model_module.to_per_pixel_preds_panoptic(
                mask_logits,
                last_class_logits[i:i+1],
                model_module.stuff_classes,
                model_module.mask_thresh,
                model_module.overlap_thresh,
            )[0]
            
            # Convert to numpy only at the end to reduce GPU-CPU transfers
            pred_np = pred.cpu().numpy()
            sem_pred, inst_pred = pred_np[..., 0], pred_np[..., 1]
            
            # Get target (also keep on GPU as long as possible)
            target_seg = model_module.to_per_pixel_targets_panoptic([batch_targets[i]])[0]
            target_np = target_seg.cpu().numpy()
            sem_target, inst_target = target_np[..., 0], target_np[..., 1]
            
            # Get is_crowd information
            is_crowd = batch_targets[i].get("is_crowd", torch.zeros(len(batch_targets[i]["labels"]), dtype=torch.bool)).cpu()
            
            batch_results.append((sem_pred, inst_pred, sem_target, inst_target, is_crowd))
    
    return batch_results

def calculate_metrics_panoptic(pred, target, is_crowd, thing_classes, stuff_classes, cuda=False):
    """
    Calculate PQ metrics using optimized approach from LightningModule.
    Returns dict with overall metrics and per-class metrics.
    
    Args:
        pred: Prediction tensor of shape [H,W,2] with semantic and instance IDs
        target: Target tensor of shape [H,W,2] with semantic and instance IDs
        is_crowd: Boolean tensor indicating if instances are crowd regions
        thing_classes: List of class IDs that are "things"
        stuff_classes: List of class IDs that are "stuff"
        cuda: Whether to run calculations on GPU (faster if data already on GPU)
        
    Returns:
        Dict with metrics: overall_pq, overall_sq, overall_rq and per-class metrics
    """
    # Create PQ metric
    pq_metric = PanopticQuality(
        thing_classes,
        stuff_classes,
        return_sq_and_rq=True,
        return_per_class=True,
    )
    
    # Convert numpy arrays to tensors if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).long()
        # Add batch dimension if needed
        if len(pred.shape) == 3:  # H,W,2
            pred = pred.unsqueeze(0)
    
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
        # Add batch dimension if needed
        if len(target.shape) == 3:  # H,W,2
            target = target.unsqueeze(0)
    
    # Move tensors to GPU if requested
    if cuda and torch.cuda.is_available():
        pred = pred.cuda()
        target = target.cuda()
        if isinstance(is_crowd, torch.Tensor):
            is_crowd = is_crowd.cuda()
    
    # Process inputs - use device of input tensors
    device = pred.device
    
    # Process inputs
    flatten_pred = _prepocess_inputs(
        pq_metric.things,
        pq_metric.stuffs,
        pred,
        pq_metric.void_color,
        pq_metric.allow_unknown_preds_category,
    )[0]
    
    flatten_target = _prepocess_inputs(
        pq_metric.things,
        pq_metric.stuffs,
        target,
        pq_metric.void_color,
        True,
    )[0]

    # Get areas - these operations are faster on GPU
    pred_areas = _get_color_areas(flatten_pred)
    target_areas = _get_color_areas(flatten_target)
    
    # Skip if no valid areas
    if not pred_areas or not target_areas:
        return {
            "overall_pq": 0.0,
            "overall_sq": 0.0,
            "overall_rq": 0.0,
            "per_class_pq": {},
            "per_class_sq": {},
            "per_class_rq": {},
        }
    
    # Calculate intersection areas
    intersection_matrix = torch.transpose(
        torch.stack((flatten_pred, flatten_target), -1), -1, -2
    )
    intersection_areas = _get_color_areas(intersection_matrix)

    # Initialize tracking
    pred_segment_matched = set()
    target_segment_matched = set()
    
    # Track metrics per class - identify unique class IDs efficiently
    classes_in_target = set()
    for target_color in target_areas:
        if target_color != pq_metric.void_color:
            class_id = target_color[0]
            classes_in_target.add(class_id)
    
    # Initialize tracking dictionaries
    true_positives = {cls_id: 0 for cls_id in classes_in_target}
    false_positives = {cls_id: 0 for cls_id in classes_in_target}
    false_negatives = {cls_id: 0 for cls_id in classes_in_target}
    iou_sum = {cls_id: 0.0 for cls_id in classes_in_target}

    # Process true positives and calculate IoU values - optimized loop
    # Pre-filter intersection pairs for better performance
    valid_intersections = []
    for pred_color, target_color in intersection_areas:
        # Skip irrelevant pairs quickly
        if target_color == pq_metric.void_color or pred_color[0] != target_color[0]:
            continue
        
        class_id = target_color[0]
        if class_id not in classes_in_target:
            continue
        
        # Skip crowd regions
        if isinstance(is_crowd, torch.Tensor) and is_crowd.numel() > target_color[1] and is_crowd[target_color[1]]:
            continue
            
        valid_intersections.append((pred_color, target_color))
    
    # Now process only valid intersections
    for pred_color, target_color in valid_intersections:
        class_id = target_color[0]
        # Calculate IoU - this is the most expensive operation
        iou = _calculate_iou(
            pred_color,
            target_color,
            pred_areas,
            target_areas,
            intersection_areas,
            pq_metric.void_color,
        )
        
        # Count as match if IoU > 0.5
        if iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            true_positives[class_id] += 1
            iou_sum[class_id] += iou

    # Handle false positives and negatives - we can move these sets to CPU for processing
    false_negative_colors = set(target_areas) - target_segment_matched
    false_positive_colors = set(pred_areas) - pred_segment_matched
    
    # Remove void pixels from errors
    false_negative_colors.discard(pq_metric.void_color)
    false_positive_colors.discard(pq_metric.void_color)
    
    # Count false negatives - process in bulk where possible
    for false_negative_color in false_negative_colors:
        # Skip crowd regions
        if isinstance(is_crowd, torch.Tensor) and is_crowd.numel() > false_negative_color[1] and is_crowd[false_negative_color[1]]:
            continue
            
        class_id = false_negative_color[0]
        if class_id in classes_in_target:
            false_negatives[class_id] += 1
    
    # Process false positives - only count classes in target
    for false_positive_color in false_positive_colors:
        class_id = false_positive_color[0]
        if class_id in classes_in_target:
            false_positives[class_id] += 1
    
    # Calculate metrics - these calculations are very fast
    # Calculate overall metrics across all classes in target
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    total_iou = sum(iou_sum.values())
    
    # Calculate overall PQ, SQ, RQ
    overall_sq = total_iou / total_tp if total_tp > 0 else 0.0
    overall_rq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    overall_pq = overall_sq * overall_rq
    
    # Calculate per-class PQ, SQ, RQ - only for classes in target
    pq_per_class_dict = {}
    sq_per_class_dict = {}
    rq_per_class_dict = {}
    
    for cls_id in classes_in_target:
        tp = true_positives[cls_id]
        fp = false_positives[cls_id]
        fn = false_negatives[cls_id]
        class_iou = iou_sum[cls_id]
        
        if tp > 0:
            sq = class_iou / tp
            rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
            pq = sq * rq
        else:
            sq, rq, pq = 0.0, 0.0, 0.0
        
        pq_per_class_dict[int(cls_id)] = float(pq)
        sq_per_class_dict[int(cls_id)] = float(sq)
        rq_per_class_dict[int(cls_id)] = float(rq)
    
    # Return all metrics
    return {
        "overall_pq": float(overall_pq),
        "overall_sq": float(overall_sq),
        "overall_rq": float(overall_rq),
        "per_class_pq": pq_per_class_dict,
        "per_class_sq": sq_per_class_dict,
        "per_class_rq": rq_per_class_dict
    }

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load class names
    class_names = load_class_names(args.class_names_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize data module
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module = getattr(importlib.import_module(data_module_name), class_name)
    data_module_kwargs = config["data"].get("init_args", {})

    data = data_module(
        path=args.data_path,
        batch_size=args.batch_size,
        check_empty_targets=False,
        **data_module_kwargs
    ).setup()

    # Suppress warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # Multi-GPU setup
    if args.num_gpus > 1:
        if torch.cuda.device_count() < args.num_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available. Using all available GPUs.")
            args.num_gpus = torch.cuda.device_count()
        
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        devices = list(range(args.num_gpus))
    else:
        # Single GPU setup
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        devices = [args.device]
    
    # Determine whether to use a single weight or multiple weights from a file
    if args.weights_file:
        experiments = load_weights_from_file(args.weights_file)
        if not experiments:
            print("No valid experiments found. Exiting.")
            return
    else:
        if not args.weights:
            print("Either --weights or --weights_file must be provided.")
            return
        # Single experiment case
        experiments = [("", args.weights)]

    # Select dataset split
    if args.split == 'train':
        dataset = data.train_dataloader().dataset
    elif args.split == 'val':
        dataset = data.val_dataloader().dataset
    else:  # test
        dataset = data.test_dataloader().dataset
    
    # Limit number of samples if requested
    num_samples = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))
    
    # Create DataLoader for batch processing
    from torch.utils.data import DataLoader, Subset
    
    # Use a subset if num_samples is specified
    if args.num_samples is not None:
        dataset = Subset(dataset, list(range(num_samples)))
    
    # Create DataLoader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: ([item[0] for item in batch], [item[1] for item in batch])
    )
    
    # Save class names list to the output directory for reference
    with open(os.path.join(args.output_dir, "class_names.txt"), "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}\t{name}\n")
    
    # Process images with each experiment's weights
    print(f"Processing {num_samples} images from {args.split} set...")
    
    # Create a full class legend to reference for all experiments (only if explicitly requested)
    if args.with_legend:
        # Create a mapping of all class indices to colors
        all_colors = {}
        for i in range(data.num_classes):
            # Generate colors directly from hsv colormap using normalized position
            all_colors[i] = plt.cm.hsv(i / max(1, data.num_classes))[:3]
        
        # Save a comprehensive legend with all classes
        legend_fig = create_class_legend(
            list(range(data.num_classes)),
            [all_colors[i] for i in range(data.num_classes)],
            class_names
        )
        legend_fig.savefig(os.path.join(args.output_dir, "full_class_legend.png"), 
                          bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close(legend_fig)
    
    # Prepare dictionary to collect instance counts and PQ
    instance_count_dict = {}
    pq_results = {}
    
    # Process each experiment (model weight)
    for i, (exp_name, weights_path) in enumerate(experiments):
        is_first_exp = (i == 0)  # Flag to identify the first experiment
        print(f"Running inference with experiment: {exp_name if exp_name else 'default'}")
        
        # Load model for this experiment
        model = load_model(config, device, weights_path, data)
        if model is None:
            print(f"Skipping experiment {exp_name} due to model loading error")
            continue
        
        # Wrap model with DataParallel or DistributedDataParallel if multiple GPUs
        if args.num_gpus > 1:
            if args.distributed:
                import torch.distributed as dist
                
                # Initialize distributed process group if not already initialized
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl', init_method='env://')
                
                model = torch.nn.parallel.DistributedDataParallel(
                    model, 
                    device_ids=[args.device],  # Use the specific device ID
                    output_device=args.device
                )
                print(f"Using DistributedDataParallel with {args.num_gpus} GPUs")
            else:
                # Make sure the model is on CUDA before wrapping with DataParallel
                model.cuda()
                model = torch.nn.DataParallel(model, device_ids=devices)
                print(f"Using DataParallel with {args.num_gpus} GPUs - devices: {devices}")
                # Ensure batch size is divisible by number of GPUs for optimal use
                if args.batch_size % args.num_gpus != 0:
                    print(f"Warning: Batch size {args.batch_size} is not divisible by number of GPUs {args.num_gpus}.")
                    print(f"For optimal GPU utilization, consider using a batch size that's a multiple of {args.num_gpus}.")
        
        # Process all images with this model
        batch_idx = 0
        for batch_imgs, batch_targets in tqdm(dataloader, desc=f"Experiment: {exp_name if exp_name else 'default'}"):
            # Process a batch of images
            if args.num_gpus > 1:
                # With multiple GPUs, use the batch implementation with DataParallel model
                # This allows DataParallel to distribute work across GPUs
                batch_results = infer_panoptic_batch(model, batch_imgs, batch_targets, device)
            else:
                # With single GPU, use the batch implementation
                batch_results = infer_panoptic_batch(model, batch_imgs, batch_targets, device)
            
            # Process results for each image in the batch
            for batch_item_idx, (sem_pred, inst_pred, sem_target, inst_target, is_crowd) in enumerate(batch_results):
                global_idx = batch_idx * args.batch_size + batch_item_idx
                if global_idx >= num_samples:
                    break
                
                # Get the image
                img = batch_imgs[batch_item_idx]
                
                # Save results and get instance counts
                pred_counts, target_counts = save_panoptic_results(
                    img, sem_pred, inst_pred, sem_target, inst_target, 
                    model.module.num_classes if hasattr(model, 'module') else model.num_classes,
                    f"{args.split}_{global_idx:04d}", 
                    args.save_mode, args.output_dir,
                    with_legend=args.with_legend,  # Only used for separate legend images
                    class_names=class_names,
                    exp_name=exp_name,
                    is_first_exp=is_first_exp
                )
                
                # Add to instance_count_dict with requested key format
                base_name = f"{args.split}_{global_idx:04d}"
                if (target_counts is not None):
                    instance_count_dict[f"{base_name}_target"] = target_counts
                if (pred_counts is not None):
                    exp_key = exp_name if exp_name else "default"
                    instance_count_dict[f"{base_name}_pred_{exp_key}"] = pred_counts
                
                # Calculate per-image PQ for prediction
                if not args.skip_metrics:
                    try:
                        # Get current model (handle DataParallel case)
                        current_model = model.module if hasattr(model, 'module') else model
                        # Get stuff/thing classes
                        stuff_classes = current_model.stuff_classes if hasattr(current_model, 'stuff_classes') else []
                        # Important: Add model.num_classes to stuff_classes like in the codebase init_metrics_panoptic
                        stuff_classes = stuff_classes + [current_model.num_classes]
                        thing_classes = [i for i in range(current_model.num_classes) if i not in stuff_classes]
                        
                        # Stack pred and target tensors
                        pred_tensor = np.stack([sem_pred, inst_pred], axis=-1)
                        target_tensor = np.stack([sem_target, inst_target], axis=-1)
                        
                        # Use the optimized metrics calculation function
                        metrics = calculate_metrics_panoptic(
                            pred_tensor, 
                            target_tensor, 
                            is_crowd,
                            thing_classes, 
                            stuff_classes,
                            cuda=torch.cuda.is_available()
                        )
                        
                        # Convert class IDs to names for output
                        exp_key = exp_name if exp_name else "default"
                        
                        # Store PQ results with class names instead of IDs
                        pq_results[f"{base_name}_pq_pred_{exp_key}"] = {
                            "overall_pq": metrics["overall_pq"],
                            "overall_sq": metrics["overall_sq"],
                            "overall_rq": metrics["overall_rq"],
                            "per_class_pq": {class_names[cls_id]: value 
                                            for cls_id, value in metrics["per_class_pq"].items()
                                            if 0 <= cls_id < len(class_names)},
                            "per_class_sq": {class_names[cls_id]: value 
                                            for cls_id, value in metrics["per_class_sq"].items()
                                            if 0 <= cls_id < len(class_names)},
                            "per_class_rq": {class_names[cls_id]: value 
                                            for cls_id, value in metrics["per_class_rq"].items()
                                            if 0 <= cls_id < len(class_names)}
                        }
                    except Exception as e:
                        print(f"Warning: PQ calculation failed for {base_name} {exp_name}: {e}")
            
            # Increment batch index
            batch_idx += 1
    
    # Save instance counts and PQ to JSON after all images are processed
    json_path = os.path.join(args.output_dir, "instance_counts.json")
    with open(json_path, "w") as jf:
        if args.skip_metrics:
            json.dump({"instance_counts": instance_count_dict}, jf, indent=2)
            print(f"Instance counts saved to {json_path} (metrics calculation skipped)")
        else:
            json.dump({"instance_counts": instance_count_dict, "pq": pq_results}, jf, indent=2)
            print(f"Instance counts and PQ saved to {json_path}")

if __name__ == "__main__":
    main()