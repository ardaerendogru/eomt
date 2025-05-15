import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import PIL.Image as Image
import gradio as gr
import os
import io
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load class names
id_to_name = {}
name_to_id = {}
with open('/home/arda/thesis/eomt/output/class_names.txt', 'r') as f:
    for line in f:
        id, name = line.strip().split('\t')
        id_to_name[int(id)] = name
        name_to_id[name] = int(id)

# Load metrics and instance counts
with open('/home/arda/thesis/eomt/output/instance_counts.json', 'r') as f:
    results = json.load(f)
    
metric_results = results['pq']
instance_counts = results['instance_counts']

# Clean up metric results
metric_results = {k.replace('_pq', ''):v for k,v in metric_results.items()}

# Helper functions
def get_image_name(idx):
    return f'val_{str(idx).zfill(4)}_input'

def get_prediction_name(idx, model_name):
    return f'val_{str(idx).zfill(4)}_pred_{model_name}'

def get_target_name(idx):
    return f'val_{str(idx).zfill(4)}_target'

def get_image_path(idx):
    return f'/home/arda/thesis/eomt/output/{get_image_name(idx)}.png'

def get_prediction_path(idx, model_name):
    return f'/home/arda/thesis/eomt/output/{get_prediction_name(idx, model_name)}.png'

def get_target_path(idx):
    return f'/home/arda/thesis/eomt/output/{get_target_name(idx)}.png'

def get_image(idx):
    return Image.open(get_image_path(idx))

def get_prediction(idx, model_name):
    return Image.open(get_prediction_path(idx, model_name))

def get_target(idx):
    return Image.open(get_target_path(idx))

def get_image_metrics(idx, model_name):
    metrics = {}
    metrics['pq'] = metric_results[get_prediction_name(idx, model_name)]['overall_pq']
    metrics['sq'] = metric_results[get_prediction_name(idx, model_name)]['overall_sq']
    metrics['rq'] = metric_results[get_prediction_name(idx, model_name)]['overall_rq']
    metrics['per_class_pq'] = metric_results[get_prediction_name(idx, model_name)]['per_class_pq']
    metrics['per_class_sq'] = metric_results[get_prediction_name(idx, model_name)]['per_class_sq']
    metrics['per_class_rq'] = metric_results[get_prediction_name(idx, model_name)]['per_class_rq']
    return metrics

def get_images_by_class_names(class_names):
    instance_counts_target = {k:v for k,v in instance_counts.items() if 'target' in k}
    matching_images = []
    
    if isinstance(class_names, str):
        class_names = [class_names]
    
    # For each image
    for img_name, instance_count in instance_counts_target.items():
        # Check if all specified class names are present in this image
        all_classes_present = all(class_name in instance_count.keys() for class_name in class_names)
        
        # If all specified classes are in this image, add it to our matching images
        if all_classes_present:
            # Extract the base image name without the "_target" suffix
            idx = int(img_name.split('_')[1])
            matching_images.append(idx)
    
    return matching_images

def visualize_comparison(idx, model_names):
    # Get per-class metrics
    combined_data = {}
    for model_name in model_names:
        model_data = metric_results[get_prediction_name(idx, model_name)]["per_class_pq"]
        combined_data[model_name] = model_data
    
    # Create DataFrame
    all_classes = set()
    for model_name in model_names:
        all_classes.update(combined_data[model_name].keys())
    all_classes = sorted(list(all_classes))
    
    df = pd.DataFrame(index=all_classes, columns=model_names)
    for model_name in model_names:
        for class_id in all_classes:
            df.loc[class_id, model_name] = combined_data[model_name].get(class_id, 0.0)
    
    # Convert class IDs to names
    df.index = [id_to_name.get(int(idx), idx) for idx in df.index]
    
    # Create figure with all the images
    fig = Figure(figsize=(5 * (len(model_names) + 2), 10))
    canvas = FigureCanvas(fig)
    
    # Add subplots
    axes = fig.subplots(nrows=1, ncols=len(model_names) + 2)
    
    # Input image
    axes[0].imshow(np.array(get_image(idx)))
    axes[0].set_title(f'Input: {idx}')
    axes[0].axis('off')
    
    # Target image
    axes[1].imshow(np.array(get_target(idx)))
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    # Model predictions
    for i, model_name in enumerate(model_names):
        axes[i+2].imshow(np.array(get_prediction(idx, model_name)))
        axes[i+2].set_title(f'{model_name}')
        axes[i+2].axis('off')
    
    fig.tight_layout()
    
    # Convert figure to image
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Calculate overall metrics
    overall_metrics = {}
    for model_name in model_names:
        metrics = get_image_metrics(idx, model_name)
        overall_metrics[model_name] = {
            'PQ': metrics['pq'],
            'SQ': metrics['sq'],
            'RQ': metrics['rq']
        }
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(overall_metrics).T
    
    return buf, df, metrics_df

def app_interface(selected_classes, available_models=['ADE_DINO_WEIGHTS', 'experiment_2_full']):
    if not selected_classes:
        return None, "Please select at least one class.", None, None
    
    # Get image indices with the selected classes
    class_names = [c for c in selected_classes if c]
    image_indices = get_images_by_class_names(class_names)
    
    if not image_indices:
        return None, f"No images found containing all these classes: {', '.join(class_names)}", None, None
    
    # Get first image for visualization
    idx = image_indices[0]
    img_buf, per_class_df, metrics_df = visualize_comparison(idx, available_models)
    
    # Format metrics table
    metrics_html = metrics_df.to_html(float_format=lambda x: f"{x:.4f}")
    
    # Format per-class metrics
    per_class_html = per_class_df.to_html(float_format=lambda x: f"{x:.4f}")
    
    # List of found images
    images_text = f"Found {len(image_indices)} images with the selected classes. Showing index: {idx}."
    available_indices = f"All image indices: {', '.join(map(str, image_indices[:20]))}"
    if len(image_indices) > 20:
        available_indices += f" and {len(image_indices) - 20} more..."
    
    info_text = f"{images_text}\n{available_indices}"
    
    return img_buf, info_text, per_class_html, metrics_html

def get_all_classes():
    # Return a list of all class names from the loaded data
    return sorted(name_to_id.keys())

# Create Gradio UI
with gr.Blocks(title="Model Comparison Visualizer") as demo:
    gr.Markdown("# Model Comparison Visualizer")
    gr.Markdown("Select classes to filter images, then view predictions from different models")
    
    with gr.Row():
        with gr.Column(scale=1):
            class_dropdown = gr.Dropdown(
                choices=get_all_classes(),
                multiselect=True,
                label="Select Classes"
            )
            submit_btn = gr.Button("Visualize Comparison")
            
            info_output = gr.Textbox(
                label="Results Information", 
                placeholder="Information about found images will appear here...",
                lines=3
            )
        
        with gr.Column(scale=3):
            image_output = gr.Image(label="Comparison Visualization", type="pil")
            
    with gr.Row():
        metrics_output = gr.HTML(label="Overall Metrics")
        per_class_metrics = gr.HTML(label="Per-Class Metrics (PQ)")
    
    submit_btn.click(
        fn=app_interface,
        inputs=[class_dropdown],
        outputs=[image_output, info_output, per_class_metrics, metrics_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 