import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import PIL.Image as Image
import gradio as gr
import os
import io
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import colorsys

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

# Get available models
available_models = []
for key in metric_results.keys():
    if '_pred_' in key:
        model_name = key.split('_pred_')[1]
        if model_name not in available_models:
            available_models.append(model_name)

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
    
    # Convert class names to IDs if they're in our mapping
    class_ids = []
    for class_name in class_names:
        if class_name in name_to_id:
            # If it's a known class name, add the ID
            class_ids.append(str(name_to_id[class_name]))
        else:
            # Otherwise keep the original string
            class_ids.append(class_name)
    
    # For each image
    for img_name, instance_count in instance_counts_target.items():
        # Get all possible keys (class ids and names)
        image_classes = set(instance_count.keys())
        
        # Check if all specified classes are present in this image
        all_classes_present = True
        for class_id in class_ids:
            # Check if the class ID is directly in the keys
            if class_id in image_classes:
                continue
            
            # Check if it's a name that maps to an ID that is in the keys
            # Or if it's an ID that maps to a name that is in the keys
            class_found = False
            
            # Try interpreting as ID → name
            try:
                if int(class_id) in id_to_name:
                    if id_to_name[int(class_id)] in image_classes:
                        class_found = True
            except (ValueError, KeyError):
                pass
            
            # Try interpreting as name → ID
            if class_id in name_to_id:
                if str(name_to_id[class_id]) in image_classes:
                    class_found = True
            
            # If the class wasn't found by any method, this image doesn't match
            if not class_found:
                all_classes_present = False
                break
                
        # If all specified classes are in this image, add it to our matching images
        if all_classes_present:
            # Extract the base image name without the "_target" suffix
            idx = int(img_name.split('_')[1])
            matching_images.append(idx)
    
    return matching_images

def get_color_for_class(class_id, total_classes=len(id_to_name)):
    """Generate a color for a class based on its ID"""
    try:
        # If class_id is in the name_to_id dictionary, use that
        if class_id in name_to_id:
            class_index = name_to_id[class_id]
        # If class_id is already a number (string or int), use that
        elif isinstance(class_id, (int, float)) or (isinstance(class_id, str) and class_id.isdigit()):
            class_index = int(class_id)
        else:
            # Otherwise use hash to get a consistent color
            class_index = hash(str(class_id)) % 10000
    except Exception:
        # Fallback to a hash-based approach for any errors
        class_index = hash(str(class_id)) % 10000
    
    hue = (class_index % total_classes) / total_classes
    saturation = 0.8
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b)

def get_class_instances_in_image(idx):
    """Get the classes and their instance counts in a specific image"""
    target_name = get_target_name(idx)
    if target_name in instance_counts:
        image_instances = instance_counts[target_name]
        # Convert class IDs to names
        named_instances = {}
        for class_id, count in image_instances.items():
            class_name = class_id
            named_instances[class_name] = count
        return named_instances
    return {}

def create_color_legend(class_ids):
    return None

def calculate_model_differences(idx, model_names):
    """Calculate the differences between models' metrics"""
    if len(model_names) < 2:
        return None
    
    diffs = {}
    model_metrics = {}
    
    # Get metrics for all models
    for model_name in model_names:
        model_metrics[model_name] = get_image_metrics(idx, model_name)
    
    # Calculate differences for all pairs of models
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            key = f"{model1} vs {model2}"
            diffs[key] = {
                'PQ_diff': model_metrics[model1]['pq'] - model_metrics[model2]['pq'],
                'SQ_diff': model_metrics[model1]['sq'] - model_metrics[model2]['sq'],
                'RQ_diff': model_metrics[model1]['rq'] - model_metrics[model2]['rq']
            }
            
            # Per-class differences
            per_class_diffs = {}
            for class_id in set(model_metrics[model1]['per_class_pq'].keys()) | set(model_metrics[model2]['per_class_pq'].keys()):
                pq1 = model_metrics[model1]['per_class_pq'].get(class_id, 0)
                pq2 = model_metrics[model2]['per_class_pq'].get(class_id, 0)
                per_class_diffs[class_id] = pq1 - pq2
            
            diffs[key]['per_class_diffs'] = per_class_diffs
    
    return diffs

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
    df.index = [class_id for class_id in df.index]
    
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
    
    # Convert BytesIO to PIL Image
    pil_img = Image.open(buf)
    
    # Also prepare individual images for zooming
    input_img = get_image(idx)
    target_img = get_target(idx)
    model_images = []
    
    # Add model predictions with labels
    for model_name in model_names:
        pred_img = get_prediction(idx, model_name)
        metrics = get_image_metrics(idx, model_name)
        model_images.append((pred_img, f"{model_name} (PQ: {metrics['pq']:.4f})"))
    
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
    
    # Get instance counts for the image
    instance_counts_df = pd.DataFrame(get_class_instances_in_image(idx).items(), 
                                      columns=['Class', 'Instances'])
    instance_counts_df = instance_counts_df.sort_values('Instances', ascending=False)
    
    # Create color legend
    color_legend = create_color_legend(all_classes)
    
    # Calculate metrics differences between models
    model_diffs = calculate_model_differences(idx, model_names)
    if model_diffs:
        diff_dfs = {}
        for key, diff in model_diffs.items():
            diff_df = pd.DataFrame({
                'Metric': ['PQ', 'SQ', 'RQ'],
                'Difference': [diff['PQ_diff'], diff['SQ_diff'], diff['RQ_diff']]
            })
            diff_dfs[key] = diff_df
    else:
        diff_dfs = None
    
    return pil_img, df, metrics_df, instance_counts_df, color_legend, diff_dfs, input_img, target_img, model_images

def app_interface(selected_classes, selected_models, image_index=0):
    if not selected_classes:
        return None, "Please select at least one class.", None, None, None, None, None, None, None, None, 0, []
    
    if not selected_models or len(selected_models) < 1:
        return None, "Please select at least one model for comparison.", None, None, None, None, None, None, None, None, 0, []
    
    # Get image indices with the selected classes
    class_names = [c for c in selected_classes if c]
    image_indices = get_images_by_class_names(class_names)
    
    if not image_indices:
        return None, f"No images found containing all these classes: {', '.join(class_names)}", None, None, None, None, None, None, None, None, 0, []
    
    # Sort image indices for consistent navigation
    image_indices.sort()
    
    # Ensure image_index is within bounds
    if image_index < 0 or image_index >= len(image_indices):
        image_index = 0
    
    # Get the actual image index from our list
    idx = image_indices[image_index]
    
    # Visualize the comparison
    img_buf, per_class_df, metrics_df, instance_counts_df, color_legend, diff_dfs, input_img, target_img, model_images = visualize_comparison(idx, selected_models)
    
    # Format metrics tables
    metrics_html = metrics_df.to_html(float_format=lambda x: f"{x:.4f}")
    per_class_html = per_class_df.to_html(float_format=lambda x: f"{x:.4f}")
    instance_counts_html = instance_counts_df.to_html(index=False)
    
    # Format model differences
    if diff_dfs:
        diff_html = "<h3>Model Comparison Differences</h3>"
        for key, df in diff_dfs.items():
            diff_html += f"<h4>{key}</h4>"
            # Color positive differences in green, negative in red
            styled_df = df.style.applymap(
                lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '',
                subset=['Difference']
            )
            diff_html += styled_df.to_html(float_format=lambda x: f"{x:.4f}")
    else:
        diff_html = "<p>Select multiple models to see differences</p>"
    
    # List of found images
    images_text = f"Found {len(image_indices)} images with the selected classes. Showing image {image_index+1}/{len(image_indices)} (index: {idx})."
    available_indices = f"All image indices: {', '.join(map(str, image_indices[:20]))}"
    if len(image_indices) > 20:
        available_indices += f" and {len(image_indices) - 20} more..."
    
    info_text = f"{images_text}\n{available_indices}"
    
    return img_buf, info_text, per_class_html, metrics_html, instance_counts_html, color_legend, diff_html, input_img, target_img, model_images, image_index, image_indices

def next_image(selected_classes, selected_models, current_index, image_indices):
    if not image_indices:
        return app_interface(selected_classes, selected_models, 0)
        
    next_index = (current_index + 1) % len(image_indices)
    return app_interface(selected_classes, selected_models, next_index)

def prev_image(selected_classes, selected_models, current_index, image_indices):
    if not image_indices:
        return app_interface(selected_classes, selected_models, 0)
        
    prev_index = (current_index - 1) % len(image_indices)
    return app_interface(selected_classes, selected_models, prev_index)

def jump_to_image(selected_classes, selected_models, target_idx, current_index, image_indices):
    if not target_idx or not image_indices:
        return app_interface(selected_classes, selected_models, current_index)
    
    # Try to find the index in our image_indices list
    try:
        target_idx = int(target_idx)
        if target_idx in image_indices:
            idx = image_indices.index(target_idx)
            return app_interface(selected_classes, selected_models, idx)
    except:
        pass
    
    # If we can't find it, keep current index
    return app_interface(selected_classes, selected_models, current_index)

def get_all_classes():
    # Return a list of all class names from the loaded data
    return sorted(name_to_id.keys())

# Create Gradio UI
with gr.Blocks(title="Model Comparison Visualizer", css="""
    .zoom-container img {
        transition: transform 0.3s ease;
        cursor: zoom-in;
    }
    .zoom-container img:hover {
        transform: scale(1.05);
    }
    .zoom-container img:active {
        transform: scale(1.5);
        cursor: zoom-out;
    }
    #gallery {
        min-height: 300px;
    }
    .metrics-container {
        max-height: 600px;
        overflow-y: auto;
    }
""") as demo:
    gr.Markdown("# Model Comparison Visualizer")
    gr.Markdown("Select classes to filter images, then compare predictions from different models")
    
    # State variables (not visible in UI)
    current_image_index = gr.State(0)
    image_indices = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            class_dropdown = gr.Dropdown(
                choices=get_all_classes(),
                multiselect=True,
                label="Select Classes"
            )
            
            model_dropdown = gr.Dropdown(
                choices=available_models,
                multiselect=True,
                value=['ADE_DINO_WEIGHTS', 'experiment_2_full'],
                label="Select Models to Compare"
            )
            
            submit_btn = gr.Button("Find Images & Visualize", variant="primary")
            
            info_output = gr.Textbox(
                label="Results Information", 
                placeholder="Information about found images will appear here...",
                lines=3
            )
            
            with gr.Row():
                prev_btn = gr.Button("← Previous Image")
                next_btn = gr.Button("Next Image →")
            
            with gr.Row():
                image_idx_input = gr.Textbox(
                    label="Jump to Image Index",
                    placeholder="Enter image index (e.g., 42)",
                )
                jump_btn = gr.Button("Go")
            
            color_legend = gr.Image(label="Class Color Legend", type="pil")
        
        with gr.Column(scale=3):
            # Overview comparison visualization
            image_output = gr.Image(
                label="Comparison Visualization", 
                type="pil", 
                elem_classes="zoom-container"
            )
            instance_counts_output = gr.HTML(label="Classes in Current Image")
            
    # Add tabbed interface for zooming individual images
    with gr.Tabs() as tabs:
        with gr.TabItem("Metrics"):
            with gr.Column(elem_classes="metrics-container"):
                with gr.Row():
                    metrics_output = gr.HTML(label="Overall Metrics")
                    per_class_metrics = gr.HTML(label="Per-Class Metrics (PQ)")
                model_diff_output = gr.HTML(label="Model Comparison")
        
        with gr.TabItem("Zoom Individual Images"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Image (Click to zoom)")
                    input_image = gr.Image(
                        label="", 
                        show_download_button=True, 
                        interactive=True,
                        elem_classes="zoom-container"
                    )
                with gr.Column():
                    gr.Markdown("### Target Segmentation (Click to zoom)")
                    target_image = gr.Image(
                        label="", 
                        show_download_button=True, 
                        interactive=True,
                        elem_classes="zoom-container"
                    )
            
            gr.Markdown("### Model Predictions (Click images to enlarge)")
            model_images_gallery = gr.Gallery(
                label="",  
                show_label=False, 
                elem_id="gallery", 
                columns=2, 
                height="auto",
                allow_preview=True,
                preview=True
            )
            
            gr.Markdown("""
            **Tips for zooming:**
            - Hover over any image to slightly enlarge it
            - Click and hold to zoom in more
            - In the gallery, click an image to open it in full screen mode
            - Use mousewheel to zoom in/out in full screen mode
            """)
            
    # Define interactions
    submit_btn.click(
        fn=app_interface,
        inputs=[class_dropdown, model_dropdown],
        outputs=[image_output, info_output, per_class_metrics, metrics_output, 
                 instance_counts_output, color_legend, model_diff_output,
                 input_image, target_image, model_images_gallery,
                 current_image_index, image_indices]
    )
    
    next_btn.click(
        fn=next_image,
        inputs=[class_dropdown, model_dropdown, current_image_index, image_indices],
        outputs=[image_output, info_output, per_class_metrics, metrics_output, 
                 instance_counts_output, color_legend, model_diff_output,
                 input_image, target_image, model_images_gallery,
                 current_image_index, image_indices]
    )
    
    prev_btn.click(
        fn=prev_image,
        inputs=[class_dropdown, model_dropdown, current_image_index, image_indices],
        outputs=[image_output, info_output, per_class_metrics, metrics_output, 
                 instance_counts_output, color_legend, model_diff_output,
                 input_image, target_image, model_images_gallery,
                 current_image_index, image_indices]
    )
    
    jump_btn.click(
        fn=jump_to_image,
        inputs=[class_dropdown, model_dropdown, image_idx_input, current_image_index, image_indices],
        outputs=[image_output, info_output, per_class_metrics, metrics_output, 
                 instance_counts_output, color_legend, model_diff_output,
                 input_image, target_image, model_images_gallery,
                 current_image_index, image_indices]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 