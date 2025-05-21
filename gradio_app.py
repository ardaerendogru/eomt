import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import PIL.Image
import gradio as gr
import os
import io
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import colorsys
import plotly.express as px
import plotly.graph_objects as go

# --- Constants and Configuration ---
CLASS_NAMES_PATH = '/home/arda/thesis/eomt/output/class_names.txt'
INSTANCE_COUNTS_PATH = '/home/arda/thesis/eomt/output/instance_counts.json'
OUTPUT_DIR = '/home/arda/thesis/eomt/output/'

# --- Data Loading ---
def load_class_mapping():
    id_to_name_map = {}
    name_to_id_map = {}
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            for line in f:
                id_val, name_val = line.strip().split('\t')
                id_to_name_map[int(id_val)] = name_val
                name_to_id_map[name_val] = int(id_val)
    except Exception as e:
        print(f"Error loading class names: {e}")
    return id_to_name_map, name_to_id_map

def load_metrics_and_counts():
    try:
        with open(INSTANCE_COUNTS_PATH, 'r') as f:
            results = json.load(f)
        metric_results_raw = results.get('pq', {})
        instance_counts_raw = results.get('instance_counts', {})
        
        # Clean up metric results keys
        metric_results_cleaned = {k.replace('_pq', ''): v for k, v in metric_results_raw.items()}
        return metric_results_cleaned, instance_counts_raw
    except Exception as e:
        print(f"Error loading metrics and instance counts: {e}")
        return {}, {}

ID_TO_NAME, NAME_TO_ID = load_class_mapping()
METRIC_RESULTS, INSTANCE_COUNTS = load_metrics_and_counts()

def get_available_models():
    models = []
    for key in METRIC_RESULTS.keys():
        if '_pred_' in key:
            model_name = key.split('_pred_')[1]
            if model_name not in models:
                models.append(model_name)
    return sorted(list(set(models)))

AVAILABLE_MODELS = get_available_models()
ALL_CLASS_NAMES = sorted(list(set(ID_TO_NAME.values())))

# --- Helper Functions for File Paths and Data Retrieval ---
def get_image_name_template(idx, type="input", model_name=None):
    prefix = f'val_{str(idx).zfill(4)}'
    if type == "input":
        return f'{prefix}_input'
    elif type == "target":
        return f'{prefix}_target'
    elif type == "pred" and model_name:
        return f'{prefix}_pred_{model_name}'
    return None

def get_image_path(idx, type="input", model_name=None):
    img_name = get_image_name_template(idx, type, model_name)
    return os.path.join(OUTPUT_DIR, f'{img_name}.png') if img_name else None

def load_pil_image(path):
    try:
        return PIL.Image.open(path) if path and os.path.exists(path) else PIL.Image.new('RGB', (100,100), color='grey') # Placeholder for missing
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return PIL.Image.new('RGB', (100,100), color='lightgrey') # Placeholder

def get_image_metrics_for_model(idx, model_name):
    pred_name = get_image_name_template(idx, "pred", model_name)
    if pred_name in METRIC_RESULTS:
        data = METRIC_RESULTS[pred_name]
        return {
            'pq': data.get('overall_pq', 0.0),
            'sq': data.get('overall_sq', 0.0),
            'rq': data.get('overall_rq', 0.0),
            'per_class_pq': data.get('per_class_pq', {}),
            'per_class_sq': data.get('per_class_sq', {}),
            'per_class_rq': data.get('per_class_rq', {}),
        }
    return {'pq': 0, 'sq': 0, 'rq': 0, 'per_class_pq': {}, 'per_class_sq': {}, 'per_class_rq': {}}

def get_instance_counts_for_image(idx):
    target_name = get_image_name_template(idx, "target")
    if target_name in INSTANCE_COUNTS:
        # Convert class IDs to names if possible
        named_instances = {}
        for class_id_str, count in INSTANCE_COUNTS[target_name].items():
            try:
                class_name = ID_TO_NAME.get(int(class_id_str), str(class_id_str))
            except ValueError:
                class_name = str(class_id_str) # Use original if not an int ID
            named_instances[class_name] = count
        return named_instances
    return {}

# --- New Helper for Overall Per-Class PQ Analysis ---
def get_all_image_indices():
    """Extracts all unique image indices from the INSTANCE_COUNTS keys.
       Assumes keys are like 'val_XXXX_target'.
    """
    indices = set()
    for key in INSTANCE_COUNTS.keys():
        if '_target' in key: # Consider only target entries to define the dataset images
            try:
                # Extract the numeric part, e.g., from 'val_0023_target'
                idx_str = key.split('_')[1]
                indices.add(int(idx_str))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse image index from key: {key} - {e}")
    return sorted(list(indices))

def get_average_per_class_pq_for_models(selected_models_list):
    """Calculates the average Per-Class PQ for selected models across all images.
       Averages PQ for a class only over images where that class is predicted by the model.
    """
    if not selected_models_list:
        return pd.DataFrame()

    all_image_indices = get_all_image_indices()
    if not all_image_indices:
        print("Warning: No image indices found to calculate average per-class PQ.")
        return pd.DataFrame()

    class_pq_sums = {model: {} for model in selected_models_list}
    class_appearances = {model: {} for model in selected_models_list}
    all_class_ids_overall = set()

    for model_name in selected_models_list:
        for img_idx in all_image_indices:
            # pred_name = get_image_name_template(img_idx, "pred", model_name)
            # model_metrics = METRIC_RESULTS.get(pred_name, {})
            # per_class_pq_data = model_metrics.get('per_class_pq', {})
            # Using the existing helper function is better as it handles missing keys gracefully
            model_image_metrics = get_image_metrics_for_model(img_idx, model_name)
            per_class_pq_data = model_image_metrics.get('per_class_pq', {}) # This is a dict like {'class_id_str': pq_float}

            for class_id_str, pq_score in per_class_pq_data.items():
                # Ensure class_id_str is consistently handled (e.g. as string if keys in ID_TO_NAME are ints)
                # However, keys from get_image_metrics_for_model['per_class_pq'] should be strings as per METRIC_RESULTS loading
                
                current_sum = class_pq_sums[model_name].get(class_id_str, 0.0)
                class_pq_sums[model_name][class_id_str] = current_sum + pq_score
                
                current_appearances = class_appearances[model_name].get(class_id_str, 0)
                class_appearances[model_name][class_id_str] = current_appearances + 1
                
                all_class_ids_overall.add(class_id_str)

    if not all_class_ids_overall:
        print("Warning: No class PQ data found across any selected model or image.")
        return pd.DataFrame()

    # Prepare data for DataFrame: index=class_names, columns=model_names
    # Sort class IDs numerically if they are digits, otherwise alphabetically for consistent column order
    sorted_class_ids = sorted(list(all_class_ids_overall), key=lambda x: int(x) if x.isdigit() else x)
    class_names_index = [(ID_TO_NAME.get(int(cid), str(cid)) if cid.isdigit() else str(cid)).split(',')[0].strip() for cid in sorted_class_ids]
    
    df_data = {model: [] for model in selected_models_list}

    for class_id_str in sorted_class_ids:
        for model_name in selected_models_list:
            total_pq = class_pq_sums[model_name].get(class_id_str, 0.0)
            num_appearances = class_appearances[model_name].get(class_id_str, 0)
            avg_pq = (total_pq / num_appearances) if num_appearances > 0 else 0.0
            df_data[model_name].append(avg_pq)
            
    avg_pq_df = pd.DataFrame(df_data, index=class_names_index)
    avg_pq_df.index.name = 'Class' # Ensure the index has a name
    return avg_pq_df

def generate_interactive_per_class_pq_plot(pq_dataframe):
    """Generates an interactive Plotly bar plot from the aggregated per-class PQ DataFrame."""
    if pq_dataframe.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display. Adjust filters or ensure models are selected and have PQ data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis_visible=False, yaxis_visible=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    # Melt dataframe for Plotly
    # Since pq_dataframe.index.name is 'Class' (set in get_average_per_class_pq_for_models),
    # reset_index() will create a 'Class' column.
    df_melted = pq_dataframe.reset_index().melt(
        id_vars='Class', # Directly use 'Class'
        var_name='Model', 
        value_name='Average PQ'
    )

    num_classes = pq_dataframe.shape[0]
    num_models = pq_dataframe.shape[1]
    
    # Approximate height calculation
    plot_height = max(400, num_classes * num_models * 15 + 150) # Base height + per class/model factor

    fig = px.bar(
        df_melted,
        y='Class',
        x='Average PQ',
        color='Model',
        orientation='h',
        barmode='group',
        title='Average Per-Class Panoptic Quality (PQ) Comparison',
        height=plot_height,
        text='Average PQ' # Add values as text on bars
    )

    fig.update_layout(
        xaxis_title='Average Panoptic Quality (PQ)',
        yaxis_title='Class',
        legend_title_text='Models',
        yaxis={'categoryorder':'total ascending'} # Sort classes by total PQ or alphabetically if preferred
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0)) # Adjust margins if needed

    return fig

# --- Core Logic: Filtering ---
def get_filtered_image_indices(selected_class_names, min_distinct, max_distinct, min_total, max_total,
                               pq_diff_model1, pq_diff_model2, pq_diff_top_n):
    
    instance_counts_target = {k: v for k, v in INSTANCE_COUNTS.items() if 'target' in k}
    candidate_indices = []

    for img_name_key, counts_data in instance_counts_target.items():
        img_idx = int(img_name_key.split('_')[1])
        
        current_distinct_classes = len(counts_data)
        current_total_instances = sum(counts_data.values())

        if min_distinct is not None and min_distinct != 0 and current_distinct_classes < min_distinct: continue
        if max_distinct is not None and max_distinct != 0 and current_distinct_classes > max_distinct: continue
        if min_total is not None and min_total != 0 and current_total_instances < min_total: continue
        if max_total is not None and max_total != 0 and current_total_instances > max_total: continue
        
        if selected_class_names: # Only apply class name filter if names are provided
            # Get the class names directly from the image's count data.
            # This assumes counts_data.keys() ARE the class names from the JSON.
            image_actual_class_names = set(counts_data.keys()) 
            
            all_selected_present = True
            for sel_name in selected_class_names: # sel_name is a name from the dropdown
                # Direct check: is the selected name present in the image's class names?
                if sel_name in image_actual_class_names:
                    continue # This selected name is found, check next selected name

                # Fallback: If the selected name has an ID, and that ID (converted back to a name via ID_TO_NAME)
                # is present in the image's class names.
                # This handles cases where dropdown names might be canonical and JSON names are variants,
                # but they map to the same underlying ID.
                if sel_name in NAME_TO_ID:
                    id_of_selected_name = NAME_TO_ID[sel_name]
                    # Ensure the ID is an int for ID_TO_NAME lookup if NAME_TO_ID stores int IDs
                    name_from_id_of_selected = ID_TO_NAME.get(int(id_of_selected_name)) 
                    if name_from_id_of_selected and name_from_id_of_selected in image_actual_class_names:
                        continue # This selected name (via its ID and then back to a canonical name) is found

                # If neither direct name match nor match via ID-to-name conversion worked for this sel_name
                all_selected_present = False
                break # No need to check other selected_class_names for this image
            
            if not all_selected_present: # If any of the selected classes was not found
                continue # Skip this image
        
        candidate_indices.append(img_idx)

    if not candidate_indices:
        return []

    # PQ difference filtering (if applicable and models are distinct)
    if pq_diff_model1 and pq_diff_model2 and pq_diff_model1 != pq_diff_model2 and pq_diff_top_n > 0:
        pq_differences = []
        for img_idx in candidate_indices:
            metrics_m1 = get_image_metrics_for_model(img_idx, pq_diff_model1)
            metrics_m2 = get_image_metrics_for_model(img_idx, pq_diff_model2)
            # Using absolute difference for sorting by "largest difference"
            pq_diff = abs(metrics_m1['pq'] - metrics_m2['pq'])
            pq_differences.append({'idx': img_idx, 'pq_diff': pq_diff})
        
        pq_differences.sort(key=lambda x: x['pq_diff'], reverse=True)
        candidate_indices = [item['idx'] for item in pq_differences[:pq_diff_top_n]]

    return sorted(list(set(candidate_indices)))


# --- Core Logic: Visualization and Data Preparation ---
def get_color_for_class(class_id_or_name, total_classes=len(ID_TO_NAME)):
    class_index = 0
    if isinstance(class_id_or_name, str):
        if class_id_or_name in NAME_TO_ID: # It's a name
            class_index = NAME_TO_ID[class_id_or_name]
        elif class_id_or_name.isdigit() and int(class_id_or_name) in ID_TO_NAME: # It's an ID string
             class_index = int(class_id_or_name)
        else: # Fallback for unknown names or non-ID strings
            class_index = hash(class_id_or_name) % total_classes 
    elif isinstance(class_id_or_name, int): # It's an int ID
        class_index = class_id_or_name
    else: # Fallback for other types
        class_index = hash(str(class_id_or_name)) % total_classes

    hue = (class_index % total_classes) / total_classes if total_classes > 0 else 0
    saturation = 0.85
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b)

def create_color_legend_image(class_names_or_ids_in_plot):
    if not class_names_or_ids_in_plot:
        return PIL.Image.new('RGB', (100, 50), color='lightgrey') # Placeholder

    unique_display_names = []
    processed_for_legend = set()

    for item in class_names_or_ids_in_plot:
        name_to_display = item # Default to item itself
        id_for_color = item

        if isinstance(item, str):
            if item.isdigit() and int(item) in ID_TO_NAME: # Is an ID string
                name_to_display = ID_TO_NAME[int(item)]
                id_for_color = int(item)
            elif item in NAME_TO_ID: # Is a name string
                name_to_display = item
                id_for_color = NAME_TO_ID[item]
            # else, it's a name not in NAME_TO_ID or some other string, use as is for display and color hashing
        elif isinstance(item, int) and item in ID_TO_NAME: # Is an int ID
            name_to_display = ID_TO_NAME[item]
            id_for_color = item
        
        if name_to_display not in processed_for_legend:
            unique_display_names.append({'display': name_to_display, 'color_ref': id_for_color})
            processed_for_legend.add(name_to_display)

    if not unique_display_names:
         return PIL.Image.new('RGB', (100, 50), color='lightgrey')

    num_items = len(unique_display_names)
    fig_height = max(1, num_items * 0.4)
    fig = Figure(figsize=(3, fig_height), dpi=100)
    ax = fig.subplots()
    
    patches = [mpatches.Patch(color=get_color_for_class(item['color_ref']), label=item['display']) for item in unique_display_names]
    
    ax.legend(handles=patches, loc='center', frameon=False, fontsize='small')
    ax.axis('off')
    fig.tight_layout(pad=0.1)

    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def calculate_metric_differences(idx, model_names_list):
    if len(model_names_list) < 2: return None
    
    model_metrics_cache = {name: get_image_metrics_for_model(idx, name) for name in model_names_list}
    differences_data = {}

    for i, model1_name in enumerate(model_names_list):
        for model2_name in model_names_list[i+1:]:
            key = f"{model1_name} vs {model2_name}"
            m1_metrics = model_metrics_cache[model1_name]
            m2_metrics = model_metrics_cache[model2_name]
            
            diffs = {
                'PQ Diff.': m1_metrics['pq'] - m2_metrics['pq'],
                'SQ Diff.': m1_metrics['sq'] - m2_metrics['sq'],
                'RQ Diff.': m1_metrics['rq'] - m2_metrics['rq']
            }
            
            # Per-class PQ differences
            per_class_pq_diffs = {}
            all_class_ids_m1 = set(m1_metrics['per_class_pq'].keys())
            all_class_ids_m2 = set(m2_metrics['per_class_pq'].keys())
            
            for class_id_str in all_class_ids_m1.union(all_class_ids_m2):
                pq1 = m1_metrics['per_class_pq'].get(class_id_str, 0.0)
                pq2 = m2_metrics['per_class_pq'].get(class_id_str, 0.0)
                class_display_name = ID_TO_NAME.get(int(class_id_str), str(class_id_str)) if class_id_str.isdigit() else str(class_id_str)
                per_class_pq_diffs[class_display_name] = pq1 - pq2
            
            differences_data[key] = {'overall': diffs, 'per_class_pq': per_class_pq_diffs}
            
    return differences_data

def generate_image_outputs(image_idx, selected_models_list):
    input_pil = load_pil_image(get_image_path(image_idx, "input"))
    target_pil = load_pil_image(get_image_path(image_idx, "target"))
    
    model_pils_with_labels = []
    for model_name in selected_models_list:
        pred_pil = load_pil_image(get_image_path(image_idx, "pred", model_name))
        metrics = get_image_metrics_for_model(image_idx, model_name)
        model_pils_with_labels.append(
            (pred_pil, f"{model_name} (PQ: {metrics['pq']:.3f})")
        )

    # Create composite overview image
    num_images_total = 2 + len(selected_models_list)
    fig_width = 4 * num_images_total
    fig_height = 4 
    
    overview_fig = Figure(figsize=(fig_width, fig_height), dpi=100)
    axes = overview_fig.subplots(nrows=1, ncols=num_images_total)
    if num_images_total == 1: axes = [axes] # Ensure axes is always a list

    axes[0].imshow(np.array(input_pil))
    axes[0].set_title(f'Input: {image_idx}')
    axes[0].axis('off')
    
    axes[1].imshow(np.array(target_pil))
    axes[1].set_title('Target')
    axes[1].axis('off')

    for i, model_name in enumerate(selected_models_list):
        pred_img_for_overview, _ = model_pils_with_labels[i]
        axes[i+2].imshow(np.array(pred_img_for_overview))
        axes[i+2].set_title(model_name)
        axes[i+2].axis('off')
    
    overview_fig.tight_layout()
    buf = io.BytesIO()
    FigureCanvas(overview_fig).print_png(buf)
    buf.seek(0)
    overview_pil = PIL.Image.open(buf)

    # Overall metrics
    overall_metrics_data = {
        model: {
            'PQ': get_image_metrics_for_model(image_idx, model)['pq'],
            'SQ': get_image_metrics_for_model(image_idx, model)['sq'],
            'RQ': get_image_metrics_for_model(image_idx, model)['rq']
        } for model in selected_models_list
    }
    overall_metrics_df = pd.DataFrame(overall_metrics_data).T.reset_index().rename(columns={'index':'Model'})
    
    # Per-class PQ metrics
    per_class_pq_data = {}
    all_involved_class_ids = set()
    for model_name in selected_models_list:
        pc_pq = get_image_metrics_for_model(image_idx, model_name)['per_class_pq']
        per_class_pq_data[model_name] = pc_pq
        all_involved_class_ids.update(pc_pq.keys())
    
    df_index_names = [ID_TO_NAME.get(int(cid), str(cid)) if cid.isdigit() else str(cid) for cid in sorted(list(all_involved_class_ids), key=lambda x: (ID_TO_NAME.get(int(x), x) if x.isdigit() else x))]
    per_class_pq_df = pd.DataFrame(index=df_index_names, columns=selected_models_list, dtype=float)

    for model_name in selected_models_list:
        for class_id_str, pq_val in per_class_pq_data[model_name].items():
            class_display_name = ID_TO_NAME.get(int(class_id_str), str(class_id_str)) if class_id_str.isdigit() else str(class_id_str)
            if class_display_name in per_class_pq_df.index:
                 per_class_pq_df.loc[class_display_name, model_name] = pq_val
    per_class_pq_df = per_class_pq_df.fillna(0.0).reset_index().rename(columns={'index':'Class'})


    # Instance counts in current image
    instance_counts_current_img = get_instance_counts_for_image(image_idx)
    instance_counts_df = pd.DataFrame(instance_counts_current_img.items(), columns=['Class', 'Instances']).sort_values(by='Instances', ascending=False)

    # Color legend (based on classes in the per_class_pq_df)
    classes_for_legend = per_class_pq_df['Class'].unique().tolist()
    color_legend_pil = create_color_legend_image(classes_for_legend)
    
    # Metric differences
    metric_diffs_data = calculate_metric_differences(image_idx, selected_models_list)
    diff_html_parts = []
    if metric_diffs_data:
        for comparison_key, diff_values in metric_diffs_data.items():
            diff_html_parts.append(f"<h4>{comparison_key}</h4>")
            # Overall diffs
            overall_df = pd.DataFrame([diff_values['overall']]).T.reset_index()
            overall_df.columns = ['Metric', 'Difference']
            styled_overall_df = overall_df.style.apply(
                lambda row: ['color: green' if row['Difference'] > 0.0001 else 'color: red' if row['Difference'] < -0.0001 else '' for _ in row], axis=1, subset=['Difference']
            ).format({'Difference': "{:.4f}"})
            diff_html_parts.append("<h5>Overall Differences:</h5>" + styled_overall_df.to_html(index=False, escape=False))

            # Per-class PQ diffs
            if diff_values['per_class_pq']:
                per_class_df = pd.DataFrame(diff_values['per_class_pq'].items(), columns=['Class', 'PQ Difference'])
                per_class_df = per_class_df.sort_values(by='Class')
                styled_per_class_df = per_class_df.style.apply(
                     lambda row: ['color: green' if row['PQ Difference'] > 0.0001 else 'color: red' if row['PQ Difference'] < -0.0001 else '' for _ in row], axis=1, subset=['PQ Difference']
                ).format({'PQ Difference': "{:.4f}"})
                diff_html_parts.append("<h5>Per-Class PQ Differences:</h5>" + styled_per_class_df.to_html(index=False, escape=False))
    
    metric_diffs_html = "".join(diff_html_parts) if diff_html_parts else "<p>Select at least two models to see detailed comparisons.</p>"

    return (overview_pil, 
            overall_metrics_df, per_class_pq_df, instance_counts_df, 
            color_legend_pil, metric_diffs_html,
            input_pil, target_pil, model_pils_with_labels)

# --- Gradio UI Event Handlers ---
def handle_generate_interactive_per_class_pq_plot(
    selected_models_list, 
    enable_diff_filter, diff_percentage, diff_mode,
    enable_thresh_filter, thresh_value, thresh_mode
    ):
    """Handler for generating and displaying the interactive per-class PQ bar plot with filters."""
    if not selected_models_list:
        fig = go.Figure()
        fig.add_annotation(
            text="Please select at least one model to generate the plot.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
        )
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig
        
    avg_pq_df = get_average_per_class_pq_for_models(selected_models_list)

    if avg_pq_df.empty:
        return generate_interactive_per_class_pq_plot(avg_pq_df) # Let plotting function handle "No data"

    # Determine models for difference filter from selected_models_list
    diff_model1_for_filter = None
    diff_model2_for_filter = None
    if enable_diff_filter and selected_models_list and len(selected_models_list) >= 2:
        diff_model1_for_filter = selected_models_list[0]
        diff_model2_for_filter = selected_models_list[1]

    # Apply Difference Filter
    if enable_diff_filter and diff_model1_for_filter and diff_model2_for_filter and \
       diff_model1_for_filter != diff_model2_for_filter and \
       diff_model1_for_filter in avg_pq_df.columns and diff_model2_for_filter in avg_pq_df.columns:
        
        m1_pq = avg_pq_df[diff_model1_for_filter]
        m2_pq = avg_pq_df[diff_model2_for_filter]
        percentage_val = diff_percentage / 100.0

        if diff_mode == "Model 1 is X% better than Model 2":
            # PQ1 > PQ2 * (1 + X/100)
            # Handles PQ2 = 0: if PQ2 is 0, PQ1 must be > 0 for the condition to be met if X >= 0.
            # If PQ1 = 0 and PQ2 = 0, not better.
            condition = m1_pq > m2_pq * (1 + percentage_val)
            # Special case: if m2_pq is 0 and percentage_val >= 0, then m1_pq must be > 0.
            condition = condition | ((m2_pq == 0) & (m1_pq > 0) & (percentage_val >=0) ) 
            avg_pq_df = avg_pq_df[condition]
        elif diff_mode == "Model 2 is X% better than Model 1":
            # PQ2 > PQ1 * (1 + X/100)
            condition = m2_pq > m1_pq * (1 + percentage_val)
            condition = condition | ((m1_pq == 0) & (m2_pq > 0) & (percentage_val >=0) )
            avg_pq_df = avg_pq_df[condition]

    # Apply Performance Threshold Filter
    if enable_thresh_filter and not avg_pq_df.empty: # Check if df is not empty from previous filter
        # Ensure models in selected_models_list are actually in the current avg_pq_df columns
        # (especially if some models were dropped due to no data for them initially)
        models_to_check_in_df = [m for m in selected_models_list if m in avg_pq_df.columns]
        
        if models_to_check_in_df: # Only proceed if there are relevant models to check
            if thresh_mode == "All selected models > Threshold":
                avg_pq_df = avg_pq_df[avg_pq_df[models_to_check_in_df].min(axis=1) > thresh_value]
            elif thresh_mode == "All selected models < Threshold":
                avg_pq_df = avg_pq_df[avg_pq_df[models_to_check_in_df].max(axis=1) < thresh_value]
                
    return generate_interactive_per_class_pq_plot(avg_pq_df)

def handle_apply_filters(sel_classes, sel_models, min_dist, max_dist, min_tot, max_tot, pq_m1, pq_m2, pq_n):
    if not sel_models or len(sel_models) < 1:
        no_model_msg = "Please select at least one model to visualize."
        empty_outputs = [None, no_model_msg, None, None, None, None, None, None, None, None, [], 0]
        return tuple(empty_outputs)

    filtered_indices = get_filtered_image_indices(
        sel_classes if sel_classes else [], min_dist, max_dist, min_tot, max_tot,
        pq_m1 if pq_n > 0 else None, 
        pq_m2 if pq_n > 0 else None,
        pq_n
    )

    if not filtered_indices:
        no_results_msg = "No images found matching all criteria. Try adjusting filters."
        empty_outputs = [None, no_results_msg, None, None, None, None, None, None, None, None, [], 0]
        return tuple(empty_outputs)
    
    current_idx_in_list = 0
    actual_image_id = filtered_indices[current_idx_in_list]
    
    (overview_pil, overall_df, per_class_df, instances_df, legend_pil, diff_html,
     input_p, target_p, model_ps) = generate_image_outputs(actual_image_id, sel_models)
    
    info_text = f"Found {len(filtered_indices)} images. Displaying image {current_idx_in_list + 1}/{len(filtered_indices)} (ID: {actual_image_id})."
    
    return (overview_pil, info_text, 
            overall_df, per_class_df, instances_df, 
            legend_pil, diff_html,
            input_p, target_p, model_ps,
            filtered_indices, current_idx_in_list)

def handle_navigation(sel_models, current_indices_list, current_display_idx, action="next", target_jump_id_str=None):
    if not sel_models or len(sel_models) < 1:
        # This case should ideally be prevented by disabling nav buttons if no models selected
        # or if current_indices_list is empty.
        return tuple([None, "Select models first.", None, None, None, None, None, None, None, None, current_indices_list, current_display_idx])

    if not current_indices_list:
        return tuple([None, "No images loaded. Apply filters first.", None, None, None, None, None, None, None, None, [], 0])

    num_images = len(current_indices_list)
    new_display_idx = current_display_idx

    if action == "next":
        new_display_idx = (current_display_idx + 1) % num_images
    elif action == "prev":
        new_display_idx = (current_display_idx - 1 + num_images) % num_images
    elif action == "jump" and target_jump_id_str:
        try:
            target_id = int(target_jump_id_str)
            if target_id in current_indices_list:
                new_display_idx = current_indices_list.index(target_id)
            else: # Target ID not in current list, keep current view
                info_text = f"Image ID {target_id} not in current filtered list ({num_images} images). Showing image {current_display_idx + 1}/{num_images} (ID: {current_indices_list[current_display_idx]})."
                # Re-fetch current image data to update info text, but don't change index
                actual_image_id = current_indices_list[current_display_idx]
                (overview_pil, overall_df, per_class_df, instances_df, legend_pil, diff_html,
                 input_p, target_p, model_ps) = generate_image_outputs(actual_image_id, sel_models)
                return (overview_pil, info_text, overall_df, per_class_df, instances_df, legend_pil, diff_html,
                        input_p, target_p, model_ps, current_indices_list, current_display_idx)
        except ValueError: # Invalid jump ID
             info_text = f"Invalid Image ID format. Showing image {current_display_idx + 1}/{num_images} (ID: {current_indices_list[current_display_idx]})."
             actual_image_id = current_indices_list[current_display_idx]
             (overview_pil, overall_df, per_class_df, instances_df, legend_pil, diff_html,
              input_p, target_p, model_ps) = generate_image_outputs(actual_image_id, sel_models)
             return (overview_pil, info_text, overall_df, per_class_df, instances_df, legend_pil, diff_html,
                     input_p, target_p, model_ps, current_indices_list, current_display_idx)


    actual_image_id = current_indices_list[new_display_idx]
    (overview_pil, overall_df, per_class_df, instances_df, legend_pil, diff_html,
     input_p, target_p, model_ps) = generate_image_outputs(actual_image_id, sel_models)
    
    info_text = f"Displaying image {new_display_idx + 1}/{num_images} (ID: {actual_image_id})."
    
    return (overview_pil, info_text, 
            overall_df, per_class_df, instances_df, 
            legend_pil, diff_html,
            input_p, target_p, model_ps,
            current_indices_list, new_display_idx) # Pass back list and new index for state

# --- Gradio UI Definition ---
css_styles = """
#gallery { min-height: 300px; }
.metrics-container { max-height: 700px; overflow-y: auto; }
.gr-dataframe table { font-size: 0.9em; }
"""
with gr.Blocks(title="Advanced Model Comparison Visualizer", css=css_styles, theme=gr.themes.Soft(font="Lato")) as demo:
    # State variables
    s_image_indices_list = gr.State([])
    s_current_display_index = gr.State(0)

    gr.Markdown("# Advanced Model Comparison Visualizer")
    gr.Markdown("Filter images by content and compare model predictions side-by-side with detailed metrics.")

    with gr.Row():
        with gr.Column(scale=1, min_width=350): # Controls Column
            gr.Markdown("### 1. Select Models & Classes")
            model_selector = gr.Dropdown(
                label="Select Models to Compare (Min 1)",
                choices=AVAILABLE_MODELS, 
                value=AVAILABLE_MODELS[:2] if len(AVAILABLE_MODELS) >= 2 else AVAILABLE_MODELS,
                multiselect=True
            )
            class_selector = gr.Dropdown(
                label="Filter by Classes (Optional, AND logic)",
                choices=ALL_CLASS_NAMES, 
                multiselect=True
            )

            with gr.Accordion("Advanced Content Filters (Optional)", open=False):
                with gr.Row():
                    min_distinct_classes_slider = gr.Number(label="Min Distinct Classes", value=None, step=1, minimum=0)
                    max_distinct_classes_slider = gr.Number(label="Max Distinct Classes", value=None, step=1, minimum=0)
                with gr.Row():
                    min_total_instances_slider = gr.Number(label="Min Total Instances", value=None, step=1, minimum=0)
                    max_total_instances_slider = gr.Number(label="Max Total Instances", value=None, step=1, minimum=0)
            
            with gr.Accordion("PQ Difference Ranking (Optional)", open=False):
                gr.Markdown("Rank images by the absolute Panoptic Quality (PQ) difference between two selected models. Only applies if Top N > 0.")
                pq_model1_selector = gr.Dropdown(choices=AVAILABLE_MODELS, label="Model 1 for PQ Diff", value=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None)
                pq_model2_selector = gr.Dropdown(choices=AVAILABLE_MODELS, label="Model 2 for PQ Diff", value=AVAILABLE_MODELS[1] if len(AVAILABLE_MODELS) > 1 else None)
                pq_top_n_slider = gr.Number(label="Show Top N Images by PQ Diff (0=disable)", value=0, step=1, minimum=0)

            apply_filters_btn = gr.Button("Apply Filters & Visualize", variant="primary")
            
            info_display_box = gr.Textbox(label="Status & Image Info", lines=3, interactive=False)
            
            gr.Markdown("### 2. Navigate Results")
            with gr.Row():
                prev_img_btn = gr.Button("⬅️ Previous")
                next_img_btn = gr.Button("Next ➡️")
            with gr.Row():
                jump_to_idx_input = gr.Textbox(label="Jump to Image ID", placeholder="e.g., 123")
                jump_img_btn = gr.Button("Go")
            
            color_legend_display = gr.Image(label="Class Color Legend", type="pil", interactive=False)

        with gr.Column(scale=3): # Outputs Column
            overview_image_display = gr.Image(label="Overview: Input | Target | Models", type="pil", interactive=True, show_share_button=False, show_download_button=True)
            
            with gr.Tabs():
                with gr.TabItem("📊 Metrics & Scores"):
                    with gr.Row():
                        with gr.Column(elem_classes="metrics-container"):
                            gr.Markdown("#### Instance Counts in Current Image (Target)")
                            current_image_instances_df = gr.DataFrame(headers=["Class", "Instances"], interactive=False)
                        with gr.Column(elem_classes="metrics-container"):
                            gr.Markdown("#### Overall Model Metrics (PQ, SQ, RQ)")
                            overall_model_metrics_df = gr.DataFrame(interactive=False)
                    with gr.Column(elem_classes="metrics-container"):
                         gr.Markdown("#### Per-Class Panoptic Quality (PQ)")
                         per_class_pq_metrics_df = gr.DataFrame(interactive=False)
                    with gr.Column(elem_classes="metrics-container"):
                        gr.Markdown("#### Model Comparison Differences")
                        model_differences_html = gr.HTML()
                
                with gr.TabItem("🖼️ Zoom & Inspect Images"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("##### Input Image")
                            input_image_display = gr.Image(label="Input", type="pil", interactive=True, show_download_button=True)
                        with gr.Column():
                            gr.Markdown("##### Target Segmentation")
                            target_image_display = gr.Image(label="Target", type="pil", interactive=True, show_download_button=True)
                    
                    gr.Markdown("##### Model Predictions (Click to enlarge, scroll in preview)")
                    model_predictions_gallery = gr.Gallery(
                        label="Model Predictions", columns=[2,3], height="auto", 
                        object_fit="contain", allow_preview=True, preview=True,
                        show_label=False, elem_id="gallery"
                    )
                    gr.Markdown("Zoom Tip: Click any image in the gallery for a full-screen preview with zoom capabilities.", elem_classes="text-xs text-gray-500")

                with gr.TabItem("📊 Overall Per-Class PQ Analysis"):
                    gr.Markdown("### Analyze Average Per-Class Panoptic Quality Across All Images")
                    gr.Markdown("This plot shows the average Panoptic Quality (PQ) for each class, calculated across all images in the dataset where the class was predicted by the respective model. Apply filters to refine the analysis.")
                    with gr.Row():
                        per_class_pq_model_selector = gr.Dropdown(
                            label="Select Models for Plot",
                            choices=AVAILABLE_MODELS,
                            value=AVAILABLE_MODELS[:2] if len(AVAILABLE_MODELS) >= 2 else AVAILABLE_MODELS, 
                            multiselect=True,
                            min_width=400
                        )
                    
                    with gr.Accordion("Advanced Plot Filters", open=False):
                        gr.Markdown("#### Difference Filter")
                        gr.Markdown("Compares the first two models selected in 'Select Models for Plot'. Requires at least two models to be selected above.")
                        enable_diff_filter_cb = gr.Checkbox(label="Enable Difference Filter", value=False)
                        diff_filter_percentage_num = gr.Number(label="Models 1 is X% different than Model 2. Define X (e.g., 20 for 20%)", value=20, minimum=0)
                        diff_filter_mode_radio = gr.Radio(
                            label="Diff: Mode", 
                            choices=["Model 1 is X% better than Model 2", "Model 2 is X% better than Model 1"], 
                            value="Model 1 is X% better than Model 2"
                        )

                        gr.Markdown("#### Performance Threshold Filter")
                        enable_thresh_filter_cb = gr.Checkbox(label="Enable Performance Threshold Filter", value=False)
                        thresh_filter_value_num = gr.Number(label="Threshold: PQ Value", value=0.5, minimum=0, maximum=1, step=0.01)
                        thresh_filter_mode_radio = gr.Radio(
                            label="Threshold: Mode", 
                            choices=["All selected models > Threshold", "All selected models < Threshold"], 
                            value="All selected models > Threshold"
                        )

                    analyze_per_class_pq_btn = gr.Button("Generate/Update Per-Class PQ Plot", variant="secondary") # min_width removed for auto-sizing
                    
                    per_class_pq_plot_display = gr.Plot(label="Average Per-Class PQ Comparison")

    # --- Component Connections ---
    outputs_for_render = [
        overview_image_display, info_display_box,
        overall_model_metrics_df, per_class_pq_metrics_df, current_image_instances_df,
        color_legend_display, model_differences_html,
        input_image_display, target_image_display, model_predictions_gallery,
        s_image_indices_list, s_current_display_index # State outputs
    ]
    
    filter_inputs = [
        class_selector, model_selector,
        min_distinct_classes_slider, max_distinct_classes_slider,
        min_total_instances_slider, max_total_instances_slider,
        pq_model1_selector, pq_model2_selector, pq_top_n_slider
    ]

    apply_filters_btn.click(
        fn=handle_apply_filters,
        inputs=filter_inputs,
        outputs=outputs_for_render
    )

    nav_common_inputs = [model_selector, s_image_indices_list, s_current_display_index]
    
    next_img_btn.click(
        fn=lambda ms, sil, scdi: handle_navigation(ms, sil, scdi, action="next"),
        inputs=nav_common_inputs,
        outputs=outputs_for_render
    )
    prev_img_btn.click(
        fn=lambda ms, sil, scdi: handle_navigation(ms, sil, scdi, action="prev"),
        inputs=nav_common_inputs,
        outputs=outputs_for_render
    )
    jump_img_btn.click(
        fn=lambda ms, sil, scdi, target_id: handle_navigation(ms, sil, scdi, action="jump", target_jump_id_str=target_id),
        inputs=nav_common_inputs + [jump_to_idx_input],
        outputs=outputs_for_render
    )

    # Connection for the new Per-Class PQ Analysis Tab
    plot_filter_inputs = [
        per_class_pq_model_selector,
        enable_diff_filter_cb,
        diff_filter_percentage_num, diff_filter_mode_radio,
        enable_thresh_filter_cb, thresh_filter_value_num, thresh_filter_mode_radio
    ]
    analyze_per_class_pq_btn.click(
        fn=handle_generate_interactive_per_class_pq_plot,
        inputs=plot_filter_inputs,
        outputs=[per_class_pq_plot_display]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True) 