# %%
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import json
import PIL.Image as Image

# %%
id_to_name = {}
name_to_id = {}
with open('/home/arda/thesis/eomt/output/class_names.txt', 'r') as f:
    for line in f:
        id, name = line.strip().split('\t')
        id_to_name[int(id)] = name
        name_to_id[name] = int(id)



# %%
with open('/home/arda/thesis/eomt/output/instance_counts.json', 'r') as f:
    results = json.load(f)

    
metric_results = results['pq']
instance_counts = results['instance_counts']


# %%
metric_results = {k.replace('_pq', ''):v for k,v in metric_results.items()}


# %%
instance_counts

# %%
experiments = [run for run in metric_results.keys() if '_pred_ADE_DINO_WEIGHTS' in run]

total_instance_count = 0
per_class_metrics_overall = {}
total_pq = 0

for experiment in experiments:
    for class_id, metrics in metric_results[experiment]['per_class_pq'].items():
        print(experiment)

        instance_count = instance_counts[experiment].get(class_id, 1)
        if class_id not in per_class_metrics_overall:
            per_class_metrics_overall[class_id] = [0,0]
        per_class_metrics_overall[class_id][0] += metrics * instance_count
        total_pq += metrics * instance_count
        per_class_metrics_overall[class_id][1] += instance_count
        total_instance_count += instance_count

for class_id, metrics in per_class_metrics_overall.items():
    if metrics[1] > 0:
        per_class_metrics_overall[class_id] = metrics[0] / metrics[1]
    else:
        per_class_metrics_overall[class_id] = 0



# %%
sum(per_class_metrics_overall.values())/len(per_class_metrics_overall.values())

# %%
new_metric_results = {}
for name, metrics in metric_results.items():
        
    metrics['overall_pq'] = float(np.mean(list(metrics['per_class_pq'].values())))
    metrics['overall_sq'] = float(np.mean(list(metrics['per_class_sq'].values())))
    metrics['overall_rq'] = float(np.mean(list(metrics['per_class_rq'].values())))
    metric_results[name] = metrics


# %%
metric_results = {k.replace('_pq', ''):v for k,v in metric_results.items()}

# %%
coco_vs_dino_metrics = {k.replace('_pq', ''):v for k,v in metric_results.items() if 'ADE' in k or 'target' in k or 'experiment_2_full' in k}
coco_vs_dino_instance_counts = {k:v for k,v in instance_counts.items() if 'ADE' in k or 'target' in k or 'experiment_2_full' in k}


# %% [markdown]
# # 0. Helper Functions

# %%
def get_image_name(idx):
    return f'val_{str(idx).zfill(4)}_input'

def get_prediction_name(idx, model_name):
    return f'val_{str(idx).zfill(4)}_pred_{model_name}'

def get_target_name(idx):
    return f'val_{str(idx).zfill(4)}_target'

def get_image(idx):
    img = Image.open(f'../output/{get_image_name(idx)}.png')
    return img

def get_prediction(idx, model_name):
    img = Image.open(f'../output/{get_prediction_name(idx, model_name)}.png')
    return img

def get_target(idx):
    img = Image.open(f'../output/{get_target_name(idx)}.png')
    return img



def get_image_metrics(idx, model_name):
    metrics = {}
    metrics['pq'] = metric_results[get_prediction_name(idx, model_name)]['overall_pq']
    metrics['sq'] = metric_results[get_prediction_name(idx, model_name)]['overall_sq']
    metrics['rq'] = metric_results[get_prediction_name(idx, model_name)]['overall_rq']
    metrics['per_class_pq'] = metric_results[get_prediction_name(idx, model_name)]['per_class_pq']
    metrics['per_class_sq'] = metric_results[get_prediction_name(idx, model_name)]['per_class_sq']
    metrics['per_class_rq'] = metric_results[get_prediction_name(idx, model_name)]['per_class_rq']
    return metrics

def get_image_instance_count(idx, class_name):
    return instance_counts[get_target_name(idx)][class_name]
    
def get_images_by_indexes(indexes, model_names):
    for idx in indexes:
        get_image(idx)
        get_target(idx)


        combined_data = {}
        for model_name in model_names:
            ade_data =coco_vs_dino_metrics[get_prediction_name(idx, model_name)]["per_class_pq"]
            combined_data[model_name] = ade_data
            # Create a pandas DataFrame
        df = pd.DataFrame(combined_data)

        # Ensure all keys from both dictionaries are included as rows
        all_classes = sorted(list(set(ade_data.keys())))
        df = df.reindex(all_classes).fillna(0.0) # Fill missing values with 0.0
        df.columns = model_names
        # Convert the DataFrame to a Markdown table
        markdown_table = df.to_markdown(floatfmt=".2f") # Added floatfmt=".2f"

        print(markdown_table)
        fig, axes = plt.subplots(nrows=1, ncols=len(model_names)+2, figsize=(5*len(model_names)+10, 10))

        axes[0].imshow(get_image(idx))
        axes[0].set_title(f'Input: {idx}')
        axes[0].axis('off')
        axes[1].imshow(get_target(idx))
        axes[1].set_title('Target')
        axes[1].axis('off')
        for i, model_name in enumerate(model_names):
            axes[i+2].imshow(get_prediction(idx, model_name))
            axes[i+2].set_title(model_name)
            axes[i+2].axis('off')
        plt.show()


def get_images_by_class_names(class_names):
    instance_counts_target = {k:v for k,v in instance_counts.items() if 'target' in k}
    matching_images = []

    
    if type(class_names) == str:
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



# %% [markdown]
# # 1. COCO vs DINOv2

# %%
coco_dino_pq_diffs = []

for i in range(2000):
    dino_pq = coco_vs_dino_metrics[get_prediction_name(i, 'ADE_DINO_WEIGHTS')]['overall_pq']
    coco_pq = coco_vs_dino_metrics[get_prediction_name(i, 'experiment_2_full')]['overall_pq']
    coco_dino_pq_diffs.append((i,dino_pq - coco_pq))

# Sort the lists by the pq value in descending order and take the top 10
top_5_dino_is_better = sorted(coco_dino_pq_diffs, key=lambda x: x[1], reverse=True)[:5]
top_5_coco_is_better = sorted(coco_dino_pq_diffs, key=lambda x: x[1], reverse=False)[:5]

print("Top 20 where DINO is better:", top_5_dino_is_better)
print("Top 20 where COCO is better:", top_5_coco_is_better)

# %% [markdown]
# ## 1.1 Most PQ Differences (Image Level)

# %% [markdown]
# ### 1.1.1 Where DINOv2 is superior (dino_pq - )

# %%
for pair in top_5_dino_is_better:
    


    idx = pair[0]
    
    ade_data =coco_vs_dino_metrics[get_prediction_name(idx, "ADE_DINO_WEIGHTS")]["per_class_pq"]
    exp_data =coco_vs_dino_metrics[get_prediction_name(idx, "experiment_2_full")]["per_class_pq"]

    combined_data = {
    "ADE": ade_data,
    "Experiment 2": exp_data
    }
        # Create a pandas DataFrame
    df = pd.DataFrame(combined_data)

    # Ensure all keys from both dictionaries are included as rows
    all_classes = sorted(list(set(ade_data.keys()) | set(exp_data.keys())))
    df = df.reindex(all_classes).fillna(0.0) # Fill missing values with 0.0
    df.columns = ['DINOv2', 'COCO']
    # Convert the DataFrame to a Markdown table
    markdown_table = df.to_markdown(floatfmt=".2f") # Added floatfmt=".2f"

    print(markdown_table)
    
    dinov2_pq = df['DINOv2'].mean()
    coco_pq = df['COCO'].mean()

    print(f"DINOv2 PQ: {dinov2_pq:.2f}, COCO PQ: {coco_pq:.2f}")

    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))
    axes[0].imshow(get_image(idx))
    axes[0].set_title(f'Input: {idx}')
    axes[0].axis('off')
    axes[1].imshow(get_target(idx))
    axes[1].set_title('Target')
    axes[1].axis('off')
    axes[2].imshow(get_prediction(idx, 'ADE_DINO_WEIGHTS'))
    axes[2].set_title(f'DINOv2 Pretrained Weights')
    axes[2].axis('off')
    axes[3].imshow(get_prediction(idx, 'experiment_2_full'))
    axes[3].set_title(f'COCO Pretrained Weights')
    axes[3].axis('off')
    plt.show()




# %% [markdown]
# ### 1.1.2 Where COCO is superior

# %%
for pair in top_5_coco_is_better:
    


    idx = pair[0]
    
    ade_data =coco_vs_dino_metrics[get_prediction_name(idx, "ADE_DINO_WEIGHTS")]["per_class_pq"]
    exp_data =coco_vs_dino_metrics[get_prediction_name(idx, "experiment_2_full")]["per_class_pq"]

    combined_data = {
    "ADE": ade_data,
    "Experiment 2": exp_data
    }
        # Create a pandas DataFrame
    df = pd.DataFrame(combined_data)

    # Ensure all keys from both dictionaries are included as rows
    all_classes = sorted(list(set(ade_data.keys()) | set(exp_data.keys())))
    df = df.reindex(all_classes).fillna(0.0) # Fill missing values with 0.0
    df.columns = ['DINOv2', 'COCO']
    # Convert the DataFrame to a Markdown table
    markdown_table = df.to_markdown(floatfmt=".2f") # Added floatfmt=".2f"

    print(markdown_table)
    
    dinov2_pq = df['DINOv2'].mean()
    coco_pq = df['COCO'].mean()

    print(f"DINOv2 PQ: {dinov2_pq:.2f}, COCO PQ: {coco_pq:.2f}")

    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))
    axes[0].imshow(get_image(idx))
    axes[0].set_title(f'Input: {idx}')
    axes[0].axis('off')
    axes[1].imshow(get_target(idx))
    axes[1].set_title('Target')
    axes[1].axis('off')
    axes[2].imshow(get_prediction(idx, 'ADE_DINO_WEIGHTS'))
    axes[2].set_title(f'DINOv2 Pretrained Weights')
    axes[2].axis('off')
    axes[3].imshow(get_prediction(idx, 'experiment_2_full'))
    axes[3].set_title(f'COCO Pretrained Weights')
    axes[3].axis('off')
    plt.show()



# %% [markdown]
# # Get Images By Classes

# %% [markdown]
# ## 1. DINO vs COCO

# %%
idxs = get_images_by_class_names(['table',
                                  'desk',
                                  'coffee table, cocktail table'
                                  ]
                                 )



models = ['ADE_DINO_WEIGHTS', 'experiment_2_full']

get_images_by_indexes(idxs, models)

# %% [markdown]
# ## Examples with most instances

# %%
img_total_instance={}
for k, v in instance_counts.items():
    if 'target' in k:
        img_total_instance[k] = sum(v.values())

sorted_img_total_instance = sorted(img_total_instance.items(), key=lambda x: x[1], reverse=True)
sorted_img_total_instance = [int(x[0].split('_')[1]) for x in sorted_img_total_instance]




# %%
get_images_by_indexes(sorted_img_total_instance[:10], models)

# %%



