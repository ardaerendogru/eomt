#!/bin/bash

# Script to run all experiments from the table
CONFIG_PATH="/home/arda/thesis/eomt/configs/ade20k/panoptic/eomt_large_640.yaml"
BASE_CMD="python main.py fit -c $CONFIG_PATH --trainer.devices 2 --data.batch_size 8 --data.path data/ade_subsets/1 --model.ckpt_path /home/arda/thesis/eomt/checkpoints/COCO_panoptic_640.bin --model.load_ckpt_class_head False"

# Function to modify YAML file
modify_yaml() {
    local file=$1
    local mask_annealing=$2
    local finetuning_type=$3
    local lr=$4
    local lr_head_multiplier=$5
    local id=$6

    # Create a temporary file
    cp $file temp_config.yaml

    # Set finetuning type based on modules
    echo "Setting experiment ID: $id"
    echo "Mask Annealing: $mask_annealing, Finetuning Type: $finetuning_type"
    echo "LR: $lr, LR Head Multiplier: $lr_head_multiplier"

    # Modify the YAML file
    sed -i "s/attn_mask_annealing_enabled:.*/attn_mask_annealing_enabled: $mask_annealing/" temp_config.yaml
    sed -i "s/finetuning_type:.*/finetuning_type: \"$finetuning_type\"/" temp_config.yaml
    sed -i "s/lr:.*/lr: $lr/" temp_config.yaml
    sed -i "s/lr_head_multiplier:.*/lr_head_multiplier: $lr_head_multiplier/" temp_config.yaml
    
    # Remove finetuning_modules if present (since we're using named types)
    sed -i "/finetuning_modules:/d" temp_config.yaml

    # Run the command
    experiment_name="experiment_${id}"
    $BASE_CMD --trainer.logger.init_args.name "$experiment_name" --config temp_config.yaml
    
    # Clean up
    rm temp_config.yaml
}

# Run experiments according to the table
echo "Starting experiments..."

# ID 1 - All modules enabled
modify_yaml $CONFIG_PATH "True" "all" "1.00E-04" "1" "1"

# ID 2 - No mask annealing, all modules
modify_yaml $CONFIG_PATH "False" "all" "1.00E-04" "1" "2"

# ID 3 - No mask annealing, all modules, LR head multiplier 10
modify_yaml $CONFIG_PATH "False" "all" "1.00E-04" "10" "3"

# ID 4 - No mask annealing, all modules
modify_yaml $CONFIG_PATH "False" "all" "1.00E-04" "1" "4"

# ID 5 - No mask annealing, all modules, lower LR
modify_yaml $CONFIG_PATH "False" "all" "5.00E-05" "1" "5"

# ID 6 - No L1, yes to everything else
modify_yaml $CONFIG_PATH "True" "full_decoder" "1.00E-04" "1" "6"

# ID 7 - Skip PiSSa

# ID 8 - Skip LoRA

# ID 9 - Only query + mask head + upscale
modify_yaml $CONFIG_PATH "False" "qhead" "1.00E-04" "1" "9"

# ID 10 - Same as 9 but with mask annealing
modify_yaml $CONFIG_PATH "True" "qhead" "1.00E-04" "1" "10"

# ID 11 - Same as 10
modify_yaml $CONFIG_PATH "True" "qhead" "1.00E-04" "1" "11"

# ID 12 - Only mask head
modify_yaml $CONFIG_PATH "True" "MLPs" "1.00E-04" "1" "12"

# ID 13 - Only query
modify_yaml $CONFIG_PATH "True" "Query" "1.00E-04" "1" "13"

# ID 14 - Only class_head (Linear)
modify_yaml $CONFIG_PATH "True" "Linear" "1.00E-04" "1" "14"

echo "All experiments completed!" 