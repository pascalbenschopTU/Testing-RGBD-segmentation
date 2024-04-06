#!/bin/bash

# Check if arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <dataset_location> <dataset_name> <use_gems> <use_cars>"
    exit 1
fi

# Assigning input arguments to variables
dataset_location=$1
dataset_name=$2
use_gems=$3
use_cars=$4

use_edge_enhancement=false
checkpoint_dir=checkpoints

# Set default values for use_gems and use_cars if not provided
if [ -z "$use_gems" ]; then
    use_gems=false
fi

if [ -z "$use_cars" ]; then
    use_cars=false
fi

# Function to create dataset
create_dataset() {
    # Check if dataset already exists
    if [ -d "DFormer/datasets/SynthDet_$dataset_name" ]; then
        echo "Dataset already exists, skipping dataset creation."
        return
    fi

    # Creating dataset
    solo2coco $dataset_location data/SynthDet/

    cd SynthDet_Pascal || exit
    python convert_solo_depth_to_coco.py $dataset_location ../data/SynthDet/coco/depth

    # Rename the dataset to the correct name
    mv "../data/SynthDet/coco" "../data/SynthDet/coco_$dataset_name"

    dataset_path="../data/SynthDet/coco_$dataset_name"

    if [ "$use_gems" == true ]; then
        # Convert the dataset to DFormer format with gems
        python convert_coco_to_dformer.py $dataset_path "../DFormer/datasets/SynthDet_$dataset_name" --gems
    elif [ "$use_cars" == true ]; then
        # Convert the dataset to DFormer format with cars
        python convert_coco_to_dformer.py $dataset_path "../DFormer/datasets/SynthDet_$dataset_name" --cars
    else
        # Convert the dataset to DFormer format
        python convert_coco_to_dformer.py $dataset_path "../DFormer/datasets/SynthDet_$dataset_name"
    fi

    if [ "$use_edge_enhancement" == true ]; then
        python add_edge_enhancement.py "../DFormer/datasets/SynthDet_$dataset_name"
    fi

    # Remove the dataset from the data folder
    rm -rf "$dataset_path"

    # Change to the DFormer directory
    cd ../DFormer || exit
}

create_dataset

# Copy the template file to the new file
cp "local_configs/SynthDet/SynthDet_template_DFormer_Tiny.py" "local_configs/SynthDet/SynthDet_${dataset_name}_Dformer_Tiny.py"
cd "local_configs/SynthDet" || exit

if [ "$use_gems" == true ]; then
    # Replace dataset name and classes in the new file with gems
    sed -i "s/C.dataset_name = .*/C.dataset_name = 'SynthDet_$dataset_name'/" "SynthDet_${dataset_name}_Dformer_Tiny.py"
    sed -i "s/classes = \"groceries\"/classes = \"gems\"/" "SynthDet_${dataset_name}_Dformer_Tiny.py"
elif [ "$use_cars" == true ]; then
    # Replace dataset name and classes in the new file with cars
    sed -i "s/C.dataset_name = .*/C.dataset_name = 'SynthDet_$dataset_name'/" "SynthDet_${dataset_name}_Dformer_Tiny.py"
    sed -i "s/classes = \"groceries\"/classes = \"cars\"/" "SynthDet_${dataset_name}_Dformer_Tiny.py"
else
    # Replace dataset name in the new file
    sed -i "s/C.dataset_name = .*/C.dataset_name = 'SynthDet_$dataset_name'/" "SynthDet_${dataset_name}_Dformer_Tiny.py"
fi

# Change back to DFormer
cd ../..

# Train the model
python utils/train.py --config=local_configs.SynthDet.SynthDet_${dataset_name}_Dformer_Tiny --gpus 1 --checkpoint_dir $checkpoint_dir

# Get the last directory starting with "run" from the given path
last_directory=$(ls -td $checkpoint_dir/SynthDet_${dataset_name}_DFormer-Tiny/run* | head -n 1)

# Get the last filename starting with "epoch" in the last directory
last_filename=$(ls -t $last_directory/epoch* | head -n 1)

echo "Last directory: $last_directory"
echo "Last filename: $last_filename"

rgb_depth_model_weights="$checkpoint_dir/SynthDet_${dataset_name}_DFormer-Tiny/$last_directory/$last_filename"

python utils/evaluate_models.py --config=local_configs.SynthDet.SynthDet_${dataset_name}_Dformer_Tiny --model_weights $rgb_depth_model_weights

new_dataset_path="../DFormer/datasets/SynthDet_$dataset_name"
mv "$new_dataset_path/Depth" "$new_dataset_path/Depth_original"
mv "$new_dataset_path/Grayscale" "$new_dataset_path/Depth"

# Train the model
python utils/train.py --config=local_configs.SynthDet.SynthDet_${dataset_name}_Dformer_Tiny --gpus 1 --checkpoint_dir $checkpoint_dir

# Get the last directory starting with "run" from the given path
last_directory=$(ls -td $checkpoint_dir/SynthDet_${dataset_name}_DFormer-Tiny/run* | head -n 1)

# Get the last filename starting with "epoch" in the last directory
last_filename=$(ls -t $last_directory/epoch* | head -n 1)

echo "Last directory: $last_directory"
echo "Last filename: $last_filename"

rgb_black_model_weights="$checkpoint_dir/SynthDet_${dataset_name}_DFormer-Tiny/$last_directory/$last_filename"

python utils/evaluate_models.py --config=local_configs.SynthDet.SynthDet_${dataset_name}_Dformer_Tiny --model_weights $rgb_black_model_weights

# Evaluate the models
python utils/create_predictions.py --model_a_path $rgb_black_model_weights --model_b_path $rgb_depth_model_weights --dir_dataset datasets/SynthDet_${dataset_name} --config=local_configs.SynthDet.SynthDet_${dataset_name}_Dformer_Tiny

# Create gen_results.txt and store the last command
echo "utils/create_predictions.py --model_a_path $rgb_black_model_weights --model_b_path $rgb_depth_model_weights --dir_dataset datasets/SynthDet_${dataset_name} --config=local_configs.SynthDet.SynthDet_${dataset_name}_Dformer_Tiny" > "$checkpoint_dir/SynthDet_${dataset_name}_DFormer-Tiny/gen_results.txt"
