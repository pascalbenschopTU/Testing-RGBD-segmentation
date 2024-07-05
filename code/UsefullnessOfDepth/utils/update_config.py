import argparse
import importlib
import sys
import cv2
import os
sys.path.append('../UsefullnessOfDepth')

def update_config(config_location, variables_to_update={}):

    # Remove keys where the value is None
    variables_to_update = {key: value for key, value in variables_to_update.items() if value is not None}

    if "dataset_name" in variables_to_update:
        variables_to_update = get_dataset_details(variables_to_update)

    # Read the original file and update the variables
    with open(config_location, 'r') as original_file:
        original_content = original_file.readlines()

    new_variables = {key: value for key, value in variables_to_update.items() if f"C.{key} = " not in original_content}
    # Update the variables in the original content
    for key, value in variables_to_update.items():
        for i, line in enumerate(original_content):
            if f'C.{key} = ' in line:
                original_content[i] = f'C.{key} = {repr(value)}\n'
                if key in new_variables:
                    new_variables.pop(key)
                break

    # Add the new variables to the end of the file
    if len(new_variables) > 0:
        original_content.append('\n')
    for key, value in new_variables.items():
        original_content.append(f'C.{key} = {repr(value)}\n')

    # Write the modified content back to the original file
    with open(config_location, 'w') as modified_file:
        modified_file.writelines(original_content)

    if ".py" in config_location:
        config_location = config_location.replace(".py", "")
        config_location = config_location.replace("\\", ".")
        while config_location.startswith("."):
            config_location = config_location[1:]
    
    config_module = importlib.reload(importlib.import_module(config_location))
    config = config_module.config

    return config


def get_dataset_details(variables_to_update):
    dataset_name = variables_to_update["dataset_name"]
    # Walk the datasets folder and find the dataset location
    if os.path.exists(dataset_name):
        dataset_name = os.path.basename(os.path.normpath(dataset_name))
        variables_to_update["dataset_name"] = dataset_name
    
    dataset_location = None
    for root, dirs, files in os.walk('datasets'):
        for dir in dirs:
            if dataset_name == dir:
                dataset_location = os.path.join(root, dir)
                break
        if dataset_location is not None:
            break

    # If there is a directory between datasets and the dataset name, update the config root_dir
    if dataset_location is not None:
        sub_dir = dataset_location.split('\\')[:-1]
        sub_dir = '/'.join(sub_dir)
        variables_to_update["root_dir"] = sub_dir

    if dataset_location is None:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in datasets folder")

    return variables_to_update


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="DFormer.local_configs.SynthDet.SynthDet_black_back_default_2_Dformer_Tiny",
        help="The config to use for training the model",
    )

    args = parser.parse_args()

    config = update_config(args.config, {'nepochs': 69, "num_train_imgs": 200, "classes": "gems", "dataset_name": "SynthDet_ATest_DFormer_Tiny"})

    