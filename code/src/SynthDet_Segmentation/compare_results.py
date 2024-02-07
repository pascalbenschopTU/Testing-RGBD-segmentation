import argparse
import importlib
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import matplotlib.patches as mpatches
from tqdm import tqdm
import torch
import torch.nn as nn
import sys
from simple_dataloader import get_val_loader
from segmentation_model import SmallUNet

sys.path.append("../../DFormer/utils/dataloader/")
from RGBXDataset import RGBXDataset

# Class names from the config file
# class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']
# class_names = ['background', 'book_dorkdiaries_aladdin', 'candy_minipralines_lindt', 'candy_raffaello_confetteria', 'cereal_capn_crunch', 'cereal_cheerios_honeynut', 'cereal_corn_flakes', 'cereal_cracklinoatbran_kelloggs', 'cereal_oatmealsquares_quaker', 'cereal_puffins_barbaras', 'cereal_raisin_bran', 'cereal_rice_krispies', 'chips_gardensalsa_sunchips', 'chips_sourcream_lays', 'cleaning_freegentle_tide', 'cleaning_snuggle_henkel', 'cracker_honeymaid_nabisco', 'cracker_lightrye_wasa', 'cracker_triscuit_avocado', 'cracker_zwieback_brandt', 'craft_yarn_caron', 'drink_adrenaline_shock', 'drink_coffeebeans_kickinghorse', 'drink_greentea_itoen', 'drink_orangejuice_minutemaid', 'drink_whippingcream_lucerne', 'footware_slippers_disney', 'hygiene_poise_pads', 'lotion_essentially_nivea', 'lotion_vanilla_nivea', 'pasta_lasagne_barilla', 'pest_antbaits_terro', 'porridge_grits_quaker', 'seasoning_canesugar_candh', 'snack_biscotti_ghiott', 'snack_breadsticks_nutella', 'snack_chips_pringles', 'snack_coffeecakes_hostess', 'snack_cookie_famousamos', 'snack_cookie_petitecolier', 'snack_cookie_quadratini', 'snack_cookie_waffeletten', 'snack_cookie_walkers', 'snack_cookies_fourre', 'snack_granolabar_kashi', 'snack_granolabar_kind', 'snack_granolabar_naturevalley', 'snack_granolabar_quaker', 'snack_salame_hillshire', 'soup_chickenenchilada_progresso', 'soup_tomato_pacific', 'storage_ziploc_sandwich', 'toiletry_tissue_softly', 'toiletry_toothpaste_colgate', 'toy_cat_melissa', 'utensil_candle_decorators', 'utensil_coffee_filters', 'utensil_cottonovals_signaturecare', 'utensil_papertowels_valuecorner', 'utensil_toiletpaper_scott', 'utensil_trashbag_valuecorner', 'vitamin_centrumsilver_adults', 'vitamin_centrumsilver_men', 'vitamin_centrumsilver_woman']
class_names = ["background", "book_dorkdiaries_aladdin", "candy_minipralines_lindt", "candy_raffaello_confetteria", "cereal_capn_crunch", "cereal_cheerios_honeynut", "cereal_corn_flakes", "cereal_cracklinoatbran_kelloggs", "cereal_oatmealsquares_quaker", "cereal_puffins_barbaras", "cereal_raisin_bran", "cereal_rice_krispies", "chips_gardensalsa_sunchips", "chips_sourcream_lays", "cleaning_freegentle_tide", "cleaning_snuggle_henkel", "cracker_honeymaid_nabisco", "cracker_lightrye_wasa", "cracker_triscuit_avocado", "cracker_zwieback_brandt", "craft_yarn_caron", "drink_adrenaline_shock", "drink_coffeebeans_kickinghorse", "drink_greentea_itoen", "drink_orangejuice_minutemaid", "drink_whippingcream_lucerne", "footware_slippers_disney", "hygiene_poise_pads", "lotion_essentially_nivea", "lotion_vanilla_nivea", "pasta_lasagne_barilla", "pest_antbaits_terro", "porridge_grits_quaker", "seasoning_canesugar_candh", "snack_biscotti_ghiott", "snack_breadsticks_nutella", "snack_chips_pringles", "snack_coffeecakes_hostess", "snack_cookie_famousamos", "snack_cookie_petitecolier", "snack_cookie_quadratini", "snack_cookie_waffeletten", "snack_cookie_walkers", "snack_cookies_fourre", "snack_granolabar_kashi", "snack_granolabar_kind", "snack_granolabar_naturevalley", "snack_granolabar_quaker", "snack_salame_hillshire", "soup_chickenenchilada_progresso", "soup_tomato_pacific", "storage_ziploc_sandwich", "toiletry_tissue_softly", "toiletry_toothpaste_colgate", "toy_cat_melissa", "utensil_candle_decorators", "utensil_coffee_filters", "utensil_cottonovals_signaturecare", "utensil_papertowels_valuecorner", "utensil_toiletpaper_scott", "utensil_trashbag_valuecorner", "vitamin_centrumsilver_adults", "vitamin_centrumsilver_men", "vitamin_centrumsilver_woman", "5 Side Diamond", "HardStar", "BeveledStar", "Hexgon", "Cubie", "Spiral", "Penta", "Heart", "Cuboid", "SphereGemLarge", "Diamondo", "Diamondo5side", "4 Side Diamond", "Character", "SoftStar", "SphereGemSmall", "CubieBeveled"]


max_length = max(len(class_name) for class_name in class_names)
class_names = [class_name.ljust(max_length) for class_name in class_names]

# N_CLASSES = 64
N_CLASSES = 81

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_results(dir_rgb, dir_rgbd, metric="ious"):
    acc_rgb_text_location = os.path.join(dir_rgb, "results.txt")
    acc_depth_text_location = os.path.join(dir_rgbd, "results.txt")

    # Accuracy values from the text file
    acc_rgb_values_text = open(acc_rgb_text_location).read()
    acc_depth_values_text = open(acc_depth_text_location).read()

    # Parse accuracy values
    acc_rgb_values = {class_name: [] for class_name in class_names}
    for line in acc_rgb_values_text.split("\n")[5:]:
        if metric in line:
            line = line.split(metric)[1].split("],")[0]
            values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            for class_name, value in zip(class_names, values):
                acc_rgb_values[class_name].append(value)

    acc_depth_values = {class_name: [] for class_name in class_names}
    for line in acc_depth_values_text.split("\n")[5:]:
        if metric in line:
            line = line.split(metric)[1].split("],")[0]
            values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            for class_name, value in zip(class_names, values):
                acc_depth_values[class_name].append(value)


    rgb_values = list(acc_rgb_values.values())
    rgb_values = [value[0] for value in rgb_values]

    depth_values = list(acc_depth_values.values())
    depth_values = [value[0] for value in depth_values]

    return rgb_values, depth_values

def plot_accuracy(rgb_values, depth_values, difference):
    sorted_indices = np.argsort(difference)

    sorted_rgb_values = np.array(rgb_values)[sorted_indices]
    sorted_depth_values = np.array(depth_values)[sorted_indices]
    sorted_class_names = np.array(class_names)[sorted_indices]

    # Make all strings in sorted_class_names the same length
    max_length = max(len(class_name) for class_name in sorted_class_names)
    sorted_class_names = [class_name.ljust(max_length) for class_name in sorted_class_names]

    # Generate histograms
    plt.figure(figsize=(20, 10))
    x = range(len(sorted_class_names))
    plt.xticks(x, sorted_class_names, rotation=-45, ha='left', rotation_mode="anchor")

    plt.bar(np.array(x) - 0.2, sorted_rgb_values, label='RGB', color='#FF69B4', width=0.4)
    plt.bar(np.array(x) + 0.2, sorted_depth_values, label='RGB + Depth', color='green', width=0.4)

    plt.title('Accuracy values for all classes over epochs')
    plt.xlabel('Class Names')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def statistics(differences, class_names):
    print("Average difference: ", np.mean(differences))
    print("Standard deviation: ", np.std(differences))
    print("Max difference: ", np.max(differences))
    print("Min difference: ", np.min(differences))

    # Get all classes that have a difference outside of 95% confidence interval
    mean = np.mean(differences)
    std = np.std(differences)
    outside_95 = []
    for i, difference in enumerate(differences):
        if difference > mean + 2 * std or difference < mean - 2 * std:
            outside_95.append(class_names[i])

    print("Classes that have a difference outside of 95% confidence interval: ", outside_95)

    classes_of_interest = [class_names.index(class_name) for class_name in outside_95]

    return classes_of_interest

@torch.no_grad()
def create_and_plot_predictions(args, classes_of_interest):
    config_module = importlib.import_module(args.config)
    config = config_module.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a = SmallUNet(args.channels_a, N_CLASSES, criterion=torch.nn.CrossEntropyLoss(reduction='mean'))
    model_a.load_state_dict(torch.load(args.model_a_path))
    model_a.to(device)

    model_b = SmallUNet(args.channels_b, N_CLASSES, criterion=torch.nn.CrossEntropyLoss(reduction='mean'))
    model_b.load_state_dict(torch.load(args.model_b_path))
    model_b.to(device)

    config.val_batch_size = 1
    val_loader = get_val_loader(RGBXDataset, config)

    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)
  
    model_a.eval()
    model_b.eval()

    for i, minibatch in enumerate(tqdm(val_loader)):
        images = minibatch["data"]
        labels = minibatch["label"]
        modal_xs = minibatch["modal_x"]

        if not any(label.item() in classes_of_interest for label in labels.unique()):
            continue
        else:
            print("Found a class of interest: ", labels.unique(), classes_of_interest)

        # print(images.shape,labels.shape)
        images = images.to(device)
        modal_xs = modal_xs.to(device)
        labels = labels.to(device)
        B, H, W = labels.shape

        logits_a = model_a(images, modal_xs)
        predictions_a = logits_a.softmax(dim=1)

        logits_b = model_b(images, modal_xs)
        predictions_b = logits_b.softmax(dim=1)
        
        plot_prediction(predictions_a.cpu().numpy(), predictions_b.cpu().numpy(), i, args)


def plot_prediction(prediction_a, prediction_b, index, args):
    dir_dataset = args.dir_dataset

    target_format = os.path.join(dir_dataset, "labels", "test_{}.png")
    depth_format = os.path.join(dir_dataset, "Depth", "test_{}.png")
    img_format = os.path.join(dir_dataset, "RGB", "test_{}.png")

    length_predictions = prediction_a.shape[0]

    # Load the target and prediction as numpy files
    for i in range(0, length_predictions):
        # Convert the prediction to RGB
        pred_rgb_black = prediction_a[i]
        pred_rgb_depth = prediction_b[i]
        
        # Load the target and depth and rgb image
        target = plt.imread(target_format.format(i+index))
        depth = plt.imread(depth_format.format(i+index))
        img = plt.imread(img_format.format(i+index))

        target = (target * 255).astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)
        img = (img * 255).astype(np.uint8)

        # Plot the target, predictions and depth and rgb image
        # Also add the class names in legend
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        cmap = plt.cm.get_cmap('viridis', N_CLASSES)
        target_im = ax[0][0].imshow(target, cmap=cmap, vmin=0, vmax=N_CLASSES - 1)
        ax[0][0].set_title("target")
        rgb_black = ax[0][1].imshow(pred_rgb_black, cmap=cmap, vmin=0, vmax=N_CLASSES - 1)
        ax[0][1].set_title("pred_rgb")
        rgb_depth = ax[0][2].imshow(pred_rgb_depth, cmap=cmap, vmin=0, vmax=N_CLASSES - 1)
        ax[0][2].set_title("pred_rgb_depth")

        # get the colors of the values, according to the 
        values = np.unique(target.ravel())
        # values = np.concatenate((values, np.unique(pred_rgb_black.ravel())))
        # values = np.concatenate((values, np.unique(pred_rgb_depth.ravel())))
        values = np.unique(values)
        
        # Convert values to integers
        values = values.astype(int)
        
        # colormap used by imshow
        colors = [target_im.cmap(target_im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [mpatches.Patch(color=colors[j], label=f"{values[j]} " + class_names[values[j]]) for j in range(len(values)) if values[j] < len(class_names)]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)

        ax[1][0].imshow(depth)
        ax[1][0].set_title("depth")
        ax[1][1].imshow(img)
        ax[1][1].set_title("rgb")

        plt.show()

def plot_predictions(predictions_dir_a, predictions_dir_b, args):
    # Load the target and prediction as numpy files
    for i in range(0, 100):
        # Convert the prediction to RGB
        pred_rgb_black = np.load(os.path.join(predictions_dir_a, f"pred_test_{i}.npy"))
        pred_rgb_depth = np.load(os.path.join(predictions_dir_b, f"pred_test_{i}.npy"))
        
        plot_prediction(pred_rgb_black, pred_rgb_depth, i, args)

def main(args):
    dir_model_a = args.model_a_path
    dir_model_a_directory = os.path.dirname(dir_model_a)
    dir_model_b = args.model_b_path
    dir_model_b_directory = os.path.dirname(dir_model_b)
    
    rgb_values, depth_values = get_results(dir_model_a_directory, dir_model_b_directory)

    difference = np.array(rgb_values) - np.array(depth_values)
    plot_accuracy(rgb_values, depth_values, difference)

    classes_of_interest = statistics(difference, class_names)

    if os.path.isdir(dir_model_a_directory) and os.path.isdir(dir_model_b_directory):
        predictions_dir_a = os.path.join(dir_model_a_directory, "predictions")
        predictions_dir_b = os.path.join(dir_model_b_directory, "predictions")
        if os.path.isdir(predictions_dir_a) and os.path.isdir(predictions_dir_b):
            plot_predictions(predictions_dir_a, predictions_dir_b, args)
            return

    create_and_plot_predictions(args, classes_of_interest)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_a_path', type=str, help='Path to the RGB model directory', default=None)
    argparser.add_argument('--model_b_path', type=str, help='Path to the RGBD model directory', default=None)
    argparser.add_argument('--dir_dataset', type=str, help='Path to the dataset directory')
    argparser.add_argument('--channels_a', type=int, default=3)
    argparser.add_argument('--channels_b', type=int, default=4)
    argparser.add_argument('--config', type=str, default='config.SynthDet_default_Segmodel')
    args = argparser.parse_args()
    
    main(args)
