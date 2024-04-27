import argparse
import importlib
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import cv2
import matplotlib.patches as mpatches
from tqdm import tqdm
import torch
import torch.nn as nn
from easydict import EasyDict

import warnings
# Filter out MMCV warnings by message
warnings.filterwarnings("ignore", message="On January 1, 2023, MMCV will release v2.0.0*")

import sys
sys.path.append('../DFormer')
from utils.dataloader.dataloader import get_train_loader,get_val_loader
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dformer_gradcam import DFormerAnalyzer



# Class names from the config file
CLASS_NAMES_SUNRGBD = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']

CLASS_NAMES_GROCERIES = ['background', 'book_dorkdiaries_aladdin', 'candy_minipralines_lindt', 'candy_raffaello_confetteria', 'cereal_capn_crunch', 'cereal_cheerios_honeynut', 'cereal_corn_flakes', 'cereal_cracklinoatbran_kelloggs', 'cereal_oatmealsquares_quaker', 'cereal_puffins_barbaras', 'cereal_raisin_bran', 'cereal_rice_krispies', 'chips_gardensalsa_sunchips', 'chips_sourcream_lays', 'cleaning_freegentle_tide', 'cleaning_snuggle_henkel', 'cracker_honeymaid_nabisco', 'cracker_lightrye_wasa', 'cracker_triscuit_avocado', 'cracker_zwieback_brandt', 'craft_yarn_caron', 'drink_adrenaline_shock', 'drink_coffeebeans_kickinghorse', 'drink_greentea_itoen', 'drink_orangejuice_minutemaid', 'drink_whippingcream_lucerne', 'footware_slippers_disney', 'hygiene_poise_pads', 'lotion_essentially_nivea', 'lotion_vanilla_nivea', 'pasta_lasagne_barilla', 'pest_antbaits_terro', 'porridge_grits_quaker', 'seasoning_canesugar_candh', 'snack_biscotti_ghiott', 'snack_breadsticks_nutella', 'snack_chips_pringles', 'snack_coffeecakes_hostess', 'snack_cookie_famousamos', 'snack_cookie_petitecolier', 'snack_cookie_quadratini', 'snack_cookie_waffeletten', 'snack_cookie_walkers', 'snack_cookies_fourre', 'snack_granolabar_kashi', 'snack_granolabar_kind', 'snack_granolabar_naturevalley', 'snack_granolabar_quaker', 'snack_salame_hillshire', 'soup_chickenenchilada_progresso', 'soup_tomato_pacific', 'storage_ziploc_sandwich', 'toiletry_tissue_softly', 'toiletry_toothpaste_colgate', 'toy_cat_melissa', 'utensil_candle_decorators', 'utensil_coffee_filters', 'utensil_cottonovals_signaturecare', 'utensil_papertowels_valuecorner', 'utensil_toiletpaper_scott', 'utensil_trashbag_valuecorner', 'vitamin_centrumsilver_adults', 'vitamin_centrumsilver_men', 'vitamin_centrumsilver_woman']
CLASS_NAMES_GEMS = ["background", "5 Side Diamond", "HardStar", "BeveledStar", "Hexgon", "Cubie", "Spiral", "Penta", "Heart", "Cuboid", "SphereGemLarge", "Diamondo", "Diamondo5side", "4 Side Diamond", "SoftStar", "SphereGemSmall", "CubieBeveled"]
CLASS_NAMES_CARS = ["background", "Car-7", "Car-1", "Car-2", "Car-4", "Car-3", "Car-6", "Car-5", "Car-8"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResultAnalyzer:
    def __init__(self, args, classes="groceries") -> None:
        config_module = importlib.import_module(args.config)
        self.config = config_module.config

        if self.config is not None:
            classes = self.config.classes

        self.dir_model_a = args.model_a_path
        self.dir_model_a_directory = os.path.dirname(self.dir_model_a)
        self.dir_model_b = args.model_b_path
        self.dir_model_b_directory = os.path.dirname(self.dir_model_b)

        self.classes_of_interest = [0]

        if self.config.class_names is not None:
            self.class_names = self.config.class_names
        else: 
            if classes == "groceries":
                self.class_names = CLASS_NAMES_GROCERIES
            elif classes == "gems":
                self.class_names = CLASS_NAMES_GEMS
            elif classes == "all":
                self.class_names = np.concatenate((CLASS_NAMES_GROCERIES, CLASS_NAMES_GEMS))
            elif classes == "sunrgbd":
                self.class_names = CLASS_NAMES_SUNRGBD
            else:
                raise ValueError(f"Classes {classes} not found")

        max_length = max(len(class_name) for class_name in self.class_names)
        self.class_names = [class_name.ljust(max_length) for class_name in self.class_names]
        self.num_classes = len(self.class_names)

        self.get_results()

    def get_results(self):
        acc_rgb_text_location = os.path.join(self.dir_model_a_directory, "results.txt")
        acc_depth_text_location = os.path.join(self.dir_model_b_directory, "results.txt")

        # Accuracy values from the text file
        acc_rgb_values_text = open(acc_rgb_text_location).read()
        acc_depth_values_text = open(acc_depth_text_location).read()

        class_metric_search = "ious"
        class_metric_start = "acc:"

        pred_metric_search = "mious"
        pred_metric_start = "miou:"

        float_regex = r"[-+]?\d*\.\d+|\d+"

        # Parse accuracy values from the text file
        self.class_iou_rgb_values = []
        self.mean_pred_iou_rgb_values = []
        self.pred_iou_rgb_values = []
        for i, line in enumerate(acc_rgb_values_text.split("\n")):
            if line.startswith(class_metric_start):
                mean_prediction_iou = line.split(pred_metric_start)[1]
                self.mean_pred_iou_rgb_values.append(float(mean_prediction_iou.strip()))
                # Get iou values for each class from the final epoch
                line = line.split(class_metric_search)[1].split("],")[0]
                values = [float(x) for x in re.findall(float_regex, line)]
                self.class_iou_rgb_values = values
                
            if line.startswith(pred_metric_start) and pred_metric_search in line:
                line = line.split(pred_metric_search)[1].split("],")[0]
                self.pred_iou_rgb_values = [float(x) for x in re.findall(float_regex, line)]

        self.class_iou_depth_values = []
        self.mean_pred_iou_depth_values = []
        self.pred_iou_depth_values = []
        for i, line in enumerate(acc_depth_values_text.split("\n")):
            if line.startswith(class_metric_start):
                mean_prediction_iou = line.split(pred_metric_start)[1]
                self.mean_pred_iou_depth_values.append(float(mean_prediction_iou.strip()))
                # Get iou values for each class from the final epoch
                line = line.split(class_metric_search)[1].split("],")[0]
                values = [float(x) for x in re.findall(float_regex, line)]
                self.class_iou_depth_values = values

            if line.startswith(pred_metric_start) and pred_metric_search in line:
                line = line.split(pred_metric_search)[1].split("],")[0]
                self.pred_iou_depth_values = [float(x) for x in re.findall(float_regex, line)]


        with open(acc_rgb_text_location, "a") as file:
            standard_deviation = np.std(self.pred_iou_rgb_values)
            mean = np.mean(self.pred_iou_rgb_values)
            file.write(f"Standard deviation: {standard_deviation}, mean: {mean}\n")

        with open(acc_depth_text_location, "a") as file:
            standard_deviation = np.std(self.pred_iou_depth_values)
            mean = np.mean(self.pred_iou_depth_values)
            file.write(f"Standard deviation: {standard_deviation}, mean: {mean}\n")

    def plot_class_values(self, difference):
        sorted_indices = np.argsort(difference)

        sorted_rgb_values = np.array(self.class_iou_rgb_values)[sorted_indices]
        sorted_depth_values = np.array(self.class_iou_depth_values)[sorted_indices]
        sorted_class_names = np.array(self.class_names)[sorted_indices]

        self.classes_of_interest = sorted_indices[:5]

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

    def plot_ious_over_epochs(self):
        plt.figure(figsize=(20, 10))
        plt.plot(range(len(self.mean_pred_iou_rgb_values)), self.mean_pred_iou_rgb_values, label='RGB', color='#FF69B4')
        plt.plot(range(len(self.mean_pred_iou_depth_values)), self.mean_pred_iou_depth_values, label='RGB + Depth', color='green')
        plt.title('Mean IOU over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean IOU')
        plt.legend()
        plt.show()

    def plot_ious_over_dataset(self):
        plt.figure(figsize=(20, 10))
        plt.plot(range(len(self.pred_iou_rgb_values)), self.pred_iou_rgb_values, 'o', label='RGB', color='#FF69B4')
        plt.plot(range(len(self.pred_iou_depth_values)), self.pred_iou_depth_values, 'o', label='RGB + Depth', color='green')
        # Generate x values for plotting the trendlines
        max_values = max(len(self.pred_iou_rgb_values), len(self.pred_iou_depth_values))
        x_values = np.arange(max_values)

        plt.plot(x_values, np.poly1d(np.polyfit(range(len(self.pred_iou_rgb_values)), self.pred_iou_rgb_values, 10))(x_values), label='Trendline RGB', color='red')
        plt.plot(x_values, np.poly1d(np.polyfit(range(len(self.pred_iou_depth_values)), self.pred_iou_depth_values, 10))(x_values), label='Trendline RGB + Depth', color='blue')
        plt.title('Mean IOU over dataset')
        # Customize x axis ticks SATURATION
        tick_values = np.arange(0, max_values, max_values / 8)
        tick_labels = np.arange(-75, 101, 25)
        plt.xticks(tick_values, tick_labels)
        plt.xlabel('Saturation')

        # Customize x axis ticks ROTATION
        # tick_values = np.arange(0, max_values, max_values / 19)
        # tick_labels = np.arange(-90, 91, 10)
        # plt.xticks(tick_values, tick_labels)
        # plt.xlabel('Rotation angle (degrees)')
        
        plt.ylabel('IOU')
        plt.legend()
        plt.show()

    def statistics(self, differences):
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
                outside_95.append(i)

        print("Classes that have a difference outside of 95% confidence interval: ", outside_95)

        return outside_95
    
    def set_config_if_dataset_specified(self, config, dataset_location):
        config.root_dir = dataset_location
        config.dataset_path = os.path.join(config.root_dir, config.dataset_name)
        config.rgb_root_folder = os.path.join(config.dataset_path, 'RGB')
        config.gt_root_folder = os.path.join(config.dataset_path, 'labels')
        config.x_root_folder = os.path.join(config.dataset_path, 'Depth')
        config.train_source = os.path.join(config.dataset_path, "train.txt")
        config.eval_source = os.path.join(config.dataset_path, "test.txt")
        return config

    @torch.no_grad()
    def create_and_plot_predictions(self, args, predictions_of_interest):
        # Get Correct depth path after Depth is changed to black
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss(reduction='mean')
        BatchNorm2d = nn.BatchNorm2d

        self.config_a = EasyDict(self.config.copy())
        self.config_a.x_channels = 3
        self.config_a.x_e_channels = 3
        model_a=segmodel(cfg=self.config_a, criterion=criterion, norm_layer=BatchNorm2d, single_GPU = True)
        print("Loaded rgb-grayscale model: ", args.model_a_path)
        model_a.load_state_dict(torch.load(args.model_a_path)["model"])
        model_a.to(device)
        self.model_a = model_a

        self.config_b = EasyDict(self.config.copy())
        self.config_b.x_channels = 3
        self.config_b.x_e_channels = 1
        model_b=segmodel(cfg=self.config_b, criterion=criterion, norm_layer=BatchNorm2d, single_GPU = True)
        print("Loaded rgb-depth model: ", args.model_b_path)
        model_b.load_state_dict(torch.load(args.model_b_path)["model"])
        model_b.to(device)
        self.model_b = model_b

        if args.test_mode:
            parent_dir = os.path.dirname(args.dir_dataset)
            self.set_config_if_dataset_specified(self.config, parent_dir)

        self.config.val_batch_size = 1
        self.config.num_workers = 0
        self.config.x_root_folder = os.path.join(self.config.dataset_path, "Depth_original")
        val_loader_depth, _ = get_val_loader(None, RGBXDataset, self.config, 1)

        self.config.x_root_folder = os.path.join(self.config.dataset_path, "Depth")
        val_loader_gray, _ = get_val_loader(None, RGBXDataset, self.config, 1)
    
        model_a.eval()
        model_b.eval()

        for i, b_depth, b_gray in tqdm(zip(range(len(val_loader_depth)), val_loader_depth, val_loader_gray)):
            if i not in predictions_of_interest:
                continue

            args.rgb_score = self.pred_iou_rgb_values[i]
            args.depth_score = self.pred_iou_depth_values[i]

            depth_image = b_depth["data"].to(device)
            modal_xs_depth = b_depth["modal_x"].to(device)
            args.depth_image = depth_image
            args.modal_xs_depth = modal_xs_depth

            gray_image = b_gray["data"].to(device)
            modal_xs_gray = b_gray["modal_x"].to(device)
            args.gray_image = gray_image
            args.modal_xs_gray = modal_xs_gray

            logits_a = model_a(gray_image, modal_xs_gray)
            logits_b = model_b(depth_image, modal_xs_depth)

            self.plot_prediction(logits_a.argmax(dim=1).cpu().numpy(), logits_b.argmax(dim=1).cpu().numpy(), i, args)

    def plot_prediction(self, prediction_a, prediction_b, index, args):
        dir_dataset = args.dir_dataset

        target_format = os.path.join(dir_dataset, "labels", "test_{}.png")
        depth_format = os.path.join(dir_dataset, "Depth_original", "test_{}.png")
        grayscale_format = os.path.join(dir_dataset, "Depth", "test_{}.png")
        img_format = os.path.join(dir_dataset, "RGB", "test_{}.png")

        save_dir = os.path.join(dir_dataset, "predictions")
        save_format = os.path.join(save_dir, "pred_test_{}.png")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        length_predictions = prediction_a.shape[0]

        # Load the target and prediction as numpy files
        for i in range(0, length_predictions):
            # Convert the prediction to RGB
            pred_rgb_grayscale = prediction_a[i]
            pred_rgb_depth = prediction_b[i]
            
            # Load the target and depth and rgb image
            target = plt.imread(target_format.format(i+index))
            depth = plt.imread(depth_format.format(i+index))
            grayscale = plt.imread(grayscale_format.format(i+index))
            img = plt.imread(img_format.format(i+index))

            target = (target * 255).astype(np.uint8)
            depth = (depth * 255).astype(np.uint8)
            grayscale = (grayscale * 255).astype(np.uint8)
            img = (img * 255).astype(np.uint8)

            pixel_acc_gray = round(np.mean(target == pred_rgb_grayscale) * 100, 2)
            pixel_acc_depth = round(np.mean(target == pred_rgb_depth) * 100, 2)

            # Plot the target, predictions and depth and rgb image
            # Also add the class names in legend
            fig, ax = plt.subplots(4, 3, figsize=(20, 20))
            target_im = ax[0][0].imshow(target, cmap='viridis', vmin=0, vmax=self.num_classes - 1)
            ax[0][0].set_title("target")
            ax[0][1].imshow(pred_rgb_grayscale, cmap='viridis', vmin=0, vmax=self.num_classes - 1)
            ax[0][1].set_title(f"pred_rgb with mIOU: {str(args.rgb_score)}, and pixel acc: {str(pixel_acc_gray)}")
            ax[0][2].imshow(pred_rgb_depth, cmap='viridis', vmin=0, vmax=self.num_classes - 1)
            ax[0][2].set_title(f"pred_rgb_depth with mIOU: {str(args.depth_score)}, and pixel acc: {str(pixel_acc_depth)}")

            # get the colors of the values, according to the 
            values = np.unique(target.ravel())
            
            # Convert values to integers
            values = values.astype(int)
            
            # colormap used by imshow
            colors = [target_im.cmap(target_im.norm(value)) for value in values]
            # create a patch (proxy artist) for every color 
            patches = [mpatches.Patch(color=colors[j], label=f"{values[j]} " + self.class_names[values[j]]) 
                       for j in range(len(values)) if values[j] < self.num_classes]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.125, 1), loc='center left', borderaxespad=0.)

            ax[1][0].imshow(img)
            ax[1][0].set_title("rgb")
            ax[1][1].imshow(grayscale, cmap='gray')
            ax[1][1].set_title("grayscale")
            ax[1][2].imshow(depth, cmap='gray')
            ax[1][2].set_title("depth")

            border_map = self.get_border_map(target, max_value=self.num_classes)
            border_mask_gray, percent_edges_correct_gray = self.evaluate_edges(border_map, target == pred_rgb_grayscale)
            border_mask_depth, percent_edges_correct_depth = self.evaluate_edges(border_map, target == pred_rgb_depth)
            rounded_percent_edges_correct_gray = round(percent_edges_correct_gray, 2)
            rounded_percent_edges_correct_depth = round(percent_edges_correct_depth, 2)

            # Plot the difference between the predictions
            difference = pred_rgb_depth - pred_rgb_grayscale
            ax[2][0].imshow(difference, cmap='viridis', vmin=-1, vmax=1)
            ax[2][0].set_title("Difference")
            ax[2][1].imshow(target == pred_rgb_grayscale, cmap='gray')
            ax[2][1].imshow(border_mask_gray, cmap='Greens', alpha=0.2)
            ax[2][1].set_title(f"Correct grayscale, with {rounded_percent_edges_correct_gray}% correct edges")
            ax[2][2].imshow(target == pred_rgb_depth, cmap='gray')
            ax[2][2].imshow(border_mask_depth, cmap='Greens', alpha=0.2)
            ax[2][2].set_title(f"Correct depth, with {rounded_percent_edges_correct_depth}% correct edges")

            # Plot the gradcam of the predictions
            with torch.enable_grad():
                dformer_analyzer_a = DFormerAnalyzer(self.model_a, cfg=self.config_a)
                dformer_analyzer_b = DFormerAnalyzer(self.model_b, cfg=self.config_b)
                category = dformer_analyzer_a.select_category(self.classes_of_interest, target)
                if args.depth_image is not None and args.modal_xs_depth is not None:
                    # dformer_analyzer_a.generate_heatmap(args.depth_image, args.modal_xs_gray, category, pred_rgb_grayscale, index)
                    # input_tensor_a = torch.stack([args.depth_image, args.modal_xs_gray], dim=1)
                    input_tensor_a = torch.cat([args.depth_image, args.modal_xs_gray], dim=1)
                    grad_cam_a = dformer_analyzer_a.analyze(input_tensor_a, category)
                if args.gray_image is not None and args.modal_xs_gray is not None:
                    # dformer_analyzer_b.generate_heatmap(args.gray_image, args.modal_xs_depth, category, pred_rgb_depth, index)
                    # input_tensor_b = torch.stack([args.gray_image, args.modal_xs_depth], dim=1)
                    input_tensor_b = torch.cat([args.gray_image, args.modal_xs_depth], dim=1)
                    grad_cam_b = dformer_analyzer_b.analyze(input_tensor_b, category)

            ax[3][0].imshow(grad_cam_a, cmap='viridis')
            ax[3][0].set_title("Gradcam grayscale")
            ax[3][1].imshow(grad_cam_b, cmap='viridis')
            ax[3][1].set_title("Gradcam depth")
            ax[3][2].imshow(np.float32(target == category), cmap='viridis')
            ax[3][2].set_title(f"Target category: {str(self.class_names[category])}")


            fig.subplots_adjust(
                left=0.0,
                right=0.9,
                bottom=0.04,
                top=0.96,
                wspace=0,
                hspace=0.15
            )
            
            plt.savefig(save_format.format(i+index))
            plt.close()

    def plot_predictions(self, predictions_dir_a, predictions_dir_b, args):
        # Load the target and prediction as numpy files
        for i in range(0, 100):
            # Convert the prediction to RGB
            pred_rgb_black = np.load(os.path.join(predictions_dir_a, f"pred_test_{i}.npy"))
            pred_rgb_depth = np.load(os.path.join(predictions_dir_b, f"pred_test_{i}.npy"))
            
            self.plot_prediction(pred_rgb_black, pred_rgb_depth, i, args)

    def get_border_map(self, segmentation_label, min_value=0, max_value=255):
        """
        Get the edges of the objects in the image.

        Parameters:
            segmentation_label (numpy.ndarray): Binary map of pixels labeled as correct or wrong.
        
        Returns:
            numpy.ndarray: Binary map of pixels along the edges of objects.
        """
        edges_before = cv2.Canny(segmentation_label, min_value, max_value)  # Apply Canny edge detection
        foreground_mask = segmentation_label != 0
        edges = edges_before & foreground_mask
    
        return edges

    def evaluate_edges(self, border_map, ground_truth, radius=2):
        """
        Calculate the accuracy of predicted borders.
        
        Parameters:
            border_map (numpy.ndarray): Binary map of pixels along the edges of objects.
            ground_truth (numpy.ndarray): Binary map of pixels labeled as correct or wrong.
            radius (int): Radius to widen the border.
        
        Returns:
            float: Percentage of correct pixels within the widened border.
        """
        border_indices = np.argwhere(border_map == 1)  # Get indices of border pixels
        
        # Create a mask by widening the border pixels
        border_mask = np.zeros_like(border_map)
        for idx in border_indices:
            x, y = idx
            border_mask[max(0, x-radius):min(border_map.shape[0], x+radius+1), 
                        max(0, y-radius):min(border_map.shape[1], y+radius+1)] = 1
            
        total_mask_pixels = np.sum(border_mask)  # Calculate the total number of border pixels
        
        # Calculate the intersection of the widened border and correct pixels
        intersection = np.sum(border_mask * ground_truth)
        
        # Calculate the percentage of correct pixels within the widened border
        border_accuracy = (intersection / total_mask_pixels) * 100
        
        return border_mask, border_accuracy

def main(args):
    result_analyzer = ResultAnalyzer(args)
    
    class_difference = np.array(result_analyzer.class_iou_rgb_values) - np.array(result_analyzer.class_iou_depth_values)
    if not args.test_mode:
        result_analyzer.plot_class_values(class_difference)
        result_analyzer.plot_ious_over_epochs()
        result_analyzer.plot_ious_over_dataset()

    prediction_difference = np.array(result_analyzer.pred_iou_depth_values) - np.array(result_analyzer.pred_iou_rgb_values)
    predictions_of_interest = result_analyzer.statistics(prediction_difference)

    result_analyzer.create_and_plot_predictions(args, predictions_of_interest)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_a_path', type=str, help='Path to the RGB model directory', default=None)
    argparser.add_argument('--model_b_path', type=str, help='Path to the RGBD model directory', default=None)
    argparser.add_argument('--dir_dataset', type=str, help='Path to the dataset directory')
    argparser.add_argument('--config', type=str, default='config.SynthDet_default_Segmodel')
    argparser.add_argument('--test_mode', action='store_true', help='Only test the predictions')
    args = argparser.parse_args()
    
    main(args)
