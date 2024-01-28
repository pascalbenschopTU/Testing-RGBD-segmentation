import argparse
import os
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Class names from the config file
# class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']
class_names = ['background', 'book_dorkdiaries_aladdin', 'candy_minipralines_lindt', 'candy_raffaello_confetteria', 'cereal_capn_crunch', 'cereal_cheerios_honeynut', 'cereal_corn_flakes', 'cereal_cracklinoatbran_kelloggs', 'cereal_oatmealsquares_quaker', 'cereal_puffins_barbaras', 'cereal_raisin_bran', 'cereal_rice_krispies', 'chips_gardensalsa_sunchips', 'chips_sourcream_lays', 'cleaning_freegentle_tide', 'cleaning_snuggle_henkel', 'cracker_honeymaid_nabisco', 'cracker_lightrye_wasa', 'cracker_triscuit_avocado', 'cracker_zwieback_brandt', 'craft_yarn_caron', 'drink_adrenaline_shock', 'drink_coffeebeans_kickinghorse', 'drink_greentea_itoen', 'drink_orangejuice_minutemaid', 'drink_whippingcream_lucerne', 'footware_slippers_disney', 'hygiene_poise_pads', 'lotion_essentially_nivea', 'lotion_vanilla_nivea', 'pasta_lasagne_barilla', 'pest_antbaits_terro', 'porridge_grits_quaker', 'seasoning_canesugar_candh', 'snack_biscotti_ghiott', 'snack_breadsticks_nutella', 'snack_chips_pringles', 'snack_coffeecakes_hostess', 'snack_cookie_famousamos', 'snack_cookie_petitecolier', 'snack_cookie_quadratini', 'snack_cookie_waffeletten', 'snack_cookie_walkers', 'snack_cookies_fourre', 'snack_granolabar_kashi', 'snack_granolabar_kind', 'snack_granolabar_naturevalley', 'snack_granolabar_quaker', 'snack_salame_hillshire', 'soup_chickenenchilada_progresso', 'soup_tomato_pacific', 'storage_ziploc_sandwich', 'toiletry_tissue_softly', 'toiletry_toothpaste_colgate', 'toy_cat_melissa', 'utensil_candle_decorators', 'utensil_coffee_filters', 'utensil_cottonovals_signaturecare', 'utensil_papertowels_valuecorner', 'utensil_toiletpaper_scott', 'utensil_trashbag_valuecorner', 'vitamin_centrumsilver_adults', 'vitamin_centrumsilver_men', 'vitamin_centrumsilver_woman']

classes = len(class_names)

def plot_predictions(dir_rgb, dir_rgbd, dir_dataset):

    # Load the target and prediction as numpy files
    pred_rgb_black_format = os.path.join(dir_rgb, "predictions", "pred_test_{}.npy")
    pred_rgb_depth_format = os.path.join(dir_rgbd, "predictions", "pred_{}.npy")

    target_format = os.path.join(dir_dataset, "labels", "test_{}.png")
    depth_format = os.path.join(dir_dataset, "Depth", "test_{}.png")
    img_format = os.path.join(dir_dataset, "RGB", "test_{}.png")

    length_predictions = len(os.listdir(dir_rgb + "/predictions"))

    # Load the target and prediction as numpy files
    for i in range(0, length_predictions):
        pred_rgb_black = np.load(pred_rgb_black_format.format(i), allow_pickle=True)
        pred_rgb_depth = np.load(pred_rgb_depth_format.format(i), allow_pickle=True)

        # Add another dimension to pred_rgb_depth
        pred_rgb_depth = np.expand_dims(pred_rgb_depth, axis=0)

        # Convert the prediction to RGB
        pred_rgb_black = convert_prediction_to_rgb(pred_rgb_black)
        pred_rgb_depth = convert_prediction_to_rgb(pred_rgb_depth)

        # Add 1 to all values to avoid black in the legend
        pred_rgb_black[pred_rgb_black != 0] += 1
        pred_rgb_depth[pred_rgb_depth != 0] += 1
        
        # Load the target and depth and rgb image
        target = plt.imread(target_format.format(i))
        depth = plt.imread(depth_format.format(i))
        img = plt.imread(img_format.format(i))

        target = (target * 255).astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)
        img = (img * 255).astype(np.uint8)

        # Plot the target, predictions and depth and rgb image
        # Also add the class names in legend
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        cmap = plt.cm.get_cmap('viridis', classes)
        target_im = ax[0][0].imshow(target, cmap=cmap, vmin=0, vmax=classes - 1)
        ax[0][0].set_title("target")
        rgb_black = ax[0][1].imshow(pred_rgb_black, cmap=cmap, vmin=0, vmax=classes - 1)
        ax[0][1].set_title("pred_rgb_black")
        rgb_depth = ax[0][2].imshow(pred_rgb_depth, cmap=cmap, vmin=0, vmax=classes - 1)
        ax[0][2].set_title("pred_rgb_depth")

        # get the colors of the values, according to the 
        values = np.unique(target.ravel())
        values = np.concatenate((values, np.unique(pred_rgb_black.ravel())))
        # values = np.concatenate((values, np.unique(pred_rgb_depth.ravel())))
        values = np.unique(values)
        values = values[1:]

        # print(len(values))
        
        # colormap used by imshow
        colors = [target_im.cmap(target_im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [mpatches.Patch(color=colors[j], label=f"{values[j]} " + class_names[values[j]]) for j in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)

        ax[1][0].imshow(depth)
        ax[1][0].set_title("depth")
        ax[1][1].imshow(img)
        ax[1][1].set_title("rgb")

        plt.show()

def convert_prediction_to_rgb(prediction, colormap=None):
    # Ensure prediction is a NumPy array
    prediction_np = np.array(prediction)

    # Extract the class indices (assuming the class dimension is the second dimension)
    class_indices = np.argmax(prediction_np, axis=1)

    # Remove the first channel
    return class_indices.squeeze()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir_rgb', type=str, help='dir of model a')
    argparser.add_argument('--dir_rgbd', type=str, help='dir of model b')
    argparser.add_argument('--dir_dataset', type=str, help='dir of dataset')
    args = argparser.parse_args()

    plot_predictions(args.dir_rgb, args.dir_rgbd, args.dir_dataset)