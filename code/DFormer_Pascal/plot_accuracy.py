import argparse
import matplotlib.pyplot as plt
import re
import numpy as np

# Class names from the config file
# class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']
class_names = ['background', 'book_dorkdiaries_aladdin', 'candy_minipralines_lindt', 'candy_raffaello_confetteria', 'cereal_capn_crunch', 'cereal_cheerios_honeynut', 'cereal_corn_flakes', 'cereal_cracklinoatbran_kelloggs', 'cereal_oatmealsquares_quaker', 'cereal_puffins_barbaras', 'cereal_raisin_bran', 'cereal_rice_krispies', 'chips_gardensalsa_sunchips', 'chips_sourcream_lays', 'cleaning_freegentle_tide', 'cleaning_snuggle_henkel', 'cracker_honeymaid_nabisco', 'cracker_lightrye_wasa', 'cracker_triscuit_avocado', 'cracker_zwieback_brandt', 'craft_yarn_caron', 'drink_adrenaline_shock', 'drink_coffeebeans_kickinghorse', 'drink_greentea_itoen', 'drink_orangejuice_minutemaid', 'drink_whippingcream_lucerne', 'footware_slippers_disney', 'hygiene_poise_pads', 'lotion_essentially_nivea', 'lotion_vanilla_nivea', 'pasta_lasagne_barilla', 'pest_antbaits_terro', 'porridge_grits_quaker', 'seasoning_canesugar_candh', 'snack_biscotti_ghiott', 'snack_breadsticks_nutella', 'snack_chips_pringles', 'snack_coffeecakes_hostess', 'snack_cookie_famousamos', 'snack_cookie_petitecolier', 'snack_cookie_quadratini', 'snack_cookie_waffeletten', 'snack_cookie_walkers', 'snack_cookies_fourre', 'snack_granolabar_kashi', 'snack_granolabar_kind', 'snack_granolabar_naturevalley', 'snack_granolabar_quaker', 'snack_salame_hillshire', 'soup_chickenenchilada_progresso', 'soup_tomato_pacific', 'storage_ziploc_sandwich', 'toiletry_tissue_softly', 'toiletry_toothpaste_colgate', 'toy_cat_melissa', 'utensil_candle_decorators', 'utensil_coffee_filters', 'utensil_cottonovals_signaturecare', 'utensil_papertowels_valuecorner', 'utensil_toiletpaper_scott', 'utensil_trashbag_valuecorner', 'vitamin_centrumsilver_adults', 'vitamin_centrumsilver_men', 'vitamin_centrumsilver_woman']

max_length = max(len(class_name) for class_name in class_names)
class_names = [class_name.ljust(max_length) for class_name in class_names]

def plot_accuracy(dir_rgb, dir_rgbd):
    acc_rgb_text_location = dir_rgb + r"\results.txt"
    acc_depth_text_location = dir_rgbd + r"\results.txt"

    # Accuracy values from the text file
    acc_rgb_values_text = open(acc_rgb_text_location).read()
    acc_depth_values_text = open(acc_depth_text_location).read()

    # Parse accuracy values
    acc_rgb_values = {class_name: [] for class_name in class_names}
    for line in acc_rgb_values_text.split("\n")[5:]:
        if line.startswith("acc:"):
            line = line.split("],")[0]
            values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            for class_name, value in zip(class_names, values):
                acc_rgb_values[class_name].append(value)

    acc_depth_values = {class_name: [] for class_name in class_names}
    for line in acc_depth_values_text.split("\n")[5:]:
        if line.startswith("acc"):
            line = line.split("],")[0]
            values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            for class_name, value in zip(class_names, values):
                acc_depth_values[class_name].append(value)

    
    

    rgb_values = list(acc_rgb_values.values())
    rgb_values = [value[0] for value in rgb_values]

    depth_values = list(acc_depth_values.values())
    depth_values = [value[0] for value in depth_values]

    difference = np.array(rgb_values) - np.array(depth_values)
    sorted_indices = np.argsort(difference)

    sorted_rgb_values = np.array(rgb_values)[sorted_indices]
    sorted_depth_values = np.array(depth_values)[sorted_indices]
    sorted_class_names = np.array(class_names)[sorted_indices]

    # Generate histograms
    plt.figure(figsize=(20, 10))
    x = range(len(sorted_class_names))
    plt.xticks(x, sorted_class_names, rotation=300)

    plt.bar(np.array(x) - 0.2, sorted_rgb_values, label='RGB + Black', color='#FF69B4', width=0.4)
    plt.bar(np.array(x) + 0.2, sorted_depth_values, label='RGB + Depth', color='green', width=0.4)

    plt.title('Accuracy values for all classes over epochs')
    plt.xlabel('Class Names')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir_rgb', type=str, help='Path to the RGB model directory')
    argparser.add_argument('--dir_rgbd', type=str, help='Path to the RGBD model directory')
    args = argparser.parse_args()
    
    plot_accuracy(args.dir_rgb, args.dir_rgbd)
