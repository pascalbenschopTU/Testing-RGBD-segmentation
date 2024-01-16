import argparse
import os
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Class names from the config file
# class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']
class_names = ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al',
                'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax',
                'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj',
                'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv',
                'bw', 'bx', 'by', 'bz', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl']

def plot_predictions(dir_rgb, dir_rgbd, dataset_dir):

    # Load the target and prediction as numpy files
    pred_rgb_black_format = os.path.join(dir_rgb, "predictions", "pred_test_{}.npy")
    pred_rgb_depth_format = os.path.join(dir_rgbd, "predictions", "pred_test_{}.npy")

    target_format = os.path.join(dataset_dir, "labels", "test_{}.png")
    depth_format = os.path.join(dataset_dir, "Depth", "test_{}.png")
    img_format = os.path.join(dataset_dir, "RGB", "test_{}.png")

    length_predictions = len(os.listdir(dir_rgb + "/predictions"))

    # Load the target and prediction as numpy files
    for i in range(0, length_predictions):
        pred_rgb_black = np.load(pred_rgb_black_format.format(i), allow_pickle=True)
        pred_rgb_depth = np.load(pred_rgb_depth_format.format(i), allow_pickle=True)

        # Convert the prediction to RGB
        pred_rgb_black = convert_prediction_to_rgb(pred_rgb_black)
        pred_rgb_depth = convert_prediction_to_rgb(pred_rgb_depth)
        
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
        cmap = plt.cm.get_cmap('viridis', 40 + 1)
        target_im = ax[0][0].imshow(target, cmap=cmap, vmin=0, vmax=40)
        ax[0][0].set_title("target")
        rgb_black = ax[0][1].imshow(pred_rgb_black, cmap=cmap, vmin=0, vmax=40)
        ax[0][1].set_title("pred_rgb_black")
        rgb_depth = ax[0][2].imshow(pred_rgb_depth, cmap=cmap, vmin=0, vmax=40)
        ax[0][2].set_title("pred_rgb_depth")

        # get the colors of the values, according to the 
        values = np.unique(target.ravel())
        values = values[1:]
        
        # colormap used by imshow
        colors = [target_im.cmap(target_im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [mpatches.Patch(color=colors[i], label=f"{values[i]} " + class_names[values[i]]) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

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
    argparser.add_argument('--dataset_dir', type=str, help='dir of dataset')
    args = argparser.parse_args()

    plot_predictions(args.dir_rgb, args.dir_rgbd, args.dataset_dir)