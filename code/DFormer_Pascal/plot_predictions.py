import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Class names from the config file
class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']



target_format = "checkpoints/SUNRGBD_DFormer-Tiny_20240102-134605/predictions/target_{}.npy"
pred_rgb_black_format = "checkpoints/SUNRGBD_DFormer-Tiny_20240102-134605/predictions/pred_{}.npy"
pred_rgb_depth_format = "checkpoints/SUNRGBD_DFormer-Tiny_20231230-120203/predictions/pred_{}.npy"

depth_format = "datasets/SUNRGBD/Depth_original/test_{}.png"
img_format = "datasets/SUNRGBD/RGB/test_{}.jpg"

# Load the target and prediction as numpy files
for i in range(0, 100):
    target = np.load(target_format.format(i + 1), allow_pickle=True)
    pred_rgb_black = np.load(pred_rgb_black_format.format(i + 1), allow_pickle=True)
    pred_rgb_depth = np.load(pred_rgb_depth_format.format(i + 1), allow_pickle=True)

    depth = plt.imread(depth_format.format(i))
    img = plt.imread(img_format.format(i))

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