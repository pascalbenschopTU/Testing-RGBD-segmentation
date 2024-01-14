import matplotlib.pyplot as plt
import re
import numpy as np

# Class names from the config file
class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']

acc_rgb_text_location = r"checkpoints\SUNRGBD_DFormer-Tiny_20240102-134605\results.txt"
acc_depth_text_location = r"checkpoints\SUNRGBD_DFormer-Tiny_20231230-120203\results.txt"

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

# Generate histograms
plt.figure()
x = range(len(class_names))
plt.xticks(x, class_names, rotation=300)


rgb_values = list(acc_rgb_values.values())
rgb_values = [value[0] for value in rgb_values]
plt.bar(np.array(x) - 0.2, rgb_values, label='RGB + Black', color='#FF69B4', width=0.4)

depth_values = list(acc_depth_values.values())
depth_values = [value[0] for value in depth_values]
plt.bar(np.array(x) + 0.2, depth_values, label='RGB + Depth', color='green', width=0.4)

plt.title('Accuracy values for all classes over epochs')
plt.xlabel('Class Names')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
