import os
import re
import numpy as np


def get_results_from_file(model_dir_directory, results_file_name="results.txt"):
        acc_rgb_text_location = os.path.join(model_dir_directory, results_file_name)

        # Accuracy values from the text file
        acc_rgb_values_text = open(acc_rgb_text_location).read()

        miou_start_search = "mious:"
        iou_std_start_search = "iou_stds:"

        float_regex = r"[-+]?\d*\.\d+|\d+"

        # Parse accuracy values from the text file
        miou_values = []
        iou_std_values = []
        for i, line in enumerate(acc_rgb_values_text.split("\n")):
            if miou_start_search in line and iou_std_start_search in line:
                line_1 = line.split(miou_start_search)[1].split("],")[0]
                miou_values.append([float(x) for x in re.findall(float_regex, line_1)])
                line_2 = line.split(iou_std_start_search)[1].split("],")[0]
                iou_std_values.append([float(x) for x in re.findall(float_regex, line_2)])

        return miou_values, iou_std_values