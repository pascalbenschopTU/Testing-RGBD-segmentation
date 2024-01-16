import numpy as np
from torchvision.datasets import CocoDetection
from PIL import Image
import os
from pycocotools import mask as coco_mask
import torch

class CocoDepthDataset(CocoDetection):
    def __init__(self, root, annFile, depth_folder, transform=None, target_transform=None):
        super(CocoDepthDataset, self).__init__(root, annFile, transform, target_transform)
        self.depth_folder = depth_folder

    def __getitem__(self, index):
        img, target = super(CocoDepthDataset, self).__getitem__(index)

        masks = []
        class_values = []
        for obj in target:
            rle = obj['segmentation']
            binary_mask = coco_mask.decode(rle)
            binary_mask = torch.from_numpy(binary_mask)
            masks.append(binary_mask)

            class_value = obj['category_id']
            class_values.append(class_value)

        
        ann = np.zeros((masks[0].shape[0], masks[0].shape[1]))

        for i in range(len(masks)):
            ann[masks[i] == 1] = class_values[i]

        # Convert ann to Image
        ann = Image.fromarray(ann)

        # Load depth image
        depth_path = os.path.join(self.depth_folder, f"depth_{index}.png")
        depth = Image.open(depth_path).convert('L')  # Assuming depth is stored as grayscale image

        if self.transform is not None:
            depth = self.transform(depth)
            ann = self.transform(ann)

        return img, depth, ann
    


