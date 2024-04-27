import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def add_edge_enhancement(rgb_image, depth_map):
    edges_before = cv2.Canny(depth_map, 50, 150)
    foreground_mask = depth_map < 200
    edges = edges_before & foreground_mask
    edges *= 255
  
    return np.maximum(rgb_image, edges[:, :, np.newaxis])



def main():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to DFormer dataset')
    parser.add_argument('dataset_root', help='Path to dataset')
    args = parser.parse_args()
    
    rgb_folder = os.path.join(args.dataset_root, 'RGB')
    depth_folder = os.path.join(args.dataset_root, 'Depth')
    rgb_edges_folder = os.path.join(args.dataset_root, 'Enhanced')
    os.makedirs(rgb_edges_folder, exist_ok=True)
    
    for (rgb_filename, depth_filename) in tqdm(zip(sorted(os.listdir(rgb_folder)), sorted(os.listdir(depth_folder)))):
        rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_filename))
        depth_map = cv2.imread(os.path.join(depth_folder, depth_filename), cv2.IMREAD_GRAYSCALE)
        
        enhanced_image = add_edge_enhancement(rgb_image, depth_map)
        
        cv2.imwrite(os.path.join(rgb_edges_folder, rgb_filename), enhanced_image)
        
        
    # Rename RGB folder to RGB_original
    os.rename(rgb_folder, os.path.join(args.dataset_root, 'RGB_original'))

    # Rename RGB_edges folder to RGB
    os.rename(rgb_edges_folder, os.path.join(args.dataset_root, 'RGB'))

if __name__ == '__main__':
    main()