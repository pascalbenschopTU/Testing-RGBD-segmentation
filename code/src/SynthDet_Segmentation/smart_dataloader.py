import numpy as np
from torch.utils import data
import cv2
import random
import sys
sys.path.append('../../DFormer/utils/')

from transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std,sign=False,config=None):
        self.config=config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign =sign

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        # modal_x = cv2.bilateralFilter(modal_x, 9, 75, 75)
            
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = normalize(modal_x, [0.48,0.48,0.48], [0.28,0.28,0.28])#[0.5,0.5,0.5]
        else:
            modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_gt, p_modal_x

class Pre(object):
    def __init__(self, norm_mean, norm_std,sign=False,config=None):
        self.config=config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign =sign

    def normalize(self, img, mean, std):
        # pytorch pretrained model need the input range: 0-1
        img = img.astype(np.float64) / 255.0
        img = img - mean
        img = img / std
        return img

    def __call__(self, rgb, gt, modal_x):
        if len(rgb.shape) != 3 or len(modal_x.shape) != 3:
            raise ValueError(f"Invalid shapes: rgb={rgb.shape}, modal_x={modal_x.shape}")
        
        # Crop the images to config.image_height x config.image_width
        rgb = resize_image(rgb, (self.config.image_height, self.config.image_width))
        modal_x = resize_image(modal_x, (self.config.image_height, self.config.image_width))
        gt = resize_image(gt, (self.config.image_height, self.config.image_width))
        
        rgb = self.normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = self.normalize(modal_x, [0.48,0.48,0.48], [0.28,0.28,0.28])
        else:
            modal_x = self.normalize(modal_x, self.norm_mean, self.norm_std)

        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        return rgb, gt, modal_x

def get_train_loader(dataset,config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel,config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader


def get_val_loader(dataset,config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_preprocess = Pre(config.norm_mean, config.norm_std,config.x_is_single_channel,config)

    val_dataset = dataset(data_setting, "val", val_preprocess)

    val_sampler = None
    is_shuffle = False
    batch_size = config.val_batch_size

    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=val_sampler)

    return val_loader