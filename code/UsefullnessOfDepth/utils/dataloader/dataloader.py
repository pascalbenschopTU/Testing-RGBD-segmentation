import cv2
import torch
import numpy as np
from torch.utils import data
import random
from .transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def resize_image(image, size):
    if not image.dtype == np.uint8 or not image.dtype == np.float32:
        image = image.astype(np.float32)
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
    def __init__(self, norm_mean, norm_std,sign=False,config=None, **kwargs):
        self.config=config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign =sign
        self.kwargs = kwargs
    
    def normalize_image_with_varying_shape(self, image):
        if image.ndim == 2:
            image = normalize(image, np.mean(self.norm_mean), np.mean(self.norm_std))
            image = np.expand_dims(image, axis=2)
        else:
            image = normalize(image, self.norm_mean, self.norm_std)

        return image
    

    def __call__(self, rgb, gt, modal_x):
        randomly_mirror_image = self.kwargs.get('random_mirror', False)
        if randomly_mirror_image:
            rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)

        random_crop_and_scale = self.kwargs.get('random_crop_and_scale', False)
        if random_crop_and_scale:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        random_black = self.kwargs.get('random_black', False)
        random_black_prob = self.kwargs.get('random_black_prob', 0.5)
        if random_black:
            if np.random.uniform() < random_black_prob:
                rgb = np.zeros_like(rgb)

        color_jitter = self.kwargs.get('random_color_jitter', False)
        min_color_jitter = self.kwargs.get('min_color_jitter', 0.7)
        max_color_jitter = self.kwargs.get('max_color_jitter', 1.3)
        if color_jitter:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            rgb = rgb.astype(np.float32)
            rgb[:, :, 0] *= np.random.uniform(min_color_jitter, max_color_jitter)
            rgb[:, :, 0] = rgb[:, :, 0] % 180
            rgb[:, :, 1] *= np.random.uniform(min_color_jitter, max_color_jitter)
            rgb[:, :, 2] *= np.random.uniform(min_color_jitter, max_color_jitter)
            rgb = np.clip(rgb, 0, 255)
            rgb = rgb.astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2RGB)

        random_noise_rgb = self.kwargs.get('random_noise_rgb', False)
        random_noise_rgb_prob = self.kwargs.get('random_noise_rgb_prob', 0.5)
        random_noise_rgb_max = self.kwargs.get('random_noise_rgb_amount', 0.1)
        if random_noise_rgb and np.random.uniform() < random_noise_rgb_prob:
            noise_std = np.random.uniform(0, random_noise_rgb_max) * (np.max(rgb) - np.min(rgb))
            rgb = rgb + np.random.normal(0, noise_std, rgb.shape)

        random_noise_modal_x = self.kwargs.get('random_noise_modal_x', False)
        random_noise_modal_x_prob = self.kwargs.get('random_noise_modal_x_prob', 0.5)
        random_noise_modal_x_max = self.kwargs.get('random_noise_modal_x_amount', 0.1)
        if random_noise_modal_x and np.random.uniform() < random_noise_modal_x_prob:
            noise_std = np.random.uniform(0, random_noise_modal_x_max) * (np.max(modal_x) - np.min(modal_x))
            modal_x = modal_x + np.random.normal(0, noise_std, modal_x.shape)

        rgb = self.normalize_image_with_varying_shape(rgb)
        modal_x = self.normalize_image_with_varying_shape(modal_x)

        if random_crop_and_scale:
            crop_size = (self.config.image_height, self.config.image_width)
            crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

            rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
            gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
            modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        elif self.kwargs.get('random_crop', False):
            # crop_size = (self.config.image_height, self.config.image_width)
            crop_size = (np.random.randint(self.config.image_height//2, self.config.image_height), 
                         np.random.randint(self.config.image_width//2, self.config.image_width))
            crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

            rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
            gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
            modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

            rgb = resize_image(rgb, (self.config.image_width, self.config.image_height))
            gt = resize_image(gt, (self.config.image_width, self.config.image_height))
            modal_x = resize_image(modal_x, (self.config.image_width, self.config.image_height))
        else:
            rgb = resize_image(rgb, (self.config.image_width, self.config.image_height))
            gt = resize_image(gt, (self.config.image_width, self.config.image_height))
            modal_x = resize_image(modal_x, (self.config.image_width, self.config.image_height))

        rgb = rgb.transpose(2, 0, 1)
        if modal_x.ndim == 2:
            modal_x = np.expand_dims(modal_x, axis=2)   
        modal_x = modal_x.transpose(2, 0, 1)

        return rgb, gt, modal_x
        

class ValPre(object):
    def __init__(self, norm_mean, norm_std,sign=False,config=None):
        self.config=config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign =sign
    def __call__(self, rgb, gt, modal_x):
        rgb = resize_image(rgb, (self.config.image_width, self.config.image_height))
        gt = resize_image(gt, (self.config.image_width, self.config.image_height))
        modal_x = resize_image(modal_x, (self.config.image_width, self.config.image_height))

        # If rgb is single channel:
        if rgb.ndim == 2:
            rgb = normalize(rgb, np.mean(self.norm_mean), np.mean(self.norm_std))
            rgb = np.expand_dims(rgb, axis=2)
        else:
            rgb = normalize(rgb, self.norm_mean, self.norm_std)

        if modal_x.ndim == 2:
            modal_x = normalize(modal_x, np.mean(self.norm_mean), np.mean(self.norm_std))
            modal_x = np.expand_dims(modal_x, axis=2)
        else:
            modal_x = normalize(modal_x, [0.48,0.48,0.48], [0.28,0.28,0.28])
        return rgb.transpose(2, 0, 1), gt, modal_x.transpose(2, 0, 1)
    

def get_train_loader(engine, dataset, config):
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
    
    train_loader_args = {
        'random_black': config.get('random_black', False),
        'random_mirror': config.get('random_mirror', False),
        'random_crop_and_scale': config.get('random_crop_and_scale', False),
        'random_crop': config.get('random_crop', False),
        'random_color_jitter': config.get('random_color_jitter', False),
    }
    
    # train_preprocess = TrainPre(config.norm_mean, config.norm_std,config.x_is_single_channel,config)
    # train_preprocess = ValPre(config.norm_mean, config.norm_std,config.x_is_single_channel,config)
    print("Using following augmentations: ", train_loader_args)
    train_preprocess = TrainPre(config.norm_mean, config.norm_std,config.x_is_single_channel,config, **train_loader_args)

    num_imgs = (config.num_train_imgs // config.batch_size + 1) * config.batch_size
    train_dataset = dataset(data_setting, "train", train_preprocess, num_imgs)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine and engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False


    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)


    return train_loader, train_sampler


def get_val_loader(engine, dataset,config,gpus=1):
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
    val_preprocess = ValPre(config.norm_mean, config.norm_std,config.x_is_single_channel,config)

    val_dataset = dataset(data_setting, "val", val_preprocess, config.num_eval_imgs)


    val_sampler = None
    is_shuffle = False
    batch_size = 1

    if engine and engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = 1
        is_shuffle = False

    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=val_sampler)

    return val_loader, val_sampler
