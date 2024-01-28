import numpy as np
import torch
from torch.utils import data

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
    train_preprocess = Pre(config.norm_mean, config.norm_std, config.x_is_single_channel,config)

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