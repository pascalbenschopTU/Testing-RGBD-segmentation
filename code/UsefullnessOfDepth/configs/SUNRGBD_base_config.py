import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = 'datasets'
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'SUNRGBD'
C.dataset_path = osp.join(C.root_dir, 'SUNRGBD')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'labels')
C.gt_format = '.png'
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, 'Depth')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.x_channels = 3
C.x_e_channels = 1
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True


C.num_train_imgs = 200
C.num_eval_imgs = 500

C.classes = 'SUNRGBD'
C.num_classes = 38
C.class_names =  ['background', 'wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']


"""Image Config"""
# Background class is 0, set to -1 to ignore
C.background = -1
C.image_height = 400
C.image_width = 400
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.model = 'CMX'
C.backbone = 'mit_b2'
C.pretrained_model = 'checkpoints/pretrained/mit_b2.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.start_epoch = 1
C.lr = 0.0002228292059834231
C.lr_power = 0.8723816185315307
C.momentum = 0.9323823925465781
C.weight_decay = 0.005905567108732078
C.lamda = 1e-6
C.batch_size = 4
C.val_batch_size = 16
C.nepochs = 40
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 1
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate=0.1
C.aux_rate =0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.5,0.75,1,1.25,1.5] # [0.75, 1, 1.25] # 0.5,0.75,1,1.25,1.5
C.eval_flip =  True # False #
C.eval_crop_size = [480, 480] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 9 # 200
C.checkpoint_step = 10

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.checkpoint_dir = 'checkpoints_fgbg\\checkpoints_fgbg_spacing_CMXb2'
C.log_dir = osp.abspath(osp.join(C.checkpoint_dir, C.dataset_name + '_' + C.backbone))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir

if __name__ == '__main__':
    print(config.nepochs)

C.date_time = '20240612-121117'
