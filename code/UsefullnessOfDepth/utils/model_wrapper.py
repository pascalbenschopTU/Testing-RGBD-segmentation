from inspect import signature
from torch import nn
import torch.nn.functional as F
import sys
import cv2
import numpy as np

sys.path.append('../UsefullnessOfDepth')
# Models
from model_DFormer.builder import EncoderDecoder as DFormer
from model_DFormer.builder import EncoderDecoderMultiTask as DFormer_multitask
from models_CMX.builder import EncoderDecoder as CMXmodel
from model_pytorch_deeplab_xception.deeplab import DeepLab
from models_segformer import SegFormer
from model_TokenFusion.segformer import WeTr as TokenFusion
from model_DACNN.model import DCNN

from utils.init_func import group_weight



DFORMER_TINY_PRETRAINED_LOCATION = "checkpoints/pretrained/DFormer_Tiny.pth.tar"
DFORMER_SMALL_PRETRAINED_LOCATION = "checkpoints/pretrained/DFormer_Small.pth.tar"
MIT_B1_PRETRAINED_LOCATION = "checkpoints/pretrained/mit_b1.pth"
MIT_B2_PRETRAINED_LOCATION = "checkpoints/pretrained/mit_b2.pth"
MIT_B3_PRETRAINED_LOCATION = "checkpoints/pretrained/mit_b3.pth"
DEEPLAB_PRETRAINED_LOCATION = "checkpoints/pretrained/xception-b5690688.pth"

class ModelWrapper(nn.Module):
    def __init__(
            self,
            config, 
            criterion=nn.CrossEntropyLoss(reduction='mean'),
            norm_layer=nn.BatchNorm2d, 
            pretrained=True,
        ):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.backbone = config.backbone
        self.criterion = criterion
        self.norm_layer = norm_layer
        self.pretrained = pretrained
        self.pretrained_weights = config.pretrained_model
        self.is_token_fusion = False

        if hasattr(self.config, "model"):
            self.model_name = self.config.model
        else:
            self.model_name = "DFormer"
        
        self.set_model()


    def get_pretrained_weights(self):
        if self.pretrained:
            if self.pretrained_weights is not None:
                return self.pretrained_weights
            else:
                if self.model_name == "DFormer" or self.model_name == "DFormer_multitask":
                    if self.backbone == "DFormer-Tiny":
                        return DFORMER_TINY_PRETRAINED_LOCATION
                    if self.backbone == "DFormer-Small":
                        return DFORMER_SMALL_PRETRAINED_LOCATION
                elif self.model_name == "DeepLab":
                    return DEEPLAB_PRETRAINED_LOCATION
                elif self.model_name == "TokenFusion" or self.model_name == "SegFormer" or self.model_name == "CMX":
                    if self.backbone.lower() == "mit_b1" or self.backbone.lower() == "mit-b1":
                        return MIT_B1_PRETRAINED_LOCATION
                    elif self.backbone.lower() == "mit_b2" or self.backbone.lower() == "mit-b2":
                        return MIT_B2_PRETRAINED_LOCATION
                    elif self.backbone.lower() == "mit_b3" or self.backbone.lower() == "mit-b3":
                        return MIT_B3_PRETRAINED_LOCATION
                else:
                    raise ValueError("Model not found")

    def set_model(self):
        if self.model_name == "DFormer" and not self.config.get("use_aux", False):
            self.model = DFormer(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "DFormer" and self.config.get("use_aux", False):
            self.model = DFormer_multitask(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "DeepLab":
            self.model = DeepLab(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "TokenFusion":
            self.is_token_fusion = True
            self.model = TokenFusion(cfg=self.config, pretrained=self.pretrained)
            self.params_list = self.model.get_param_groups(self.config.lr, self.config.weight_decay)
        elif self.model_name == "SegFormer":
            self.model = SegFormer(backbone=self.backbone, cfg=self.config, criterion=self.criterion)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
            if self.pretrained:
                self.model.init_pretrained(self.get_pretrained_weights())
        elif self.model_name == "CMX":
            self.model = CMXmodel(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "DACNN":
            self.model = DCNN(n_classes=self.config.num_classes, learned_billinear=False)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        else:
            raise ValueError("Model not found")
        
        self.is_rgb_model = len(signature(self.model.forward).parameters) == 1
        print("Model: ", self.model_name, " BackBone: ", self.backbone, " Pretrained: ", self.pretrained, " RGB Model: ", self.is_rgb_model)
    
    def forward(self, x, x_e):
        if self.model is None:
            raise ValueError("Model not found")
        
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        if len(x_e.size()) == 3:
            x_e = x_e.unsqueeze(0)

        output = None
        
        # Check if self.model has a forward function that accepts only x or also x_e
        if self.is_rgb_model:
            output = self.model(x)
        else:
            output = self.model(x, x_e)

        if not isinstance(output, tuple) and not isinstance(output, list) and output.size()[-2:] != x.size()[-2:]:
            output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=False)

        return output
