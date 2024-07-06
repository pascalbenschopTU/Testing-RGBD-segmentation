from inspect import signature
from torch import nn
import torch.nn.functional as F
import torch
import sys

sys.path.append('../UsefullnessOfDepth')
# Models
from model_DFormer.builder import EncoderDecoder as DFormer
from models_CMX.builder import EncoderDecoder as CMXmodel
from model_pytorch_deeplab_xception.deeplab import DeepLab
from models_segformer import SegFormer
from model_TokenFusion.segformer import WeTr as TokenFusion
from models_Gemini.segformer import WeTr as Gemini
from model_HIDANet.model import HiDANet as HIDANet

from utils.init_func import group_weight

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
            self.model_name = "DFormer_Tiny"
        self.set_model()

    def set_model(self):
        if self.model_name == "DFormer":
            self.model = DFormer(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "DeepLab":
            self.model = DeepLab(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "TokenFusion":
            self.is_token_fusion = True
            self.model = TokenFusion(cfg=self.config, pretrained=self.pretrained)
            self.params_list = self.model.get_param_groups(self.config.lr, self.config.weight_decay)
        elif self.model_name == "Gemini":
            self.is_token_fusion = True
            self.model = Gemini(cfg=self.config, pretrained=self.pretrained)
            self.params_list = self.model.get_param_groups(self.config.lr, self.config.weight_decay)
        elif self.model_name == "SegFormer":
            self.model = SegFormer(backbone=self.backbone, cfg=self.config, criterion=self.criterion)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
            if self.pretrained:
                self.model.init_pretrained(self.config.pretrained_model)
        elif self.model_name == "CMX":
            self.model = CMXmodel(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "HIDANet":
            self.model = HIDANet()
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
    

    def get_loss(self, output, target, criterion):
        if self.config.get('use_aux', False):
            foreground_mask = target != self.config.background
            label = (foreground_mask > 0).long()
            loss = criterion(output[0], target.long()) + self.config.aux_rate * criterion(output[1], label)
            output = output[0]
        else:
            if self.is_token_fusion and isinstance(output, list):  # Output of TokenFusion
                output, masks = output
                loss = 0
                for out in output:
                    soft_output = F.log_softmax(out, dim=1)
                    loss += criterion(soft_output, target)

                if self.config.lamda > 0 and masks is not None:
                    L1_loss = 0
                    for mask in masks:
                        L1_loss += sum([torch.abs(m).sum().cuda() for m in mask])
                    loss += self.config.lamda * L1_loss
            elif self.model_name == "HIDANet":
                loss = self.model.calculate_loss(output, target)
            else:
                loss = criterion(output, target.long())

        return loss
