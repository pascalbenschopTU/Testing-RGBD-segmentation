import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch_deeplab_xception.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model_pytorch_deeplab_xception.aspp import build_aspp
from model_pytorch_deeplab_xception.decoder import build_decoder
from model_pytorch_deeplab_xception.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, cfg=None, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False, criterion=nn.CrossEntropyLoss(reduction='none'),
                 norm_layer=nn.BatchNorm2d):
        super(DeepLab, self).__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes
        backbone = cfg.backbone
        self.criterion = criterion
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = SynchronizedBatchNorm2d if sync_bn else norm_layer
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, cfg=cfg)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


