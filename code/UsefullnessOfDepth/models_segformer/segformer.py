import torch
from torch import Tensor
from torch.nn import functional as F
from models_segformer.base import BaseModel
from models_segformer.heads import SegFormerHead


class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', cfg=None, criterion=None) -> None:
        num_classes = cfg.num_classes
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)
        self.criterion = criterion
        self.cfg = cfg

    def forward2(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y
    
    def forward(self, x: Tensor, depth: Tensor=None, label: Tensor=None) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape

        if label is not None:
            loss = self.criterion(y, label)[label != self.cfg.background].mean()
            return loss
        
        return y


if __name__ == '__main__':
    model = SegFormer('MiT-B0')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)