from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class Bicubic(nn.Module):
    ''' Bicubic '''
    def __init__(self, upscale):
        super().__init__()
        self.upscale = upscale

    def forward(self, x):
        ''' forward '''
        return F.interpolate(x, scale_factor=self.upscale, mode='bicubic')
