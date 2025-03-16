''' ABPN '''
import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class ABPN(nn.Module):
    ''' ABPN '''
    def __init__(self, num_feat, num_block, scale) -> None:
        super().__init__()
        self.scale = scale
        self.backbone = nn.Sequential(
            nn.Conv2d(1, num_feat, 3, padding=1),
            nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, padding=1), nn.ReLU()) for _ in range(num_block)],
            nn.Conv2d(num_feat, (scale**2)*3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d((scale**2)*3, (scale**2)*1, 3, padding=1),
        )

        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, input):
        ''' forward '''
        shortcut = torch.repeat_interleave(input, self.scale * self.scale, dim=1)
        y = self.backbone(input) + shortcut
        y = self.upsampler(y)
        return y
