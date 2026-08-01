import torch.nn.functional as F
from torch import nn

from speech_enhancement.models.depthwise import (
    DepthwiseSeparableConv2d,
    DepthwiseSeparableConvTranspose2d,
)


class Decoder(nn.Module):
    """Decoder from Fig. 5: DSConv, DSTransConv, DSConv, DSTransConv, DSConv."""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv2d(
                in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConvTranspose2d(
                in_channels // 2,
                in_channels // 4,
                kernel_size=(4, 8),
                stride=(4, 8),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(
                in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConvTranspose2d(
                in_channels // 8,
                in_channels // 16,
                kernel_size=(1, 4),
                stride=(1, 4),
                output_padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(
                in_channels // 16, out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, original_size):
        x = self.layers(x)
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=True)
        return x
