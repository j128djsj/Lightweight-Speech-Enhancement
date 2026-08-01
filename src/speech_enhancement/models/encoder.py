import torch
import torch.nn.functional as F
from torch import nn

from speech_enhancement.models.aspp import ASPP2D
from speech_enhancement.models.depthwise import DepthwiseSeparableConv2d


class Encoder(nn.Module):
    """Six-stage DSConv encoder with ASPP and low-mid-high fusion."""

    out_channels = 512 * 3 // 2

    def __init__(self, in_channels=1, aspp_channels=512 * 5 // 2):
        super().__init__()
        channels = [16, 32, 64, 128, 256, 256]
        specs = [
            (in_channels, channels[0], (3, 3), (1, 1), (1, 1)),
            (channels[0], channels[1], (3, 3), (2, 2), (1, 1)),
            (channels[1], channels[2], (3, 3), (2, 2), (1, 1)),
            (channels[2], channels[3], (3, 3), (1, 2), (1, 1)),
            (channels[3], channels[4], (3, 3), (1, 2), (1, 1)),
            (channels[4], channels[5], (3, 3), (1, 2), (1, 0)),
        ]
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    DepthwiseSeparableConv2d(cin, cout, kernel, stride, padding, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=False),
                )
                for cin, cout, kernel, stride, padding in specs
            ]
        )
        self.aspp = ASPP2D(channels[-1])
        self.low_projection = DepthwiseSeparableConv2d(
            channels[0], 256, kernel_size=(4, 32), stride=(4, 32), padding=(1, 1), bias=False
        )
        self.mid_projection = DepthwiseSeparableConv2d(
            channels[2], 256, kernel_size=(2, 8), stride=(2, 8), padding=(0, 1), bias=False
        )
        self.high_projection = DepthwiseSeparableConv2d(
            aspp_channels, 256, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        low_level = None
        mid_level = None
        for idx, block in enumerate(self.blocks, start=1):
            x = block(x)
            if idx == 1:
                low_level = x
            elif idx == 3:
                mid_level = x

        high_level = self.high_projection(self.aspp(x))
        low_level = self.low_projection(low_level)
        mid_level = self.mid_projection(mid_level)

        target_size = (
            max(high_level.size(2), mid_level.size(2), low_level.size(2)),
            max(high_level.size(3), mid_level.size(3), low_level.size(3)),
        )
        features = [
            F.interpolate(item, size=target_size, mode="bilinear", align_corners=True)
            for item in (high_level, mid_level, low_level)
        ]
        return self.dropout(torch.cat(features, dim=1))
