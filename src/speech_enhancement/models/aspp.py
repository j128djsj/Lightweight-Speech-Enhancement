import torch
import torch.nn.functional as F
from torch import nn

from speech_enhancement.models.depthwise import DepthwiseSeparableConv2d


class ASPP2D(nn.Module):
    """Atrous spatial pyramid pooling with dilation rates 2, 4 and 8."""

    def __init__(self, in_channels=256, branch_channels=256):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                DepthwiseSeparableConv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    bias=False,
                ),
                DepthwiseSeparableConv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=4,
                    dilation=4,
                    bias=False,
                ),
                DepthwiseSeparableConv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=8,
                    dilation=8,
                    bias=False,
                ),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.pool_projection = DepthwiseSeparableConv2d(
            in_channels, branch_channels, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        pooled = self.pool_projection(self.pool(x))
        pooled = F.interpolate(
            pooled, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        features.append(pooled)
        return self.dropout(torch.cat(features, dim=1))
