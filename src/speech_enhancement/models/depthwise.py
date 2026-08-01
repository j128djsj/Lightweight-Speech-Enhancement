import torch
from torch import nn


def _to_pair(value):
    return (value, value) if isinstance(value, int) else value


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise spatial filtering followed by pointwise channel mixing."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=_to_pair(kernel_size),
            stride=_to_pair(stride),
            padding=_to_pair(padding),
            dilation=_to_pair(dilation),
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DepthwiseSeparableConvTranspose2d(nn.Module):
    """Depthwise transpose convolution followed by pointwise mixing."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=_to_pair(kernel_size),
            stride=_to_pair(stride),
            padding=_to_pair(padding),
            output_padding=_to_pair(output_padding),
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
