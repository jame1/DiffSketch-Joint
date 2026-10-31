"""Shared multi-scale encoder built upon MobileNetV3-Small stages."""
from typing import List, Tuple
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from .blocks import ConvBlock


class SharedEncoder(nn.Module):
    """
    Lightweight shared convolutional encoder adapted from MobileNetV3-Small.
    Replaces the first 3-channel conv with a 4-channel conv (damaged RGB + mask)
    and removes the classification head, outputting multi-scale features at
    1/2, 1/4, and 1/8 spatial resolutions.

    Input:  x [B, 4, H, W]
    Output: List[Tensor] -> [f_1/2 [B, C1, H/2, W/2],
                             f_1/4 [B, C2, H/4, W/4],
                             f_1/8 [B, C3, H/8, W/8]]
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None).features

        first_conv: nn.Conv2d = backbone[0][0]
        backbone[0][0] = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        self.stage1 = backbone[0]
        self.stage2 = backbone[1]
        self.stage3 = nn.Sequential(backbone[2], backbone[3])

        self.proj1 = ConvBlock(16, base_channels, stride=1)
        self.proj2 = ConvBlock(16, base_channels * 2, stride=1)
        self.proj3 = ConvBlock(24, base_channels * 4, stride=1)
        self.out_channels: Tuple[int, int, int] = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return [self.proj1(s1), self.proj2(s2), self.proj3(s3)]