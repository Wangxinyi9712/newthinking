from __future__ import annotations

import torch
import torch.nn as nn


def _conv(dim: int):
    return nn.Conv3d if dim == 3 else nn.Conv2d


def _norm(dim: int):
    return nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d


class SegDiscriminator(nn.Module):
    """
    输入: concat([image, prob_map])，通道数 = in_channels + out_channels(二分类时通常1)
    输出: patch-level real/fake logits
    """
    def __init__(self, in_channels: int, out_channels: int = 1, base_ch: int = 32, dim: int = 3):
        super().__init__()
        conv = _conv(dim)
        norm = _norm(dim)
        cin = in_channels + out_channels

        self.net = nn.Sequential(
            conv(cin, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_ch, base_ch * 2, 4, stride=2, padding=1),
            norm(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            norm(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_ch * 4, base_ch * 4, 3, stride=1, padding=1),
            norm(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_ch * 4, 1, 1, stride=1, padding=0),
        )

    def forward(self, x_img: torch.Tensor, x_prob: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_img, x_prob], dim=1)
        return self.net(x)