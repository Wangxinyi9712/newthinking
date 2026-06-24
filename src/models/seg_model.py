from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridUNet(nn.Module):
    """
    Stable 3D U-Net.

    Contract:
        - forward(x) -> logits
        - forward(x, return_features=True) -> (logits, feature_map)

    feature_map:
        spatial feature map with shape [B, feat_channels, D, H, W],
        used by prototype_contrast_loss safely with internal voxel sampling.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        channels: tuple[int, int, int, int] | list[int] = (16, 32, 64, 128),
        use_transformer: bool = False,
    ):
        super().__init__()

        if len(channels) != 4:
            raise ValueError(f"channels must have 4 values, got {channels}")

        c1, c2, c3, c4 = [int(c) for c in channels]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.seg_head = nn.Conv3d(c1, out_channels, kernel_size=1)

        # Lightweight projection for prototype contrastive learning.
        self.feature_head = nn.Sequential(
            nn.Conv3d(c1, min(32, c1), kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(min(32, c1), 32, kernel_size=1),
        )

        self.use_transformer = bool(use_transformer)

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x

        return F.interpolate(
            x,
            size=ref.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False, return_feat: bool = False):
        need_features = bool(return_features or return_feat)

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        d3 = self._align(self.up3(x4), x3)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self._align(self.up2(d3), x2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self._align(self.up1(d2), x1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        logits = self.seg_head(d1)

        if need_features:
            feat = self.feature_head(d1)
            return logits, feat

        return logits