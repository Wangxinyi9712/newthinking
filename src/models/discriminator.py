import torch
import torch.nn as nn


def conv(dim):
    return nn.Conv3d if dim == 3 else nn.Conv2d


class SegDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, base=32, dim=3):
        super().__init__()
        Conv = conv(dim)

        self.net = nn.Sequential(
            Conv(in_channels + out_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2),

            Conv(base, base * 2, 4, 2, 1),
            nn.InstanceNorm3d(base * 2) if dim == 3 else nn.InstanceNorm2d(base * 2),
            nn.LeakyReLU(0.2),

            Conv(base * 2, base * 4, 4, 2, 1),
            nn.InstanceNorm3d(base * 4) if dim == 3 else nn.InstanceNorm2d(base * 4),
            nn.LeakyReLU(0.2),

            Conv(base * 4, 1, 1),
        )

    def forward(self, x, p):
        return self.net(torch.cat([x, p], dim=1))