import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim):
    return nn.Conv3d if dim == 3 else nn.Conv2d


class SpectralGate(nn.Module):
    def __init__(self, channels, dim=3):
        super().__init__()
        conv = _conv(dim)
        self.conv = conv(channels, channels, 1)

    def forward(self, x):
        freq = torch.fft.fftn(x, dim=(-3, -2, -1))
        amp = freq.abs()
        amp = self.conv(amp)
        return x * torch.sigmoid(amp)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)

class FrequencyGate(nn.Module):
    def forward(self, x):
        fft = torch.fft.fftn(x, dim=tuple(range(2, x.ndim)))
        amp = torch.abs(fft)
        return x * (amp / (amp.mean() + 1e-6))