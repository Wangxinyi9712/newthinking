import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.InstanceNorm3d(cout),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(cout, cout, 3, padding=1),
            nn.InstanceNorm3d(cout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, channels=(16, 32, 64, 128)):
        super().__init__()

        c1, c2, c3, c4 = channels

        self.pool = nn.MaxPool3d(2)

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = nn.Conv3d(c1, out_channels, 1)

    def forward(self, x, return_features=False):

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        out = self.head(d1)

        if return_features:
            return out, d1

        return out