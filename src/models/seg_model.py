import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridUNet(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, channels=(32,64,128,256)):
        super().__init__()

        c1,c2,c3,c4 = channels

        self.pool = nn.MaxPool3d(2, ceil_mode=True)

        self.enc1 = nn.Sequential(nn.Conv3d(in_channels,c1,3,padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv3d(c1,c2,3,padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv3d(c2,c3,3,padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv3d(c3,c4,3,padding=1), nn.ReLU())

        self.up3 = nn.ConvTranspose3d(c4,c3,2,stride=2)
        self.up2 = nn.ConvTranspose3d(c3,c2,2,stride=2)
        self.up1 = nn.ConvTranspose3d(c2,c1,2,stride=2)

        self.dec3 = nn.Conv3d(c3*2, c3, 3, padding=1)
        self.dec2 = nn.Conv3d(c2*2, c2, 3, padding=1)
        self.dec1 = nn.Conv3d(c1*2, c1, 3, padding=1)

        self.seg_head = nn.Conv3d(c1, out_channels, 1)

    def align(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x

    def forward(self, x, return_feat=False):

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        d3 = self.align(self.up3(x4), x3)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.align(self.up2(d3), x2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.align(self.up1(d2), x1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        logits = self.seg_head(d1)

        feat = d1

        return (logits, feat) if return_feat else logits