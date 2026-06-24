import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c1, c2, 3, padding=1),
            nn.InstanceNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.Conv3d(c2, c2, 3, padding=1),
            nn.InstanceNorm3d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridUNet(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, channels=(32,64,128,256)):
        super().__init__()

        c1,c2,c3,c4 = channels

        self.pool = nn.MaxPool3d(2, ceil_mode=True)

        self.enc1 = Conv(in_channels,c1)
        self.enc2 = Conv(c1,c2)
        self.enc3 = Conv(c2,c3)
        self.enc4 = Conv(c3,c4)

        self.up3 = nn.ConvTranspose3d(c4,c3,2,stride=2)
        self.up2 = nn.ConvTranspose3d(c3,c2,2,stride=2)
        self.up1 = nn.ConvTranspose3d(c2,c1,2,stride=2)

        self.dec3 = Conv(c3*2,c3)
        self.dec2 = Conv(c2*2,c2)
        self.dec1 = Conv(c1*2,c1)

        self.seg_head = nn.Conv3d(c1,out_channels,1)

        # 🔥 stable feature head (GLOBAL POOL)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.feat_proj = nn.Linear(c1, 128)

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
        d3 = self.dec3(torch.cat([d3,x3],1))

        d2 = self.align(self.up2(d3), x2)
        d2 = self.dec2(torch.cat([d2,x2],1))

        d1 = self.align(self.up1(d2), x1)
        d1 = self.dec1(torch.cat([d1,x1],1))

        logits = self.seg_head(d1)

        feat = self.global_pool(d1).flatten(1)
        feat = self.feat_proj(feat)

        return (logits, feat) if return_feat else logits