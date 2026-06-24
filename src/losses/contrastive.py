import torch
import torch.nn.functional as F


def prototype_contrast_loss(feat, mask, memory, num_classes=2):

    # feat: [B,C,D,H,W]
    B, C, D, H, W = feat.shape

    loss = 0.0

    feat_flat = feat.permute(0,2,3,4,1).reshape(-1, C)
    mask_flat = mask.view(-1)

    for c in range(num_classes):

        idx = (mask_flat == c)

        if idx.sum() == 0:
            continue

        pos = feat_flat[idx]

        proto = pos.mean(dim=0)

        if c in memory:
            proto = 0.9 * memory[c] + 0.1 * proto

        memory[c] = proto.detach()

        loss += (1 - F.cosine_similarity(pos, proto.unsqueeze(0), dim=1)).mean()

    return loss