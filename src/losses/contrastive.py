import torch
import torch.nn.functional as F


def prototype_contrast_loss(feat, mask, memory, num_classes=2):

    feat = feat.detach().float()
    mask = mask.detach().float()

    B,C,D,H,W = feat.shape

    feat = feat.permute(0,2,3,4,1).reshape(-1, C)
    mask = mask.view(-1)

    loss = 0.0

    for c in range(num_classes):

        idx = (mask == c)
        if idx.sum() < 10:
            continue

        pos = feat[idx]

        proto = memory.get(c, pos.mean(0)).to(feat.device)

        # cosine + L2 hybrid (TMI improvement)
        cos = F.cosine_similarity(pos, proto.unsqueeze(0), dim=1)
        l2 = (pos - proto).pow(2).mean(dim=1)

        loss += (1 - cos).mean() + 0.1 * l2.mean()

        memory[c] = pos.mean(0).detach()

    return loss / max(num_classes, 1)