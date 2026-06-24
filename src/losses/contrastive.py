import torch
import torch.nn.functional as F


def prototype_contrast_loss(feat, mask, memory, num_classes=2):

    B,C,D,H,W = feat.shape

    feat = feat.permute(0,2,3,4,1).reshape(-1, C)
    mask = mask.view(-1)

    loss = 0.0

    for c in range(num_classes):

        idx = (mask == c)
        if idx.sum() == 0:
            continue

        pos_feat = feat[idx]

        proto = memory.get(c, pos_feat.mean(0))

        proto = proto.to(feat.device)

        sim = F.cosine_similarity(
            pos_feat,
            proto.unsqueeze(0),
            dim=1
        )

        loss += (1 - sim).mean()

        memory[c] = pos_feat.mean(0).detach()

    return loss / num_classes