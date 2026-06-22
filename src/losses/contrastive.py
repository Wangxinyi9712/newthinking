import torch
import torch.nn.functional as F


def prototype_contrast_loss(feats, masks, temperature=0.1):
    """
    class-wise prototype contrastive loss
    """

    b, c, *spatial = feats.shape
    feats = feats.view(b, c, -1).permute(0, 2, 1)  # B,N,C
    masks = masks.view(b, -1)

    loss = 0.0

    for i in range(b):
        f = feats[i]
        m = masks[i]

        pos = f[m > 0.5]
        neg = f[m <= 0.5]

        if len(pos) == 0 or len(neg) == 0:
            continue

        pos_mean = pos.mean(0, keepdim=True)
        neg_mean = neg.mean(0, keepdim=True)

        sim = F.cosine_similarity(pos_mean, neg_mean)
        loss += torch.relu(sim)

    return loss / max(b, 1)