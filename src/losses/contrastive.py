import torch
import torch.nn.functional as F


class PrototypeMemory:
    def __init__(self, num_classes=2, dim=128, momentum=0.95):
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.prototypes = torch.zeros(num_classes, dim).cuda()

    def update(self, feat, mask, weight=None):

        feat = F.adaptive_avg_pool3d(feat, 1).view(feat.size(0), -1)
        mask = mask.squeeze(1)

        if weight is None:
            weight = torch.ones(feat.size(0)).cuda()

        for c in range(self.num_classes):

            idx = (mask == c).view(mask.size(0), -1).any(dim=1)

            if idx.sum() == 0:
                continue

            w = weight[idx].view(-1, 1)
            proto = (feat[idx] * w).mean(0)

            self.prototypes[c] = (
                self.momentum * self.prototypes[c]
                + (1 - self.momentum) * proto
            )


# -------------------------
# final contrastive loss
# -------------------------
def prototype_contrast_loss(feat, mask, memory):

    feat = F.adaptive_avg_pool3d(feat, 1).view(feat.size(0), -1)
    mask = mask.squeeze(1)

    loss = 0.0

    for c in range(memory.num_classes):

        idx = (mask == c).view(mask.size(0), -1).any(dim=1)

        if idx.sum() == 0:
            continue

        feat_c = feat[idx]
        proto = memory.prototypes[c]

        sim = F.cosine_similarity(
            feat_c,
            proto.unsqueeze(0).expand_as(feat_c),
            dim=1
        )

        loss += (1 - sim).mean()

    return loss