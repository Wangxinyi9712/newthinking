from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_contrast_loss(
    feat: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Memory-safe prototype contrastive loss

    feat:
        B,C,D,H,W

    mask:
        B,1,D,H,W
    """

    B, C = feat.shape[:2]

    feat = F.normalize(feat, dim=1)

    loss = 0.0
    valid = 0

    for b in range(B):

        f = feat[b]

        m = (mask[b] > 0.5)

        if m.sum() < 10:
            continue

        fg_proto = f[:, m.squeeze(0)].mean(dim=1)

        bg_proto = f[:, (~m).squeeze(0)].mean(dim=1)

        fg_proto = F.normalize(fg_proto, dim=0)
        bg_proto = F.normalize(bg_proto, dim=0)

        sim = (fg_proto * bg_proto).sum()

        loss += sim

        valid += 1

    if valid == 0:
        return torch.tensor(
            0.0,
            device=feat.device,
            requires_grad=True,
        )

    return loss / valid