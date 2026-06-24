from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_contrast_loss(
    feat: torch.Tensor,
    mask: torch.Tensor,
    memory: dict[int, torch.Tensor],
    num_classes: int = 2,
    max_samples_per_class: int = 2048,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    Memory-safe voxel prototype contrastive loss.

    Args:
        feat:  [B, C, D, H, W]
        mask:  [B, 1, D, H, W]
        memory: dict used as EMA prototype bank
    """
    if feat.ndim != 5:
        raise ValueError(f"feat must be 5D [B,C,D,H,W], got {feat.shape}")

    feat = feat.float()
    mask = mask.detach().float()

    if mask.ndim == 4:
        mask = mask.unsqueeze(1)

    if mask.shape[2:] != feat.shape[2:]:
        mask = F.interpolate(mask, size=feat.shape[2:], mode="nearest")

    mask = (mask > 0.5).long()

    b, c, d, h, w = feat.shape

    feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, c)
    mask_flat = mask.reshape(-1)

    feat_flat = F.normalize(feat_flat, dim=1)

    total = feat_flat.new_tensor(0.0)
    valid_classes = 0

    for cls in range(num_classes):
        idx = torch.nonzero(mask_flat == cls, as_tuple=False).flatten()

        if idx.numel() < 16:
            continue

        if idx.numel() > max_samples_per_class:
            perm = torch.randperm(idx.numel(), device=feat.device)[:max_samples_per_class]
            idx = idx[perm]

        cls_feat = feat_flat[idx]
        batch_proto = cls_feat.mean(dim=0)
        batch_proto = F.normalize(batch_proto, dim=0)

        old_proto = memory.get(cls)
        if old_proto is None:
            proto = batch_proto.detach()
        else:
            proto = momentum * old_proto.to(feat.device) + (1.0 - momentum) * batch_proto.detach()
            proto = F.normalize(proto, dim=0)

        memory[cls] = proto.detach()

        sim = F.cosine_similarity(cls_feat, proto.unsqueeze(0), dim=1)
        total = total + (1.0 - sim).mean()
        valid_classes += 1

    if valid_classes == 0:
        return feat_flat.new_tensor(0.0)

    return total / valid_classes