import torch
import torch.nn.functional as F


def supervised_loss(logits, target):

    if logits.shape != target.shape:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")

    return F.binary_cross_entropy_with_logits(logits, target)


def unsupervised_loss(logits, pseudo):

    logits = F.interpolate(logits, size=pseudo.shape[2:], mode="trilinear", align_corners=False)

    return F.mse_loss(torch.sigmoid(logits), pseudo)