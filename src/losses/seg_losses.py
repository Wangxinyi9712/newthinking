import torch
import torch.nn.functional as F


def supervised_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target.float())


def unsupervised_loss(pred, teacher, tau=0.7):
    pseudo = (teacher > tau).float()
    return F.binary_cross_entropy_with_logits(pred, pseudo)


def cps_loss(p1, p2):
    return F.mse_loss(torch.sigmoid(p1), torch.sigmoid(p2))


def structural_loss(pred, target):
    return F.l1_loss(torch.sigmoid(pred), target.float())


def minority_sensitive_loss(pred, target, weight):
    return F.binary_cross_entropy_with_logits(pred, target.float()) * weight.mean()


def adaptive_tau_from_quantile(p):
    return torch.quantile(p.detach(), 0.7).clamp(0.5, 0.9)