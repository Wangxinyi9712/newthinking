import torch
import torch.nn.functional as F


def supervised_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def spectral_consistency_loss(s, t):
    def fft(x):
        return torch.fft.fftn(x, dim=tuple(range(2, x.ndim))).abs()

    return F.mse_loss(fft(s), fft(t))


def entropy(p):
    return -(p * torch.log(p + 1e-6))


def unsupervised_loss(logits, pseudo):
    p = torch.sigmoid(logits)
    return F.mse_loss(p, pseudo)


def prototype_loss(feat, mask):
    return torch.tensor(0.0, device=feat.device)