import torch
import torch.nn.functional as F


def supervised_loss(logits, target):
    target = target.float()
    return F.binary_cross_entropy_with_logits(logits, target)


def unsupervised_loss(logits, pseudo):
    return F.mse_loss(torch.sigmoid(logits), pseudo)


def spectral_consistency_loss(s, t):

    # CRITICAL FIX: always FP32
    s = s.float().detach()
    t = t.float().detach()

    def fft(x):
        return torch.fft.fftn(x, dim=(2,3,4)).abs()

    return F.mse_loss(fft(s), fft(t))