import torch
import torch.nn.functional as F


def supervised_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def unsupervised_loss(logits, pseudo):
    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


def spectral_consistency_loss(s, t):

    def fft(x):
        return torch.fft.fftn(x, dim=(2,3,4)).abs()

    return F.mse_loss(fft(s), fft(t))