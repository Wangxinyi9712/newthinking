import torch
import torch.nn.functional as F


def supervised_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target.float())


def unsupervised_loss(logits, pseudo):
    return F.mse_loss(torch.sigmoid(logits), pseudo)


def spectral_consistency_loss(s, t):

    def fft(x):
        x = x.float().detach()
        return torch.fft.fftn(x, dim=(2,3,4)).abs()

    return F.mse_loss(fft(s), fft(t))