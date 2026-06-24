import torch
import torch.nn.functional as F


def _clean(x):
    if hasattr(x, "detach"):
        x = x.detach()
    return x.float()


def supervised_loss(logits, target):
    logits = _clean(logits)
    target = _clean(target)

    target = F.interpolate(target, size=logits.shape[2:], mode="trilinear", align_corners=False)

    return F.binary_cross_entropy_with_logits(logits, target)


def unsupervised_loss(logits, pseudo):
    logits = _clean(logits)
    pseudo = _clean(pseudo)

    pseudo = F.interpolate(pseudo, size=logits.shape[2:], mode="trilinear", align_corners=False)

    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


def spectral_consistency_loss(s, t):
    s = _clean(s)
    t = _clean(t)

    # ❗ FORCE CPU FFT SAFE MODE (fix fp16/cuFFT crash)
    s = s.float()
    t = t.float()

    def fft(x):
        return torch.fft.fftn(x, dim=(2,3,4)).abs()

    return F.mse_loss(fft(s), fft(t))