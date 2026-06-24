import torch
import torch.nn.functional as F


def supervised_loss(logits, target):
    target = F.interpolate(target.float(), size=logits.shape[2:], mode="trilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(logits, target)


def unsupervised_loss(logits, pseudo):
    pseudo = F.interpolate(pseudo.float(), size=logits.shape[2:], mode="trilinear", align_corners=False)
    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


def spectral_consistency_loss(student, teacher):

    def fft(x):
        return torch.fft.fftn(x.float(), dim=(2,3,4)).abs()

    return F.mse_loss(fft(student), fft(teacher))