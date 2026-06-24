import torch
import torch.nn.functional as F


def supervised_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def unsupervised_loss(logits, pseudo):
    logits = F.interpolate(logits, size=pseudo.shape[2:], mode="trilinear", align_corners=False)
    return F.mse_loss(torch.sigmoid(logits), pseudo)


def spectral_consistency_loss(student, teacher):
    # 🔥 disable AMP FFT crash
    student = student.float()
    teacher = teacher.float()

    def fft(x):
        return torch.fft.fftn(x, dim=(2,3,4)).abs()

    return F.mse_loss(fft(student), fft(teacher))