import torch
import torch.nn.functional as F


# -------------------------
# safe BCE
# -------------------------
def supervised_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target.float())


# -------------------------
# SAFE unsupervised loss
# -------------------------
def unsupervised_loss(logits, pseudo):
    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


# -------------------------
# 🔥 SAFE FFT (CRITICAL FIX)
# -------------------------
def _safe_fft(x):

    # ALWAYS FP32
    x = x.float().detach()

    # move out of meta / monai wrapper safety
    x = x.contiguous()

    fft = torch.fft.fftn(x, dim=(2, 3, 4))
    return fft.abs()


# -------------------------
# spectral consistency loss (FIXED)
# -------------------------
def spectral_consistency_loss(student_logits, teacher_logits):

    s = _safe_fft(student_logits)
    t = _safe_fft(teacher_logits)

    return F.mse_loss(s, t)


# -------------------------
# entropy (stable)
# -------------------------
def entropy(p):
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


# -------------------------
# CPS loss
# -------------------------
def cps_loss(p1, p2):
    return F.mse_loss(torch.sigmoid(p1), torch.sigmoid(p2))