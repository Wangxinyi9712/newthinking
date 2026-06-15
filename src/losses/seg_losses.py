import torch
import torch.nn.functional as F


# --------------------------
# supervised loss
# --------------------------
def supervised_loss(logits, target):
    target = target.float()
    return F.binary_cross_entropy_with_logits(logits, target)


# --------------------------
# spectral consistency loss (NEW TMI CORE)
# --------------------------
def spectral_consistency_loss(student_logits, teacher_logits):

    def fft(x):
        return torch.fft.fftn(x, dim=(2, 3, 4)).abs()

    s = fft(student_logits)
    t = fft(teacher_logits)

    return F.mse_loss(s, t)


# --------------------------
# uncertainty (entropy)
# --------------------------
def entropy(p):
    return -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))


def mutual_information(p):
    return entropy(p) - entropy(p.mean(dim=0, keepdim=True))


# --------------------------
# pseudo consistency
# --------------------------
def unsupervised_loss(student_logits, teacher_prob):

    student_prob = torch.sigmoid(student_logits)
    return F.mse_loss(student_prob, teacher_prob)


# --------------------------
# CPS loss
# --------------------------
def cps_loss(p1, p2):
    return F.mse_loss(torch.sigmoid(p1), torch.sigmoid(p2))