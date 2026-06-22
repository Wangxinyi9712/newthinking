import torch
import torch.nn.functional as F


# --------------------------
# supervised
# --------------------------
def supervised_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


# --------------------------
# unsupervised
# --------------------------
def unsupervised_loss(logits, pseudo):
    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


# --------------------------
# spectral consistency (REPLACED GAN)
# --------------------------
def spectral_consistency_loss(s, t):
    def fft(x):
        return torch.fft.fftn(x, dim=(-3, -2, -1)).abs()

    return F.mse_loss(fft(s), fft(t))


# --------------------------
# entropy uncertainty
# --------------------------
def entropy(p):
    return -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))


def mutual_information(p):
    return entropy(p) - entropy(p.mean(dim=0, keepdim=True))


# --------------------------
# prototype contrast (TMI boost)
# --------------------------
def prototype_contrast_loss(feat, label):
    """
    simple global prototype contrast
    """
    feat = F.adaptive_avg_pool3d(feat, 1).flatten(1)

    pos = feat[label.flatten() > 0].mean(dim=0, keepdim=True)
    neg = feat[label.flatten() == 0].mean(dim=0, keepdim=True)

    return F.mse_loss(pos, neg)