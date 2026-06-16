import torch
import torch.nn.functional as F


# =========================================================
# 1. supervised loss
# =========================================================
def supervised_loss(logits, target):
    target = target.float()
    return F.binary_cross_entropy_with_logits(logits, target)


# =========================================================
# 2. spectral consistency loss (TMI CORE)
# =========================================================
def spectral_consistency_loss(s_logits, t_logits):
    def fft(x):
        return torch.fft.fftn(x, dim=tuple(range(2, x.ndim))).abs()

    s = fft(s_logits)
    t = fft(t_logits)
    return F.mse_loss(s, t)


# =========================================================
# 3. topology-aware surrogate loss (HD95 proxy)
# =========================================================
def topology_loss(logits, target):
    prob = torch.sigmoid(logits)

    # boundary emphasis (gradient magnitude)
    dx = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    dy = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])

    boundary = dx.mean() + dy.mean()
    return boundary


# =========================================================
# 4. uncertainty
# =========================================================
def entropy(p):
    return -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))


def mutual_information(p):
    return entropy(p) - entropy(p.mean(dim=0, keepdim=True))


def uncertainty_map(p):
    return entropy(p)


# =========================================================
# 5. unsupervised consistency
# =========================================================
def unsupervised_loss(logits, pseudo):
    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


# =========================================================
# 6. CPS dual student consistency
# =========================================================
def cps_loss(p1, p2):
    return F.mse_loss(torch.sigmoid(p1), torch.sigmoid(p2))


# =========================================================
# 7. feature contrastive loss (light TMI version)
# =========================================================
def feature_contrastive_loss(f_s, f_t, temperature=0.07):
    """
    global pooled InfoNCE
    """
    b = f_s.shape[0]

    fs = F.normalize(f_s.mean(dim=(2, 3, 4)), dim=1)
    ft = F.normalize(f_t.mean(dim=(2, 3, 4)), dim=1)

    logits = torch.mm(fs, ft.t()) / temperature
    labels = torch.arange(b, device=fs.device)

    return F.cross_entropy(logits, labels)