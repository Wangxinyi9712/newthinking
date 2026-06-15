import torch
import torch.nn.functional as F

def entropy(p):
    p = torch.clamp(p, 1e-6, 1-1e-6)
    return -p * torch.log(p) - (1-p) * torch.log(1-p)


def kl(p, q):
    p = torch.clamp(p, 1e-6, 1-1e-6)
    q = torch.clamp(q, 1e-6, 1-1e-6)
    return p * torch.log(p / q)