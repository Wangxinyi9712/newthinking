import torch


def frequency_filter(x):

    x = x.float().detach()

    fft = torch.fft.fftn(x, dim=(2,3,4))
    amp = fft.abs()

    # 🔥 spectral normalization (OOD suppression)
    amp = amp / (amp.mean() + 1e-6)

    low_mask = amp < amp.mean()

    return x * low_mask.float()