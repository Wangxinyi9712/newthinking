import torch


def frequency_filter(x):

    # ❗ FORCE float32 + CPU-safe FFT
    x = x.float().detach()

    fft = torch.fft.fftn(x, dim=(2,3,4))
    amp = fft.abs()

    amp = amp / (amp.mean() + 1e-6)

    return x * amp