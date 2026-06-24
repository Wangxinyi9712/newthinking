import torch


def frequency_filter(x):

    x = x.float().detach()

    fft = torch.fft.fftn(x, dim=(2, 3, 4))
    amp = torch.abs(fft)

    b, c, d, h, w = amp.shape

    mask = torch.zeros_like(amp)
    mask[:, :, : d // 4, : h // 4, : w // 4] = 1.0

    amp = amp * mask

    phase = torch.angle(fft)
    fft = torch.polar(amp, phase)

    return torch.fft.ifftn(fft, dim=(2, 3, 4)).real