import torch


def frequency_filter(x):

    x = x.float().detach()

    fft = torch.fft.fftn(x, dim=(2,3,4))
    amp = torch.abs(fft)

    mask = torch.ones_like(amp)
    mask[:, :, ::2, ::2, ::2] = 0.5

    amp = amp * mask

    return torch.fft.ifftn(torch.polar(amp, torch.angle(fft)), dim=(2,3,4)).real