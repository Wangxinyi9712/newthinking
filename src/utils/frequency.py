import torch


def frequency_filter(logits, threshold=0.6):
    """
    FFT-based pseudo label filtering
    remove high-frequency noise
    """

    prob = torch.sigmoid(logits)

    fft = torch.fft.fftn(prob, dim=tuple(range(2, prob.ndim)))
    amp = torch.abs(fft)

    mask = amp < amp.mean() * threshold

    filtered = torch.fft.ifftn(fft * mask, dim=tuple(range(2, prob.ndim))).real

    return filtered.clamp(0, 1)