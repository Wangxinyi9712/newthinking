import torch


def frequency_filter(x, mode="lowpass"):

    # -------------------------
    # 1. 保证稳定输入
    # -------------------------
    if x.dtype != torch.float32:
        x = x.float()

    x = x.detach()

    # -------------------------
    # 2. FFT in FP32 ONLY
    # -------------------------
    fft = torch.fft.fftn(x, dim=(2, 3, 4))
    amp = torch.abs(fft)

    # -------------------------
    # 3. low-frequency mask（stable version）
    # -------------------------
    b, c, d, h, w = amp.shape

    mask = torch.zeros_like(amp)

    kd = d // 4
    kh = h // 4
    kw = w // 4

    mask[:, :, :kd, :kh, :kw] = 1.0

    amp = amp * mask

    # -------------------------
    # 4. reconstruction safe path
    # -------------------------
    phase = torch.angle(fft)
    fft_filtered = torch.polar(amp, phase)

    out = torch.fft.ifftn(fft_filtered, dim=(2, 3, 4)).real

    return out