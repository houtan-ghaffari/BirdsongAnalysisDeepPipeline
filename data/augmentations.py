__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import random
import torch


def apply_bernoulli_noise(x: torch.Tensor, p_apply: float = 1, min_keep_prob: float = 0.7) -> torch.Tensor:
    """
    Applies Bernoulli noise to a batched tensor as a data augmentation.

    Args:
        x: Input tensor of shape (B, ...).
        p_apply: The probability (0.0 to 1.0) that a given sample is augmented.
        min_keep_prob: The lower bound for the noise intensity (keeps between min_keep_prob and 1.0).
    """
    if p_apply <= 0.0:
        return x

    shape = [x.size(0)] + [1] * (x.ndim - 1)

    # decide the items in the batch get augmented (1 = augment, 0 = skip)
    apply_mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(p_apply).bool()

    # random noise probabilities for the whole batch
    p = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(min_keep_prob, 1.0)

    # if a sample is not selected for augmentation, force its keep-probability to 1.0 (no noise)
    p = torch.where(apply_mask, p, 1.0)

    return x * torch.bernoulli(p.expand_as(x))


def apply_random_gain(x: torch.Tensor) -> torch.Tensor:
    """ x: a batch of audio waveforms of shape (B, T) """
    return x * random.uniform(0.5, 1)


def apply_color_noise(x: torch.Tensor, p_apply: float) -> torch.Tensor:
    """
    Applies power-law noise to a 1D audio waveform or a batch of waveforms.
    Expects x of shape (batch, num_samples) or (num_samples,).
    """
    squeeze_output = False
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze_output = True

    batch_size, num_samples = x.shape
    device = x.device
    dtype = x.dtype

    # f_decay: 1.0 is pink, 2.0 is brown, -1.0 is blue, -2.0 is violet
    f_decay = torch.empty((batch_size, 1), dtype=dtype, device=device).uniform_(-2.0, 2.0)
    snr_db = torch.empty((batch_size, 1), dtype=dtype, device=device).uniform_(5.0, 25.0)
    noise = torch.randn(batch_size, num_samples, dtype=dtype, device=device)

    spec = torch.fft.rfft(noise, dim=1)
    freqs = torch.arange(1, spec.shape[1] + 1, dtype=dtype, device=device).unsqueeze(0)
    mask = 1.0 / (freqs ** (f_decay / 2.0))  # divide f_decay by 2 because we are working in power not amplitude
    spec *= mask

    noise = torch.fft.irfft(spec, n=num_samples, dim=1)
    noise = noise / (1e-8 + noise.square().mean(dim=-1, keepdim=True).sqrt())

    clean_rms = x.square().mean(dim=-1, keepdim=True).sqrt()
    noise_amp = clean_rms / (10 ** (snr_db / 20.0))

    prob_mask = torch.empty((batch_size, 1), dtype=dtype, device=device).bernoulli_(p_apply)
    noise_amp = noise_amp * prob_mask

    out = x + (noise_amp * noise)
    return out.squeeze(0) if squeeze_output else out
