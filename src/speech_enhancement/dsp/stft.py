import torch


def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """Return compressed magnitude, phase, and real/imag channels."""
    window = torch.hann_window(win_size, device=y.device)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = torch.abs(spec).pow(compress_factor)
    phase = torch.angle(spec)
    complex_channels = torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)
    return mag, phase, complex_channels


def mag_phase_istft(mag, phase, n_fft, hop_size, win_size, compress_factor=1.0, center=True, length=None):
    """Invert a compressed magnitude and phase spectrum."""
    phase = phase.to(mag.device)
    mag = torch.clamp(mag, min=0).pow(1.0 / compress_factor)
    spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    window = torch.hann_window(win_size, device=spec.device)
    return torch.istft(
        spec,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        length=length,
    )


def griffin_lim_reconstruct(
    mag,
    init_phase,
    n_fft,
    hop_size,
    win_size,
    compress_factor=1.0,
    n_iter=3,
    center=True,
    length=None,
):
    """Reconstruct a waveform with Griffin-Lim phase refinement."""
    if mag.dim() == 4 and mag.size(1) == 1:
        mag = mag.squeeze(1)
    if init_phase.dim() == 4 and init_phase.size(1) == 1:
        init_phase = init_phase.squeeze(1)

    mag = torch.clamp(mag, min=0).pow(1.0 / compress_factor)
    phase = init_phase.to(device=mag.device, dtype=mag.dtype)
    window = torch.hann_window(win_size, device=mag.device, dtype=mag.dtype)
    spec = torch.polar(mag, phase)

    for _ in range(max(int(n_iter), 0)):
        wav = torch.istft(
            spec,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=center,
            length=length,
        )
        refined = torch.stft(
            wav,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        phase = torch.angle(refined)
        spec, mag, phase = _match_phase_to_mag(phase, mag)

    wav = torch.istft(
        spec,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        length=length,
    )
    return wav, phase


def _match_phase_to_mag(phase, mag):
    if phase.shape[-2:] != mag.shape[-2:]:
        min_freq = min(phase.size(-2), mag.size(-2))
        min_time = min(phase.size(-1), mag.size(-1))
        phase = phase[..., :min_freq, :min_time]
        mag = mag[..., :min_freq, :min_time]
    return torch.polar(mag, phase), mag, phase
