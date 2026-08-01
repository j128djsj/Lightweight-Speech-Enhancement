import torch
import torch.nn.functional as F


def compressed_to_linear_mag(mag, compress_factor):
    return torch.clamp(mag, min=0.0).pow(1.0 / compress_factor)


def complex_channels(compressed_mag, phase, compress_factor):
    mag = compressed_to_linear_mag(compressed_mag.float(), compress_factor)
    phase = phase.float()
    return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


def align_time_frequency(a, b):
    min_freq = min(a.size(-2), b.size(-2))
    min_time = min(a.size(-1), b.size(-1))
    return a[..., :min_freq, :min_time], b[..., :min_freq, :min_time]


def magnitude_loss(enhanced_mag, clean_mag):
    enhanced_mag, clean_mag = align_time_frequency(enhanced_mag.float(), clean_mag.float())
    return F.mse_loss(enhanced_mag, clean_mag)


def complex_spectrum_loss(enhanced_mag, enhanced_phase, clean_mag, clean_phase, config):
    enhanced_spec = complex_channels(enhanced_mag, enhanced_phase, config.compress_factor)
    clean_spec = complex_channels(clean_mag, clean_phase, config.compress_factor)
    enhanced_spec, clean_spec = align_time_frequency(enhanced_spec, clean_spec)
    return F.mse_loss(enhanced_spec[:, 0:1], clean_spec[:, 0:1]) + F.mse_loss(
        enhanced_spec[:, 1:2], clean_spec[:, 1:2]
    )


def stft_consistency_loss(enhanced_mag, phase, audio_lengths, config):
    linear_mag = compressed_to_linear_mag(enhanced_mag.float(), config.compress_factor).squeeze(1)
    phase = phase.float().squeeze(1)
    window = torch.hann_window(config.win_size, device=enhanced_mag.device, dtype=linear_mag.dtype)
    spec = torch.polar(linear_mag, phase)
    wav = torch.istft(
        spec,
        config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window=window,
        center=True,
        length=int(audio_lengths.max().item()),
    )
    projected = torch.stft(
        wav,
        config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    spec, projected = align_time_frequency(spec, projected)
    return F.mse_loss(spec.real, projected.real) + F.mse_loss(spec.imag, projected.imag)


def classifier_loss(logits, speaker_labels):
    speaker_mask = speaker_labels >= 0
    if speaker_mask.any():
        return F.cross_entropy(logits[speaker_mask], speaker_labels[speaker_mask])
    return logits.sum() * 0.0


def generator_loss_components(
    enhanced_mag,
    classifier_logits,
    input_phase,
    clean_mag,
    clean_phase,
    speaker_labels,
    audio_lengths,
    metric_gan_loss,
    config,
):
    mag_loss = magnitude_loss(enhanced_mag, clean_mag)
    con_loss = stft_consistency_loss(enhanced_mag, input_phase, audio_lengths, config)
    com_loss = complex_spectrum_loss(enhanced_mag, input_phase, clean_mag, clean_phase, config)

    clean_spec = complex_channels(clean_mag, clean_phase, config.compress_factor)
    enhanced_spec = complex_channels(enhanced_mag, input_phase, config.compress_factor)
    metric_loss = metric_gan_loss(clean_spec, enhanced_spec, mode="generator")
    cls_loss = classifier_loss(classifier_logits, speaker_labels)

    total = (
        config.lambda_mag * mag_loss
        + config.lambda_consistency * con_loss
        + config.lambda_complex * com_loss
        + config.lambda_metric * metric_loss
        + config.lambda_classifier * cls_loss
    )
    return {
        "total": total,
        "mag": mag_loss.detach(),
        "consistency": con_loss.detach(),
        "complex": com_loss.detach(),
        "metric": metric_loss.detach(),
        "classifier": cls_loss.detach(),
    }
