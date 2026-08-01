import numpy as np
import torch
import torch.nn.functional as F

from speech_enhancement.dsp.stft import griffin_lim_reconstruct

try:
    import pesq
except Exception:
    pesq = None


def normalize_pesq_score(score, config):
    value = (score - config.min_pesq) / (config.max_pesq - config.min_pesq)
    return float(np.clip(value, 0.0, 1.0))


def compute_pesq_score(clean_audio, enhanced_audio, sampling_rate):
    if pesq is None:
        return None
    try:
        min_len = min(len(clean_audio), len(enhanced_audio))
        if min_len <= 0:
            return None
        clean_audio = np.clip(clean_audio[:min_len], -1.0, 1.0).astype(np.float32)
        enhanced_audio = np.clip(enhanced_audio[:min_len], -1.0, 1.0).astype(np.float32)
        return float(pesq.pesq(sampling_rate, clean_audio, enhanced_audio, "wb"))
    except Exception:
        return None


@torch.no_grad()
def reconstruct_waveforms(compressed_mag, phase, audio_lengths, gla_iters, config):
    wav, _ = griffin_lim_reconstruct(
        compressed_mag.float(),
        phase.float(),
        config.n_fft,
        config.hop_size,
        config.win_size,
        compress_factor=config.compress_factor,
        n_iter=gla_iters,
        center=True,
        length=int(audio_lengths.max().item()),
    )
    return wav


@torch.no_grad()
def batch_normalized_pesq_targets(clean_mag, clean_phase, enhanced_mag, input_phase, lengths, config):
    clean_wav = reconstruct_waveforms(clean_mag, clean_phase, lengths, gla_iters=0, config=config)
    enhanced_wav = reconstruct_waveforms(
        enhanced_mag, input_phase, lengths, gla_iters=config.gla_iters, config=config
    )

    scores = []
    for idx in range(clean_wav.size(0)):
        length = int(lengths[idx].item())
        clean_np = clean_wav[idx, :length].detach().cpu().numpy()
        enhanced_np = enhanced_wav[idx, :length].detach().cpu().numpy()
        score = compute_pesq_score(clean_np, enhanced_np, config.sampling_rate)
        if score is None:
            proxy = 1.0 - F.l1_loss(enhanced_mag[idx : idx + 1], clean_mag[idx : idx + 1]).item()
            scores.append(float(np.clip(proxy, 0.0, 1.0)))
        else:
            scores.append(normalize_pesq_score(score, config))
    return torch.tensor(scores, dtype=torch.float32, device=enhanced_mag.device).view(-1, 1)
