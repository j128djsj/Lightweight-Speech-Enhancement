import numpy as np
import torch
from torch.cuda.amp import autocast

from speech_enhancement.training.losses import complex_channels, generator_loss_components
from speech_enhancement.training.metrics import (
    batch_normalized_pesq_targets,
    compute_pesq_score,
    reconstruct_waveforms,
)
from speech_enhancement.training.utils import move_batch_to_device, set_requires_grad


@torch.no_grad()
def validate(model, discriminator, metric_gan_loss, valid_loader, device, use_amp, eval_limit, config):
    model.eval()
    discriminator.eval()
    set_requires_grad(discriminator, False)

    totals = {"valid_loss": 0.0, "disc_loss": 0.0, "batches": 0}
    enhanced_scores = []
    original_scores = []
    pesq_rows = []
    samples_seen = 0

    for batch in valid_loader:
        batch = move_batch_to_device(batch, device)
        input_audio, input_mag, input_phase, clean_mag, clean_phase, speakers, lengths = batch

        with autocast(enabled=use_amp):
            enhanced_mag, logits = model(input_mag, return_classifier=True)
            losses = generator_loss_components(
                enhanced_mag,
                logits,
                input_phase,
                clean_mag,
                clean_phase,
                speakers,
                lengths,
                metric_gan_loss,
                config,
            )
            clean_spec = complex_channels(clean_mag, clean_phase, config.compress_factor)
            enhanced_spec = complex_channels(enhanced_mag, input_phase, config.compress_factor)

        target_pesq = batch_normalized_pesq_targets(
            clean_mag, clean_phase, enhanced_mag, input_phase, lengths, config
        )
        with autocast(enabled=use_amp):
            disc_loss = metric_gan_loss(
                clean_spec, enhanced_spec, target_scores=target_pesq, mode="discriminator"
            )

        totals["valid_loss"] += losses["total"].item()
        totals["disc_loss"] += disc_loss.item()
        totals["batches"] += 1

        clean_wav = reconstruct_waveforms(clean_mag, clean_phase, lengths, 0, config)
        enhanced_wav = reconstruct_waveforms(enhanced_mag, input_phase, lengths, config.gla_iters, config)
        samples_seen = collect_pesq_rows(
            clean_wav,
            enhanced_wav,
            input_audio,
            lengths,
            samples_seen,
            eval_limit,
            enhanced_scores,
            original_scores,
            pesq_rows,
            config,
        )
        if samples_seen >= eval_limit:
            break

    batches = max(totals["batches"], 1)
    return {
        "valid_loss": totals["valid_loss"] / batches,
        "disc_valid_loss": totals["disc_loss"] / batches,
        "original_pesq": float(np.mean(original_scores)) if original_scores else 0.0,
        "enhanced_pesq": float(np.mean(enhanced_scores)) if enhanced_scores else 0.0,
        "pesq_rows": pesq_rows,
    }


def collect_pesq_rows(
    clean_wav,
    enhanced_wav,
    noisy_wav,
    lengths,
    samples_seen,
    eval_limit,
    enhanced_scores,
    original_scores,
    rows,
    config,
):
    for idx in range(clean_wav.size(0)):
        if samples_seen >= eval_limit:
            break
        length = int(lengths[idx].item())
        clean_np = clean_wav[idx, :length].detach().cpu().numpy()
        enhanced_np = enhanced_wav[idx, :length].detach().cpu().numpy()
        noisy_np = noisy_wav[idx, :length].detach().cpu().numpy()

        enhanced_score = compute_pesq_score(clean_np, enhanced_np, config.sampling_rate)
        original_score = compute_pesq_score(clean_np, noisy_np, config.sampling_rate)
        if enhanced_score is not None:
            enhanced_scores.append(enhanced_score)
        if original_score is not None:
            original_scores.append(original_score)
        if enhanced_score is not None and original_score is not None:
            rows.append(
                {
                    "sample_id": samples_seen,
                    "original_pesq": original_score,
                    "enhanced_pesq": enhanced_score,
                    "pesq_improvement": enhanced_score - original_score,
                }
            )
        samples_seen += 1
    return samples_seen
