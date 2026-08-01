from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

from speech_enhancement.training.losses import complex_channels, generator_loss_components
from speech_enhancement.training.metrics import batch_normalized_pesq_targets
from speech_enhancement.training.utils import set_requires_grad


def unpack_training_batch(batch):
    _, input_mag, input_phase, clean_mag, clean_phase, speakers, lengths = batch
    return input_mag, input_phase, clean_mag, clean_phase, speakers, lengths


def discriminator_step(
    model,
    discriminator,
    metric_gan_loss,
    clean_mag,
    clean_phase,
    input_mag,
    input_phase,
    lengths,
    optimizer,
    scaler,
    use_amp,
    config,
):
    optimizer.zero_grad(set_to_none=True)
    set_requires_grad(discriminator, True)
    with autocast(enabled=use_amp):
        enhanced_mag = model(input_mag).detach()
        clean_spec = complex_channels(clean_mag, clean_phase, config.compress_factor)
        enhanced_spec = complex_channels(enhanced_mag, input_phase, config.compress_factor)

    target_pesq = batch_normalized_pesq_targets(
        clean_mag, clean_phase, enhanced_mag, input_phase, lengths, config
    )
    with autocast(enabled=use_amp):
        loss = metric_gan_loss(clean_spec, enhanced_spec, target_scores=target_pesq, mode="discriminator")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(discriminator.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


def generator_step(
    model,
    discriminator,
    metric_gan_loss,
    input_mag,
    input_phase,
    clean_mag,
    clean_phase,
    speakers,
    lengths,
    optimizer,
    scaler,
    use_amp,
    config,
):
    optimizer.zero_grad(set_to_none=True)
    set_requires_grad(discriminator, False)
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

    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return losses, grad_norm
