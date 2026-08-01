import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

from speech_enhancement.config import PaperConfig
from speech_enhancement.data import build_speaker_map_from_jsons
from speech_enhancement.models import MetricDiscriminator, MetricGANLoss, SpeechEnhancementModel
from speech_enhancement.training.checkpointing import save_checkpoint
from speech_enhancement.training.data_loading import make_dataset, make_loader
from speech_enhancement.training.evaluation import validate
from speech_enhancement.training.reporting import build_epoch_row, log_batch, log_epoch
from speech_enhancement.training.steps import (
    discriminator_step,
    generator_step,
    unpack_training_batch,
)
from speech_enhancement.training.utils import (
    move_batch_to_device,
    set_random_seed,
    setup_logger,
    write_metrics_csv,
)


def train_from_args(args):
    config = build_config(args)
    logger = setup_logger(config.results_dir / "training.log")
    set_random_seed(args.seed)

    paths = build_json_paths(args, config)
    validate_input_files(paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Device: %s", device)

    speaker_to_idx = build_speaker_map_from_jsons(paths["train_noisy"])
    if not speaker_to_idx:
        raise RuntimeError("No speaker IDs were found. Check your training JSON files.")
    logger.info("Speakers for classifier loss: %d", len(speaker_to_idx))

    loaders = build_loaders(args, config, paths, speaker_to_idx)
    model = SpeechEnhancementModel(num_speakers=len(speaker_to_idx)).to(device)
    discriminator = MetricDiscriminator(input_channels=4).to(device)
    metric_gan_loss = MetricGANLoss(discriminator)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    disc_scheduler = optim.lr_scheduler.ExponentialLR(disc_optimizer, gamma=config.lr_gamma)
    scaler = GradScaler(enabled=use_amp)
    disc_scaler = GradScaler(enabled=use_amp)

    best_pesq = -float("inf")
    history = []
    start_time = time.time()

    for epoch in range(config.epochs):
        train_stats = train_one_epoch(
            model,
            discriminator,
            metric_gan_loss,
            loaders["train"],
            optimizer,
            disc_optimizer,
            scaler,
            disc_scaler,
            device,
            use_amp,
            config,
            args.log_interval,
            logger,
            epoch,
        )

        valid_stats = validate(
            model,
            discriminator,
            metric_gan_loss,
            loaders["valid"],
            device,
            use_amp,
            args.eval_limit,
            config,
        )
        row = build_epoch_row(epoch, train_stats, valid_stats, optimizer, disc_optimizer)
        history.append(row)
        write_metrics_csv(config.results_dir / "training_history.csv", history)
        write_metrics_csv(config.results_dir / "pesq_analysis" / f"epoch_{epoch + 1}_pesq.csv", valid_stats["pesq_rows"])

        if valid_stats["enhanced_pesq"] > best_pesq:
            best_pesq = valid_stats["enhanced_pesq"]
            save_training_checkpoint(
                config, "best_model_metricgan.pth", epoch, model, discriminator,
                optimizer, disc_optimizer, row, speaker_to_idx
            )
            logger.info("Saved new best checkpoint with PESQ %.3f", best_pesq)

        if (epoch + 1) % args.save_interval == 0:
            save_training_checkpoint(
                config, f"epoch_{epoch + 1}_model.pth", epoch, model, discriminator,
                optimizer, disc_optimizer, row, speaker_to_idx
            )

        scheduler.step()
        disc_scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_epoch(logger, config.epochs, row, time.time() - train_stats["epoch_start"])

    logger.info("Training complete in %.2f minutes", (time.time() - start_time) / 60)


def train_one_epoch(
    model,
    discriminator,
    metric_gan_loss,
    train_loader,
    optimizer,
    disc_optimizer,
    scaler,
    disc_scaler,
    device,
    use_amp,
    config,
    log_interval,
    logger,
    epoch,
):
    model.train()
    discriminator.train()
    stats = {"train_loss": 0.0, "disc_train_loss": 0.0, "batches": 0, "epoch_start": time.time()}

    for batch_idx, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch, device)
        input_mag, input_phase, clean_mag, clean_phase, speakers, lengths = unpack_training_batch(batch)

        disc_loss = discriminator_step(
            model, discriminator, metric_gan_loss, clean_mag, clean_phase,
            input_mag, input_phase, lengths, disc_optimizer, disc_scaler, use_amp, config
        )
        losses, gen_grad = generator_step(
            model, discriminator, metric_gan_loss, input_mag, input_phase, clean_mag,
            clean_phase, speakers, lengths, optimizer, scaler, use_amp, config
        )

        stats["train_loss"] += losses["total"].item()
        stats["disc_train_loss"] += disc_loss.item()
        stats["batches"] += 1
        if batch_idx > 0 and batch_idx % log_interval == 0:
            log_batch(logger, epoch, batch_idx, len(train_loader), losses, disc_loss, gen_grad)

    batches = max(stats["batches"], 1)
    stats["train_loss"] /= batches
    stats["disc_train_loss"] /= batches
    return stats

def build_config(args):
    return replace(
        PaperConfig(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        lr_gamma=args.lr_gamma,
        grad_clip=args.grad_clip,
    )


def build_json_paths(args, config):
    return {
        "train_noisy": Path(args.train_noisy_json or config.train_noisy_json),
        "valid_noisy": Path(args.valid_noisy_json or config.valid_noisy_json),
    }


def validate_input_files(paths):
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset JSON files:\n" + "\n".join(missing))


def build_loaders(args, config, paths, speaker_to_idx):
    train_dataset = make_dataset(config, paths, True, speaker_to_idx)
    valid_dataset = make_dataset(config, paths, False, speaker_to_idx, args.eval_full_length)
    return {
        "train": make_loader(train_dataset, config.batch_size, shuffle=True, drop_last=True),
        "valid": make_loader(
            valid_dataset,
            1 if args.eval_full_length else config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        ),
    }

def save_training_checkpoint(config, filename, *checkpoint_args):
    save_checkpoint(config.checkpoints_dir / filename, *checkpoint_args, config=config)
