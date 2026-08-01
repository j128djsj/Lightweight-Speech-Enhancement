import torch


def save_checkpoint(path, epoch, model, discriminator, optimizer, disc_optimizer, metrics, speaker_to_idx, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "discriminator_optimizer_state_dict": disc_optimizer.state_dict(),
            "metrics": metrics,
            "speaker_to_idx": speaker_to_idx,
            "config": {
                "sampling_rate": config.sampling_rate,
                "segment_size": config.segment_size,
                "n_fft": config.n_fft,
                "hop_size": config.hop_size,
                "win_size": config.win_size,
                "compress_factor": config.compress_factor,
                "gla_iters": config.gla_iters,
                "train_snrs": config.train_snrs,
                "eval_snrs": config.eval_snrs,
                "loss_weights": {
                    "mag": config.lambda_mag,
                    "consistency": config.lambda_consistency,
                    "complex": config.lambda_complex,
                    "metric": config.lambda_metric,
                    "classifier": config.lambda_classifier,
                },
            },
        },
        path,
    )
