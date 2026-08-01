def build_epoch_row(epoch, train_stats, valid_stats, optimizer, disc_optimizer):
    return {
        "epoch": epoch + 1,
        "train_loss": train_stats["train_loss"],
        "disc_train_loss": train_stats["disc_train_loss"],
        "valid_loss": valid_stats["valid_loss"],
        "disc_valid_loss": valid_stats["disc_valid_loss"],
        "original_pesq": valid_stats["original_pesq"],
        "enhanced_pesq": valid_stats["enhanced_pesq"],
        "pesq_improvement": valid_stats["enhanced_pesq"] - valid_stats["original_pesq"],
        "lr": optimizer.param_groups[0]["lr"],
        "disc_lr": disc_optimizer.param_groups[0]["lr"],
    }


def log_batch(logger, epoch, batch_idx, total_batches, losses, disc_loss, grad_norm):
    logger.info(
        "Epoch %d Batch %d/%d | G %.5f | D %.5f | LMag %.5f LCon %.5f LCom %.5f "
        "LMetric %.5f LCls %.5f | grad %.3f",
        epoch + 1,
        batch_idx,
        total_batches,
        losses["total"].item(),
        disc_loss.item(),
        losses["mag"].item(),
        losses["consistency"].item(),
        losses["complex"].item(),
        losses["metric"].item(),
        losses["classifier"].item(),
        float(grad_norm),
    )


def log_epoch(logger, total_epochs, row, elapsed):
    logger.info(
        "Epoch %d/%d | train %.5f disc %.5f | valid %.5f valid_disc %.5f | "
        "PESQ %.3f -> %.3f (+%.3f) | lr %.2e | %.1fs",
        row["epoch"],
        total_epochs,
        row["train_loss"],
        row["disc_train_loss"],
        row["valid_loss"],
        row["disc_valid_loss"],
        row["original_pesq"],
        row["enhanced_pesq"],
        row["pesq_improvement"],
        row["lr"],
        elapsed,
    )
