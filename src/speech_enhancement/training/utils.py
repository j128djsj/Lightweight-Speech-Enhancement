import csv
import logging
import random

import numpy as np
import torch


def setup_logger(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("speech_enhancement")


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_last_dim(tensor, target_size):
    pad = target_size - tensor.size(-1)
    return torch.nn.functional.pad(tensor, (0, pad)) if pad > 0 else tensor


def move_batch_to_device(batch, device):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad_(enabled)


def write_metrics_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
