import multiprocessing
import os

import torch
from torch.utils.data import DataLoader

from speech_enhancement.data import VoiceBankDemandDataset
from speech_enhancement.training.utils import pad_last_dim


def collate_speech_batch(batch):
    input_audio, input_mag, input_phase, clean_mag, clean_phase, speaker, length = zip(*batch)
    max_audio_len = max(item.size(-1) for item in input_audio)
    max_frames = max(item.size(-1) for item in input_mag)

    return (
        torch.stack([pad_last_dim(item, max_audio_len) for item in input_audio]),
        torch.stack([pad_last_dim(item, max_frames) for item in input_mag]),
        torch.stack([pad_last_dim(item, max_frames) for item in input_phase]),
        torch.stack([pad_last_dim(item, max_frames) for item in clean_mag]),
        torch.stack([pad_last_dim(item, max_frames) for item in clean_phase]),
        torch.stack(speaker),
        torch.stack(length),
    )


def make_dataset(config, paths, is_train, speaker_to_idx, eval_full_length=False):
    return VoiceBankDemandDataset(
        noisy_json=str(paths["train_noisy"] if is_train else paths["valid_noisy"]),
        sampling_rate=config.sampling_rate,
        segment_size=config.segment_size if is_train or not eval_full_length else None,
        n_fft=config.n_fft,
        hop_size=config.hop_size,
        win_size=config.win_size,
        compress_factor=config.compress_factor,
        is_train=is_train,
        snr_choices=config.train_snrs if is_train else config.eval_snrs,
        cache_size=100,
        speaker_to_idx=speaker_to_idx,
    )


def make_loader(dataset, batch_size, shuffle, drop_last=False, num_workers=None):
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else min(multiprocessing.cpu_count(), 8)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_speech_batch,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)
