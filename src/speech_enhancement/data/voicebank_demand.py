import json
import random
import threading
from functools import lru_cache

import librosa
import numpy as np
import torch
import torch.utils.data

from speech_enhancement.data.speakers import build_speaker_map_from_jsons, extract_speaker_id
from speech_enhancement.dsp.stft import mag_phase_stft


class VoiceBankDemandDataset(torch.utils.data.Dataset):
    """VoiceBank+DEMAND pairs represented by paired clean/noise JSON items."""

    def __init__(
        self,
        noisy_json,
        sampling_rate=16000,
        segment_size=32000,
        n_fft=400,
        hop_size=100,
        win_size=400,
        compress_factor=0.5,
        is_train=True,
        max_amplitude=0.95,
        snr_choices=None,
        cache_size=100,
        speaker_to_idx=None,
    ):
        self.noisy_items = self._load_json(noisy_json)
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.is_train = is_train
        self.max_amplitude = max_amplitude
        self.snr_choices = tuple(snr_choices) if snr_choices is not None else None

        self.speaker_to_idx = speaker_to_idx or build_speaker_map_from_jsons(noisy_json)
        self.num_speakers = len(self.speaker_to_idx)
        self._cache_lock = threading.Lock()
        self._init_audio_cache(cache_size)

    @staticmethod
    def _load_json(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_audio_cache(self, cache_size):
        @lru_cache(maxsize=cache_size)
        def cached_load(audio_path):
            audio, orig_sr = librosa.load(audio_path, sr=None, duration=None)
            if orig_sr != self.sampling_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=orig_sr,
                    target_sr=self.sampling_rate,
                    res_type="kaiser_best",
                )
            return audio.astype(np.float32)

        self._cached_load_audio = cached_load

    def _load_audio(self, audio_path):
        with self._cache_lock:
            return self._cached_load_audio(audio_path).copy()

    def _choose_snr(self, index):
        if not self.snr_choices:
            return 12.5
        if self.is_train:
            return float(random.choice(self.snr_choices))
        return float(self.snr_choices[index % len(self.snr_choices)])

    def _mix_audio(self, clean, noise, snr):
        clean_len = len(clean)
        clean = torch.as_tensor(clean, dtype=torch.float32)
        noise = torch.as_tensor(noise, dtype=torch.float32)

        if len(noise) > clean_len:
            start = random.randint(0, len(noise) - clean_len)
            noise_segment = noise[start : start + clean_len]
        else:
            repeat = int(np.ceil(clean_len / max(len(noise), 1)))
            noise_segment = torch.tile(noise, (repeat,))[:clean_len]

        clean_power = torch.sum(clean**2) + 1e-10
        noise_power = torch.sum(noise_segment**2) + 1e-10
        scale = torch.sqrt(clean_power / (noise_power * (10 ** (snr / 10))))
        noisy_audio = clean + scale * noise_segment
        actual_len = min(len(clean), len(noisy_audio))
        return clean[:actual_len], noisy_audio[:actual_len]

    def _normalize_audio(self, clean_audio, noisy_audio):
        clean_audio = torch.as_tensor(clean_audio, dtype=torch.float32)
        noisy_audio = torch.as_tensor(noisy_audio, dtype=torch.float32)
        norm = torch.sqrt(
            torch.tensor(len(noisy_audio), dtype=torch.float32)
            / (torch.sum(noisy_audio**2.0) + 1e-10)
        )
        clean_audio = torch.clamp(clean_audio * norm, -self.max_amplitude, self.max_amplitude)
        noisy_audio = torch.clamp(noisy_audio * norm, -self.max_amplitude, self.max_amplitude)
        return clean_audio, noisy_audio

    def _crop_or_pad(self, clean_audio, noisy_audio):
        audio_len = clean_audio.size(1)
        if self.segment_size is None:
            if audio_len < self.win_size:
                pad = self.win_size - audio_len
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad))
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad))
            return clean_audio, noisy_audio, audio_len

        if audio_len >= self.segment_size:
            start = random.randint(0, audio_len - self.segment_size) if self.is_train else (
                audio_len - self.segment_size
            ) // 2
            clean_audio = clean_audio[:, start : start + self.segment_size]
            noisy_audio = noisy_audio[:, start : start + self.segment_size]
        else:
            pad = self.segment_size - audio_len
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad))
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad))
        return clean_audio, noisy_audio, min(audio_len, self.segment_size)

    def __getitem__(self, index):
        item = self.noisy_items[index]
        clean_path = item["clean"]
        speaker_id = extract_speaker_id(clean_path)
        speaker_label = self.speaker_to_idx.get(speaker_id, -1)

        clean_audio = self._load_audio(clean_path)
        noise_audio = self._load_audio(item["noise"])
        clean_audio, noisy_audio = self._mix_audio(clean_audio, noise_audio, self._choose_snr(index))

        clean_audio, noisy_audio = self._normalize_audio(clean_audio, noisy_audio)
        clean_audio, noisy_audio, audio_len = self._crop_or_pad(
            clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        )

        clean_mag, clean_phase, _ = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )
        input_mag, input_phase, _ = mag_phase_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )

        return (
            noisy_audio.squeeze(0),
            torch.clamp(input_mag, 0, 1e4),
            input_phase,
            torch.clamp(clean_mag, 0, 1e4),
            clean_phase,
            torch.tensor(speaker_label, dtype=torch.long),
            torch.tensor(audio_len, dtype=torch.long),
        )

    def __len__(self):
        return len(self.noisy_items)
