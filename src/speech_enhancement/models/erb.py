import numpy as np
import torch
import torch.nn as nn


class AuditorySpectralCompressor(nn.Module):
    """Fixed ERB filter-bank compressor used by the AISC module."""

    def __init__(self, preserved_bins, erb_bands, nfft=400, high_lim=8000, fs=16000):
        super().__init__()
        filters = self.erb_filter_banks(preserved_bins, erb_bands, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.preserved_bins = preserved_bins
        self.erb_fc = nn.Linear(nfreqs - preserved_bins, erb_bands, bias=False)
        self.ierb_fc = nn.Linear(erb_bands, nfreqs - preserved_bins, bias=False)
        self.erb_fc.weight = nn.Parameter(filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(filters.T, requires_grad=False)

    @staticmethod
    def hz_to_erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def erb_to_hz(erb_value):
        return (10 ** (erb_value / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, preserved_bins, erb_bands, nfft=400, high_lim=8000, fs=16000):
        low_lim = preserved_bins / nfft * fs
        erb_low = self.hz_to_erb(low_lim)
        erb_high = self.hz_to_erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_bands)
        bins = np.round(self.erb_to_hz(erb_points) / fs * nfft).astype(np.int32)
        filters = np.zeros([erb_bands, nfft // 2 + 1], dtype=np.float32)

        filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)

        for idx in range(erb_bands - 2):
            filters[idx + 1, bins[idx] : bins[idx + 1]] = (
                np.arange(bins[idx], bins[idx + 1]) - bins[idx] + 1e-12
            ) / (bins[idx + 1] - bins[idx] + 1e-12)
            filters[idx + 1, bins[idx + 1] : bins[idx + 2]] = (
                bins[idx + 2] - np.arange(bins[idx + 1], bins[idx + 2]) + 1e-12
            ) / (bins[idx + 2] - bins[idx + 1] + 1e-12)

        filters[-1, bins[-2] : bins[-1] + 1] = 1 - filters[-2, bins[-2] : bins[-1] + 1]
        return torch.from_numpy(np.abs(filters[:, preserved_bins:]))

    def compress(self, x):
        x_low = x[..., : self.preserved_bins]
        x_high = self.erb_fc(x[..., self.preserved_bins :])
        return torch.cat([x_low, x_high], dim=-1)

    def decompress(self, x_erb):
        x_low = x_erb[..., : self.preserved_bins]
        x_high = self.ierb_fc(x_erb[..., self.preserved_bins :])
        return torch.cat([x_low, x_high], dim=-1)
