import math

import torch
import torch.nn as nn
from einops import rearrange


class LightS4Block(nn.Module):
    """Diagonal S4 block implemented as FFT convolution."""

    def __init__(self, input_dim, state_dim=256, activation="silu"):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.a_log = nn.Parameter(torch.empty(state_dim))
        self.b = nn.Parameter(torch.empty(state_dim))
        self.c = nn.Parameter(torch.empty(input_dim, state_dim))
        self.d = nn.Parameter(torch.ones(input_dim))
        self.log_step = nn.Parameter(torch.empty(1))
        self.activation = nn.SiLU() if activation == "silu" else nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.a_log, -1.0, 0.0)
        nn.init.normal_(self.b, std=1.0)
        nn.init.normal_(self.c, std=1.0 / math.sqrt(self.state_dim))
        nn.init.constant_(self.log_step, math.log(0.01))

    def compute_kernel(self, length):
        device = self.a_log.device
        step = torch.exp(self.log_step)
        a = -torch.exp(self.a_log)
        dt_a = a * step
        t = torch.arange(length, device=device, dtype=a.dtype)
        a_powers = torch.exp(dt_a.unsqueeze(1) * t.unsqueeze(0))
        b_bar = (torch.exp(dt_a) - 1.0) / (a + 1e-7) * self.b
        return torch.einsum("in,n,nt->it", self.c, b_bar, a_powers)

    def forward(self, x):
        _, _, length = x.shape
        kernel = self.compute_kernel(length)
        fft_length = 2 ** math.ceil(math.log2(2 * length))
        x_f = torch.fft.rfft(x, n=fft_length)
        kernel_f = torch.fft.rfft(kernel, n=fft_length)
        y = torch.fft.irfft(x_f * kernel_f, n=fft_length)[..., :length]
        y = y + x * self.d.view(1, -1, 1)
        return self.activation(y)


class FeatureMask(nn.Module):
    """Dual-path time/frequency LightS4 feature mask."""

    def __init__(self, in_channels):
        super().__init__()
        self.s4_time = LightS4Block(input_dim=in_channels)
        self.s4_freq = LightS4Block(input_dim=in_channels)
        self.bias_t = nn.Parameter(torch.zeros(in_channels))
        self.bias_f = nn.Parameter(torch.zeros(in_channels))
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        batch, channels, freq, time = x.size()

        time_input = rearrange(x, "b c f t -> (b f) c t")
        time_output = self.s4_time(time_input)
        time_output = rearrange(time_output, "(b f) c t -> b c f t", b=batch, f=freq)

        freq_input = rearrange(x, "b c f t -> (b t) c f")
        freq_output = self.s4_freq(freq_input)
        freq_output = rearrange(freq_output, "(b t) c f -> b c f t", b=batch, t=time)

        gate_t = torch.sigmoid(time_output + self.bias_t.view(1, -1, 1, 1))
        gate_f = torch.sigmoid(freq_output + self.bias_f.view(1, -1, 1, 1))
        fused = (time_output * gate_f) + (freq_output * gate_t)
        return self.final_act(fused)
