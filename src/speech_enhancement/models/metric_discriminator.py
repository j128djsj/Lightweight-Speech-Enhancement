import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormDepthwiseConv2d(nn.Module):
    """Spectrally normalized depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        )
        self.pointwise = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MetricDiscriminator(nn.Module):
    """Differentiable PESQ proxy for MetricGAN-style training."""

    def __init__(self, input_channels=4):
        super().__init__()
        channels = [16, 32, 64, 128]
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channels
        for out_ch in channels:
            self.conv_blocks.append(
                nn.Sequential(
                    SpectralNormDepthwiseConv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.PReLU(out_ch),
                )
            )
            in_ch = out_ch

        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        for block in self.conv_blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.sigmoid(self.fc2(F.relu(self.fc1(x))))


class MetricGANLoss(nn.Module):
    """Discriminator and generator objectives from the paper."""

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.mse_loss = nn.MSELoss()

    def discriminator_loss(self, clean_clean_pairs, clean_enhanced_pairs, target_scores):
        clean_clean = self.discriminator(clean_clean_pairs)
        clean_enhanced = self.discriminator(clean_enhanced_pairs)
        target_scores = target_scores.to(clean_enhanced.device).view_as(clean_enhanced)
        return self.mse_loss(clean_clean, torch.ones_like(clean_clean)) + self.mse_loss(
            clean_enhanced, target_scores
        )

    def generator_loss(self, clean_enhanced_pairs):
        predicted_score = self.discriminator(clean_enhanced_pairs)
        return self.mse_loss(predicted_score, torch.ones_like(predicted_score))

    def forward(self, clean_spec, enhanced_spec, target_scores=None, mode="generator"):
        clean_enhanced = torch.cat([clean_spec, enhanced_spec], dim=1)
        if mode == "generator":
            return self.generator_loss(clean_enhanced)
        if mode == "discriminator":
            clean_clean = torch.cat([clean_spec, clean_spec], dim=1)
            return self.discriminator_loss(clean_clean, clean_enhanced, target_scores)
        raise ValueError("mode must be 'generator' or 'discriminator'")
