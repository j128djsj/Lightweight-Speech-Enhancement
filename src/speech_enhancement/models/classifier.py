import torch.nn as nn


class SpeakerClassifier(nn.Module):
    """Training-only classifier used by the paper's classifier loss."""

    def __init__(self, input_channels, num_speakers, hidden_channels=512):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_speakers),
        )

    def forward(self, x):
        return self.classifier(self.global_pool(x).flatten(1))
