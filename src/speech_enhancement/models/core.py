import torch.nn.functional as F
from torch import nn

from speech_enhancement.models.classifier import SpeakerClassifier
from speech_enhancement.models.decoder import Decoder
from speech_enhancement.models.encoder import Encoder
from speech_enhancement.models.erb import AuditorySpectralCompressor
from speech_enhancement.models.feature_mask import FeatureMask


class SpeechEnhancementModel(nn.Module):
    """AISC + DSConv/ASPP encoder + LightS4 FeatureMask + decoder."""

    def __init__(self, num_speakers=None):
        super().__init__()
        self.aisc = AuditorySpectralCompressor(38, 37, nfft=400)
        self.encoder = Encoder(in_channels=1, aspp_channels=512 * 5 // 2)
        self.feature_mask = FeatureMask(in_channels=1536 // 2)
        self.decoder = Decoder(in_channels=1536 // 2)
        self.classifier = (
            SpeakerClassifier(input_channels=1536 // 2, num_speakers=num_speakers)
            if num_speakers is not None and num_speakers > 0
            else None
        )

    def forward(self, x, return_classifier=False):
        original_size = x.shape[2:4]
        x_erb = self.aisc.compress(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        encoded = self.encoder(x_erb)
        masked = encoded * self.feature_mask(encoded)

        logits = None
        if return_classifier:
            if self.classifier is None:
                raise RuntimeError("Create the model with num_speakers to use classifier loss.")
            logits = self.classifier(masked)

        decoded_erb = self.decoder(masked, x_erb.shape[2:4])
        decoded = self.aisc.decompress(decoded_erb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        if decoded.shape[2:] != original_size:
            decoded = F.interpolate(decoded, size=original_size, mode="bilinear", align_corners=True)

        if return_classifier:
            return decoded, logits
        return decoded
