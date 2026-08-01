import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from thop import profile

from speech_enhancement.models import SpeechEnhancementModel


def main():
    parser = argparse.ArgumentParser(description="Profile inference MACs with THOP.")
    parser.add_argument("--freq_bins", type=int, default=201)
    parser.add_argument("--time_frames", type=int, default=321)
    args = parser.parse_args()

    model = SpeechEnhancementModel().eval()
    example = torch.randn(1, 1, args.freq_bins, args.time_frames)
    macs, params = profile(model, inputs=(example,), verbose=False)
    print(f"MACs: {macs / 1e9:.4f} G")
    print(f"Params: {params / 1e6:.4f} M")


if __name__ == "__main__":
    main()
