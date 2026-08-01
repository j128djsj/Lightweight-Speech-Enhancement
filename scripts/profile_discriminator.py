import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from speech_enhancement.models import MetricDiscriminator


def main():
    model = MetricDiscriminator()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MetricDiscriminator params: {total:,} ({total / 1e6:.4f}M)")

    example = torch.randn(1, 4, 201, 321)
    output = model(example)
    print(f"Input shape: {tuple(example.shape)}")
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
