import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_enhancement.models import SpeechEnhancementModel


def format_count(value):
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Count model parameters.")
    parser.add_argument("--num_speakers", type=int, default=None)
    args = parser.parse_args()

    model = SpeechEnhancementModel(num_speakers=args.num_speakers)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {format_count(total)} ({total})")
    print(f"Trainable params: {format_count(trainable)} ({trainable})")
    print("\nBy module:")
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {format_count(count)} ({count})")


if __name__ == "__main__":
    main()
