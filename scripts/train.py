import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_enhancement.config import PaperConfig
from speech_enhancement.training import train_from_args


def parse_args():
    defaults = PaperConfig()
    parser = argparse.ArgumentParser(description="Train the DSP lightweight speech enhancement model.")
    parser.add_argument("--train_noisy_json", type=str, default=None)
    parser.add_argument("--valid_noisy_json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--lr_gamma", type=float, default=defaults.lr_gamma)
    parser.add_argument("--grad_clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--eval_limit", type=int, default=100)
    parser.add_argument("--eval_full_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train_from_args(parse_args())
