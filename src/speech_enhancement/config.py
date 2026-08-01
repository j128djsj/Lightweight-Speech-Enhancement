from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PaperConfig:
    """Default settings from the DSP paper."""

    sampling_rate: int = 16000
    segment_size: int = 32000
    n_fft: int = 400
    hop_size: int = 100
    win_size: int = 400
    compress_factor: float = 0.5
    gla_iters: int = 3

    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 5e-4
    lr_gamma: float = 0.99
    grad_clip: float = 1.0

    train_snrs: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)
    eval_snrs: tuple[float, ...] = (2.5, 7.5, 12.5, 17.5)

    lambda_mag: float = 0.9
    lambda_consistency: float = 0.1
    lambda_complex: float = 0.1
    lambda_metric: float = 0.05
    lambda_classifier: float = 0.1

    min_pesq: float = -0.5
    max_pesq: float = 4.5

    results_dir: Path = PROJECT_ROOT / "results"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    train_noisy_json: Path = PROJECT_ROOT / "json" / "train_noisy.json"
    valid_noisy_json: Path = PROJECT_ROOT / "json" / "valid_noisy.json"
