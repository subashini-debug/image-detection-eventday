"""Configuration dataclasses for model architecture, data loading, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Architecture hyperparameters for :class:`HybridAIDetector`."""

    image_size: int = 224
    fft_size: int = 128
    base_channels: int = 32
    freq_channels: int = 24
    forensic_dim: int = 32
    forensic_hidden_dim: int = 64
    head_hidden_dim: int = 128
    dropout: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DataConfig:
    """Dataset splitting and augmentation configuration."""

    image_size: int = 224
    fft_size: int = 128
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    jpeg_aug_prob: float = 0.35
    jpeg_quality_min: int = 65
    jpeg_quality_max: int = 98


@dataclass
class TrainConfig:
    """Optimization, regularization, and calibration configuration."""

    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.03
    early_stopping_patience: int = 5
    grad_clip_norm: float = 2.0
    calibrate_temperature: bool = True
    tune_threshold: bool = True
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
