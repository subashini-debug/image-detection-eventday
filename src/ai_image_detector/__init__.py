"""Hybrid real vs. AI-generated image detection package."""

from ai_image_detector.config import DataConfig, ModelConfig, TrainConfig
from ai_image_detector.model import HybridAIDetector
from ai_image_detector.predictor import AIDetectorPredictor, PredictionResult

__all__ = [
    "AIDetectorPredictor",
    "DataConfig",
    "HybridAIDetector",
    "ModelConfig",
    "PredictionResult",
    "TrainConfig",
]

__version__ = "0.1.0"
