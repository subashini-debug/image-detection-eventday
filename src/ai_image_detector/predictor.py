"""Inference engine for classifying single images or directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai_image_detector.config import ModelConfig
from ai_image_detector.dataset import IMAGE_EXTENSIONS, prepare_model_inputs
from ai_image_detector.forensics import load_rgb_image, summarize_forensics
from ai_image_detector.trainer import load_checkpoint, resolve_device


@dataclass
class PredictionResult:
    """Structured inference output for one image."""

    image_path: str
    label: str  # "ai_generated" | "real"
    ai_probability: float
    real_probability: float
    confidence: float
    threshold: float
    forensics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AIDetectorPredictor:
    """Runs calibrated inference and optional forensic explanation on images."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        threshold: float | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model, payload = load_checkpoint(checkpoint_path, device=self.device)
        self.model_config = ModelConfig.from_dict(payload["model_config"])
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(payload.get("optimal_threshold", 0.5))
        )

    @torch.no_grad()
    def _score_rgb(self, rgb: np.ndarray) -> float:
        spatial, fft_map, forensic_vec = prepare_model_inputs(
            rgb=rgb,
            image_size=self.model_config.image_size,
            fft_size=self.model_config.fft_size,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        prob = self.model.predict_proba(
            spatial.unsqueeze(0).to(self.device),
            fft_map.unsqueeze(0).to(self.device),
            forensic_vec.unsqueeze(0).to(self.device),
        )
        return float(prob.squeeze().item())

    def predict_image(
        self,
        image_path: str | Path,
        tta: bool = True,
        explain: bool = False,
    ) -> PredictionResult:
        """Predict whether a single image is real or AI-generated."""
        path = Path(image_path)
        rgb = load_rgb_image(path)

        probs = [self._score_rgb(rgb)]
        if tta:
            flipped = np.ascontiguousarray(rgb[:, ::-1, :])
            probs.append(self._score_rgb(flipped))
            h, w = rgb.shape[:2]
            if min(h, w) >= 64:
                ch, cw = int(h * 0.85), int(w * 0.85)
                top = (h - ch) // 2
                left = (w - cw) // 2
                center_crop = np.ascontiguousarray(rgb[top : top + ch, left : left + cw, :])
                probs.append(self._score_rgb(center_crop))

        ai_prob = float(np.mean(probs))
        real_prob = 1.0 - ai_prob
        is_ai = ai_prob >= self.threshold
        label = "ai_generated" if is_ai else "real"
        confidence = ai_prob if is_ai else real_prob

        forensics = summarize_forensics(rgb, image_path=path) if explain else None
        return PredictionResult(
            image_path=str(path),
            label=label,
            ai_probability=round(ai_prob, 4),
            real_probability=round(real_prob, 4),
            confidence=round(confidence, 4),
            threshold=round(self.threshold, 4),
            forensics=forensics,
        )

    def predict_path(
        self,
        target: str | Path,
        tta: bool = True,
        explain: bool = False,
    ) -> list[PredictionResult]:
        """Predict on either a single image file or all images inside a directory."""
        path = Path(target)
        if path.is_file():
            return [self.predict_image(path, tta=tta, explain=explain)]
        if path.is_dir():
            files = [
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not files:
                raise ValueError(f"No supported image files found in directory: {path}")
            return [self.predict_image(f, tta=tta, explain=explain) for f in files]
        raise FileNotFoundError(f"Target path not found: {path}")
