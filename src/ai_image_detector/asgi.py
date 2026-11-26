"""ASGI entry point for production servers (Render, Railway, Gunicorn/Uvicorn)."""

from __future__ import annotations

import os
from pathlib import Path

from ai_image_detector.server import create_app

_default_ckpt = Path(os.getenv("AI_DETECTOR_CHECKPOINT", "runs/demo/best_model.pt"))
app = create_app(
    checkpoint_path=_default_ckpt if _default_ckpt.is_file() else None,
    device=os.getenv("AI_DETECTOR_DEVICE", "cpu"),
)
