"""ASGI entry point for production servers (Render, Railway, Vercel, Uvicorn)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ai_image_detector.server import create_app  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_default_ckpt = Path(os.getenv("AI_DETECTOR_CHECKPOINT", str(_REPO_ROOT / "runs" / "demo" / "best_model.pt")))
app = create_app(
    checkpoint_path=_default_ckpt if _default_ckpt.is_file() else None,
    device=os.getenv("AI_DETECTOR_DEVICE", "cpu"),
)
