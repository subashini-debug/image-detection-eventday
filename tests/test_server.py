"""Tests for the FastAPI web UI and inference endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ai_image_detector.config import ModelConfig
from ai_image_detector.forensics import FORENSIC_DIM
from ai_image_detector.model import HybridAIDetector
from ai_image_detector.server import create_app
from ai_image_detector.trainer import save_checkpoint


def _sample_png_bytes() -> bytes:
    arr = np.random.default_rng(0).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_server_inspect_without_checkpoint() -> None:
    app = create_app(checkpoint_path=None)
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "AI Image Detector" in home.text

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False

    resp = client.post(
        "/api/inspect",
        files={"file": ("test.png", _sample_png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "highlights" in body and "metadata" in body


def test_server_predict_with_checkpoint(tmp_path: Path) -> None:
    cfg = ModelConfig(image_size=96, fft_size=64, base_channels=16, freq_channels=12)
    model = HybridAIDetector(cfg)
    ckpt = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=cfg,
        feature_mean=np.zeros(FORENSIC_DIM, dtype=np.float32),
        feature_std=np.ones(FORENSIC_DIM, dtype=np.float32),
        optimal_threshold=0.5,
        metrics={"val_f1": 0.9},
    )

    app = create_app(checkpoint_path=ckpt, device="cpu")
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    with torch.no_grad():
        resp = client.post(
            "/api/predict?explain=true&tta=true",
            files={"file": ("probe.png", _sample_png_bytes(), "image/png")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"real", "ai_generated"}
    assert 0.0 <= body["ai_probability"] <= 1.0
    assert "forensics" in body
