"""Tests for the HybridAIDetector architecture and checkpoint serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ai_image_detector.config import ModelConfig
from ai_image_detector.forensics import FORENSIC_DIM
from ai_image_detector.model import HybridAIDetector
from ai_image_detector.trainer import load_checkpoint, save_checkpoint


def test_hybrid_detector_forward_and_backward() -> None:
    cfg = ModelConfig(image_size=128, fft_size=64, base_channels=16, freq_channels=12)
    model = HybridAIDetector(cfg)
    assert model.count_parameters() > 0

    spatial = torch.randn(4, 6, 128, 128)
    fft_map = torch.rand(4, 1, 64, 64)
    forensic = torch.randn(4, FORENSIC_DIM)

    out = model.forward_features(spatial, fft_map, forensic)
    assert out["logit"].shape == (4,)
    assert out["spatial_emb"].shape == (4, cfg.base_channels * 8)
    assert out["freq_emb"].shape == (4, cfg.freq_channels * 6)
    assert out["forensic_emb"].shape == (4, cfg.forensic_hidden_dim)

    probs = model.predict_proba(spatial, fft_map, forensic)
    assert probs.shape == (4,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))

    loss = out["logit"].pow(2).mean()
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = ModelConfig(image_size=96, fft_size=64, base_channels=16, freq_channels=12)
    model = HybridAIDetector(cfg)
    model.temperature.fill_(1.35)
    model.eval()

    mean = np.zeros(FORENSIC_DIM, dtype=np.float32)
    std = np.ones(FORENSIC_DIM, dtype=np.float32)
    ckpt = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        model_config=cfg,
        feature_mean=mean,
        feature_std=std,
        optimal_threshold=0.48,
        metrics={"val_f1": 0.9},
    )

    loaded, payload = load_checkpoint(ckpt, device="cpu")
    assert payload["optimal_threshold"] == 0.48
    assert abs(float(loaded.temperature.item()) - 1.35) < 1e-5

    spatial = torch.randn(2, 6, 96, 96)
    fft_map = torch.rand(2, 1, 64, 64)
    forensic = torch.randn(2, FORENSIC_DIM)

    with torch.no_grad():
        p1 = model.predict_proba(spatial, fft_map, forensic)
        p2 = loaded.predict_proba(spatial, fft_map, forensic)
    assert torch.allclose(p1, p2, atol=1e-6)
