"""Tests for forensic, spectral, and metadata extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from ai_image_detector.forensics import (
    FORENSIC_DIM,
    FORENSIC_FEATURE_NAMES,
    compute_fft_spectrum,
    compute_srm_residuals,
    extract_forensic_features,
    inspect_metadata,
    summarize_forensics,
)


def test_forensic_feature_dimensions_and_finiteness() -> None:
    rng = np.random.default_rng(123)
    rgb = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    srm = compute_srm_residuals(rgb)
    assert srm.shape == (3, 128, 128)
    assert srm.dtype == np.float32
    assert np.all(np.isfinite(srm))
    assert float(srm.min()) >= -1.0 and float(srm.max()) <= 1.0

    fft_map = compute_fft_spectrum(rgb, fft_size=64)
    assert fft_map.shape == (1, 64, 64)
    assert np.all(np.isfinite(fft_map))
    assert float(fft_map.min()) >= 0.0 and float(fft_map.max()) <= 1.0 + 1e-6

    vec = extract_forensic_features(rgb)
    assert vec.shape == (FORENSIC_DIM,)
    assert len(FORENSIC_FEATURE_NAMES) == FORENSIC_DIM
    assert np.all(np.isfinite(vec))


def test_forensics_handles_constant_image() -> None:
    constant = np.full((96, 96, 3), 128, dtype=np.uint8)
    vec = extract_forensic_features(constant)
    assert vec.shape == (FORENSIC_DIM,)
    assert np.all(np.isfinite(vec))


def test_metadata_inspection_detects_ai_png_tags(tmp_path: Path) -> None:
    img_path = tmp_path / "generated.png"
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "Prompt: landscape\nModel: Stable Diffusion XL")
    img.save(img_path, format="PNG", pnginfo=meta)

    info = inspect_metadata(img_path)
    assert info["format"] == "PNG"
    assert info["width"] == 64
    assert "stable diffusion" in info["ai_metadata_matches"]

    summary = summarize_forensics(np.zeros((64, 64, 3), dtype=np.uint8), image_path=img_path)
    assert "highlights" in summary
    assert "metadata" in summary
