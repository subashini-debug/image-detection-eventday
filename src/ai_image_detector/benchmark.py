"""Synthetic benchmark dataset generator for smoke-testing and pipeline validation.

Note:
    Images produced here are procedural approximations of camera-captured vs.
    generative-model frequency/noise characteristics. They are intended for local
    testing, CI, and pipeline verification — not as a substitute for real-world
    datasets (e.g., GenImage, CIFAKE, DiffusionDB + camera photos) in production.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _fbm_field(rng: np.random.Generator, height: int, width: int, octaves: int = 4) -> np.ndarray:
    """Generate smooth multi-octave 2D value noise in ``[0, 1]``."""
    field = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    total_amp = 0.0
    for oct_idx in range(octaves):
        grid_h = max(4, height // (2 ** (octaves - oct_idx + 1)))
        grid_w = max(4, width // (2 ** (octaves - oct_idx + 1)))
        coarse = rng.uniform(0.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)
        up = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        field += amp * up
        total_amp += amp
        amp *= 0.5
    field /= max(1e-6, total_amp)
    return np.clip(field, 0.0, 1.0)


def _draw_random_scene(rng: np.random.Generator, size: int) -> np.ndarray:
    """Create a colorful structured base scene with gradients and geometric shapes."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    angle = float(rng.uniform(0, 2.0 * np.pi))
    grad = 0.5 + 0.35 * (np.cos(angle) * (xx - 0.5) + np.sin(angle) * (yy - 0.5))

    base_colors = rng.uniform(0.15, 0.88, size=(2, 3)).astype(np.float32)
    img = (
        grad[:, :, None] * base_colors[0][None, None, :]
        + (1.0 - grad[:, :, None]) * base_colors[1][None, None, :]
    )

    fbm_r = _fbm_field(rng, size, size, octaves=4)
    fbm_g = _fbm_field(rng, size, size, octaves=4)
    fbm_b = _fbm_field(rng, size, size, octaves=4)
    texture = np.stack([fbm_r, fbm_g, fbm_b], axis=-1)
    img = 0.65 * img + 0.35 * texture

    canvas = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    n_shapes = int(rng.integers(3, 8))
    for _ in range(n_shapes):
        color = tuple(int(c) for c in rng.integers(20, 240, size=3))
        if rng.random() < 0.5:
            center = (int(rng.integers(16, size - 16)), int(rng.integers(16, size - 16)))
            radius = int(rng.integers(10, max(14, size // 5)))
            cv2.circle(canvas, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        else:
            pt1 = (int(rng.integers(8, size - 24)), int(rng.integers(8, size - 24)))
            pt2 = (
                min(size - 4, pt1[0] + int(rng.integers(16, size // 3))),
                min(size - 4, pt1[1] + int(rng.integers(16, size // 3))),
            )
            cv2.rectangle(canvas, pt1, pt2, color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas.astype(np.float32) / 255.0


def _make_real_like_image(rng: np.random.Generator, size: int) -> np.ndarray:
    """Simulate natural optical blur, vignette, Bayer-correlated sensor noise, and shot noise."""
    scene = _draw_random_scene(rng, size)
    # Optical lens softness.
    scene = cv2.GaussianBlur(scene, (3, 3), sigmaX=float(rng.uniform(0.45, 0.85)))

    # Natural radial lens vignetting.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    dist2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    vignette = 1.0 - float(rng.uniform(0.12, 0.30)) * dist2
    scene = scene * vignette[:, :, None]

    # Correlated luminance + chromatic sensor noise (simulating demosaicing correlation).
    shared_noise = rng.normal(0.0, float(rng.uniform(0.012, 0.028)), size=(size, size, 1)).astype(
        np.float32
    )
    indep_noise = rng.normal(0.0, float(rng.uniform(0.004, 0.010)), size=(size, size, 3)).astype(
        np.float32
    )
    shot_scale = np.sqrt(np.clip(scene, 0.02, 1.0))
    sensor_noise = (0.75 * shared_noise + 0.25 * indep_noise) * shot_scale
    noisy = np.clip(scene + sensor_noise, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def _make_ai_like_image(rng: np.random.Generator, size: int) -> np.ndarray:
    """Simulate generative artifacts: over-smoothing, spectral roll-off, and upsampling grid."""
    scene = _draw_random_scene(rng, size)

    # Bilateral / diffusion-style local smoothing preserving sharp edges but suppressing micro-texture.
    scene_u8 = np.clip(scene * 255.0, 0, 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(scene_u8, d=7, sigmaColor=40, sigmaSpace=40).astype(np.float32) / 255.0

    # Subtle periodic upsampling / checkerboard harmonic artifact (4px or 8px stride).
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    stride = float(rng.choice([4.0, 8.0]))
    amp = float(rng.uniform(0.008, 0.022))
    phase_x = float(rng.uniform(0, np.pi))
    phase_y = float(rng.uniform(0, np.pi))
    grid = amp * (
        np.cos(2.0 * np.pi * xx / stride + phase_x) * np.cos(2.0 * np.pi * yy / stride + phase_y)
    )

    # Channel-independent high-frequency synthesis residual.
    synth_res = rng.normal(0.0, float(rng.uniform(0.003, 0.009)), size=(size, size, 3)).astype(
        np.float32
    )
    synth_res = cv2.GaussianBlur(synth_res, (3, 3), sigmaX=0.5)

    out = np.clip(smooth + grid[:, :, None] + synth_res, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def generate_sample_dataset(
    output_dir: str | Path,
    num_per_class: int = 60,
    image_size: int = 224,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate a balanced synthetic ``real/`` and ``ai/`` dataset on disk."""
    root = Path(output_dir)
    real_dir = root / "real"
    ai_dir = root / "ai"
    real_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for idx in range(num_per_class):
        real_img = _make_real_like_image(rng, image_size)
        ai_img = _make_ai_like_image(rng, image_size)

        # Mix JPEG and PNG across BOTH classes so file format never leaks the label.
        use_jpeg = (idx % 2 == 0)
        if use_jpeg:
            q_real = int(rng.integers(80, 96))
            q_ai = int(rng.integers(80, 96))
            Image.fromarray(real_img, mode="RGB").save(
                real_dir / f"real_{idx:04d}.jpg", format="JPEG", quality=q_real
            )
            Image.fromarray(ai_img, mode="RGB").save(
                ai_dir / f"ai_{idx:04d}.jpg", format="JPEG", quality=q_ai
            )
        else:
            Image.fromarray(real_img, mode="RGB").save(
                real_dir / f"real_{idx:04d}.png", format="PNG"
            )
            Image.fromarray(ai_img, mode="RGB").save(
                ai_dir / f"ai_{idx:04d}.png", format="PNG"
            )

    return {"root": root, "real_dir": real_dir, "ai_dir": ai_dir}
