"""Image forensic, frequency-domain, and noise-residual feature extraction."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image

FORENSIC_FEATURE_NAMES: tuple[str, ...] = (
    "srm1_mean_abs",
    "srm1_std",
    "srm1_tail_ratio",
    "srm2_mean_abs",
    "srm2_std",
    "srm2_tail_ratio",
    "srm3_mean_abs",
    "srm3_std",
    "srm3_tail_ratio",
    "fft_band_0",
    "fft_band_1",
    "fft_band_2",
    "fft_band_3",
    "fft_band_4",
    "fft_band_5",
    "fft_band_6",
    "fft_band_7",
    "fft_high_freq_ratio",
    "fft_spectral_slope",
    "fft_grid_peak_ratio",
    "dct_low_energy_ratio",
    "dct_mid_energy_ratio",
    "dct_high_energy_ratio",
    "dct_ac_kurtosis",
    "ela_mean",
    "ela_std",
    "ela_p95",
    "rgb_rg_residual_corr",
    "rgb_gb_residual_corr",
    "chroma_luma_noise_ratio",
    "laplacian_log_var",
    "hsv_saturation_std",
)

FORENSIC_DIM: int = len(FORENSIC_FEATURE_NAMES)

# Classic Steganalysis Rich Model (SRM) high-pass noise residual kernels.
_SRM_KERNEL_1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
) / 4.0

_SRM_KERNEL_2 = np.array(
    [
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ],
    dtype=np.float32,
) / 12.0

_SRM_KERNEL_3 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
) / 2.0

SRM_KERNELS: tuple[np.ndarray, np.ndarray, np.ndarray] = (
    _SRM_KERNEL_1,
    _SRM_KERNEL_2,
    _SRM_KERNEL_3,
)

_AI_SOFTWARE_KEYWORDS = (
    "stable diffusion",
    "automatic1111",
    "comfyui",
    "midjourney",
    "dall-e",
    "dalle",
    "novelai",
    "invokeai",
    "firefly",
    "imagen",
    "flux",
    "ideogram",
    "leonardo",
    "diffusers",
)

_EPS = 1e-8


def load_rgb_image(source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Load an image source into a uint8 RGB numpy array of shape ``(H, W, 3)``."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"), dtype=np.uint8)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def compute_srm_residuals(rgb: np.ndarray) -> np.ndarray:
    """Compute 3 bounded SRM high-pass residual maps of shape ``(3, H, W)`` in ``[-1, 1]``."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    residuals = []
    for kernel in SRM_KERNELS:
        filtered = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        # Squash large edges while preserving subtle sensor/diffusion noise structure.
        bounded = np.tanh(filtered * 15.0)
        residuals.append(bounded)
    return np.stack(residuals, axis=0).astype(np.float32)


def compute_fft_spectrum(rgb: np.ndarray, fft_size: int = 128) -> np.ndarray:
    """Compute centered 2D log-magnitude FFT spectrum of shape ``(1, fft_size, fft_size)``."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if gray.shape[0] != fft_size or gray.shape[1] != fft_size:
        gray = cv2.resize(gray, (fft_size, fft_size), interpolation=cv2.INTER_AREA)

    # Apply Hann window to suppress boundary leakage.
    win_y = np.hanning(fft_size).astype(np.float32)
    win_x = np.hanning(fft_size).astype(np.float32)
    window = np.outer(win_y, win_x)
    centered_gray = (gray - gray.mean()) * window

    f = np.fft.fft2(centered_gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    max_val = float(mag.max())
    if max_val > _EPS:
        mag = mag / max_val
    return mag[np.newaxis, :, :].astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel() - float(a.mean())
    b_flat = b.ravel() - float(b.mean())
    denom = float(np.sqrt(np.sum(a_flat * a_flat) * np.sum(b_flat * b_flat)))
    if denom < _EPS:
        return 0.0
    return float(np.clip(np.sum(a_flat * b_flat) / denom, -1.0, 1.0))


def _compute_fft_stats(gray_float: np.ndarray) -> list[float]:
    size = 128
    resized = cv2.resize(gray_float, (size, size), interpolation=cv2.INTER_AREA)
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2((resized - resized.mean()) * win))
    power = np.log1p(np.abs(f) ** 2)

    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = float(np.sqrt(cy**2 + cx**2))
    norm_r = radius / (max_r + _EPS)

    # 8 radial bands (normalized to total band energy).
    band_edges = np.linspace(0.0, 1.0, 9)
    band_means = np.zeros(8, dtype=np.float32)
    for idx in range(8):
        mask = (norm_r >= band_edges[idx]) & (norm_r < band_edges[idx + 1])
        if idx == 7:
            mask = (norm_r >= band_edges[idx]) & (norm_r <= band_edges[idx + 1])
        if np.any(mask):
            band_means[idx] = float(power[mask].mean())

    band_sum = float(band_means.sum())
    if band_sum > _EPS:
        band_norm = (band_means / band_sum).tolist()
    else:
        band_norm = [0.125] * 8

    # High frequency ratio: outer half radius vs full spectrum.
    total_power = float(power.sum())
    high_mask = norm_r >= 0.45
    high_ratio = float(power[high_mask].sum() / (total_power + _EPS))

    # Spectral roll-off slope across radial bins (excluding DC bin 0).
    x_bins = np.log1p(np.arange(1, 8, dtype=np.float32))
    y_bins = np.log1p(band_means[1:])
    if float(np.std(y_bins)) > _EPS:
        slope = float(np.polyfit(x_bins, y_bins, 1)[0])
    else:
        slope = 0.0

    # Grid/checkerboard peak ratio along horizontal & vertical frequency axes in outer ring.
    outer_ring = (norm_r >= 0.35) & (norm_r <= 0.95)
    axis_mask = outer_ring & ((np.abs(yy - cy) <= 2) | (np.abs(xx - cx) <= 2))
    bg_mask = outer_ring & (~axis_mask)
    axis_power = float(power[axis_mask].mean()) if np.any(axis_mask) else 0.0
    bg_power = float(np.median(power[bg_mask])) if np.any(bg_mask) else 0.0
    grid_peak_ratio = float(axis_power / (bg_power + 1e-4))

    return [*band_norm, high_ratio, slope, grid_peak_ratio]


def _compute_dct_stats(gray_float: np.ndarray) -> list[float]:
    size = 128
    resized = cv2.resize(gray_float, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    abs_dct = np.abs(dct)

    yy, xx = np.ogrid[:size, :size]
    freq_dist = np.sqrt(yy**2 + xx**2) / float(np.sqrt(2 * (size**2)))

    # Exclude DC component (0, 0).
    ac_mask = (yy + xx) > 0
    total_ac = float(abs_dct[ac_mask].sum()) + _EPS

    low_mask = ac_mask & (freq_dist < 0.25)
    mid_mask = ac_mask & (freq_dist >= 0.25) & (freq_dist < 0.60)
    high_mask = ac_mask & (freq_dist >= 0.60)

    low_ratio = float(abs_dct[low_mask].sum() / total_ac)
    mid_ratio = float(abs_dct[mid_mask].sum() / total_ac)
    high_ratio = float(abs_dct[high_mask].sum() / total_ac)

    ac_vals = dct[ac_mask]
    std = float(np.std(ac_vals))
    if std > _EPS:
        centered = (ac_vals - float(np.mean(ac_vals))) / std
        kurtosis = float(np.log1p(np.mean(centered**4)))
    else:
        kurtosis = 0.0

    return [low_ratio, mid_ratio, high_ratio, kurtosis]


def _compute_ela_stats(rgb: np.ndarray, quality: int = 90) -> list[float]:
    pil_img = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recomp_rgb = np.asarray(recompressed.convert("RGB"), dtype=np.float32)

    diff = np.abs(rgb.astype(np.float32) - recomp_rgb) / 255.0
    ela_mean = float(np.mean(diff))
    ela_std = float(np.std(diff))
    ela_p95 = float(np.percentile(diff, 95))
    return [ela_mean, ela_std, ela_p95]


def extract_forensic_features(rgb: np.ndarray) -> np.ndarray:
    """Extract the 32-dim handcrafted forensic feature vector from an RGB uint8 image."""
    gray_float = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 1. SRM residual statistics (9 features).
    srm_features: list[float] = []
    for kernel in SRM_KERNELS:
        res = cv2.filter2D(gray_float, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        abs_res = np.abs(res)
        mean_abs = float(np.mean(abs_res))
        std_res = float(np.std(res))
        p95 = float(np.percentile(abs_res, 95))
        tail_ratio = float(p95 / (mean_abs + 1e-4))
        srm_features.extend([mean_abs, std_res, tail_ratio])

    # 2. FFT spectral statistics (11 features).
    fft_features = _compute_fft_stats(gray_float)

    # 3. DCT frequency statistics (4 features).
    dct_features = _compute_dct_stats(gray_float)

    # 4. Error Level Analysis (3 features).
    ela_features = _compute_ela_stats(rgb, quality=90)

    # 5. Cross-channel noise residual correlation & color/sharpness (5 features).
    rgb_float = rgb.astype(np.float32) / 255.0
    r_res = cv2.filter2D(rgb_float[:, :, 0], -1, _SRM_KERNEL_1, borderType=cv2.BORDER_REFLECT)
    g_res = cv2.filter2D(rgb_float[:, :, 1], -1, _SRM_KERNEL_1, borderType=cv2.BORDER_REFLECT)
    b_res = cv2.filter2D(rgb_float[:, :, 2], -1, _SRM_KERNEL_1, borderType=cv2.BORDER_REFLECT)

    rg_corr = _safe_corr(r_res, g_res)
    gb_corr = _safe_corr(g_res, b_res)

    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    y_res = cv2.filter2D(ycbcr[:, :, 0], -1, _SRM_KERNEL_1, borderType=cv2.BORDER_REFLECT)
    cr_res = cv2.filter2D(ycbcr[:, :, 1], -1, _SRM_KERNEL_1, borderType=cv2.BORDER_REFLECT)
    cb_res = cv2.filter2D(ycbcr[:, :, 2], -1, _SRM_KERNEL_1, borderType=cv2.BORDER_REFLECT)
    chroma_noise = 0.5 * (float(np.std(cr_res)) + float(np.std(cb_res)))
    chroma_luma_ratio = float(chroma_noise / (float(np.std(y_res)) + 1e-4))

    lap = cv2.Laplacian(gray_float, cv2.CV_32F)
    lap_log_var = float(np.log1p(float(np.var(lap)) * 1000.0))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    sat_std = float(np.std(hsv[:, :, 1]))

    color_features = [rg_corr, gb_corr, chroma_luma_ratio, lap_log_var, sat_std]

    vec = np.array(
        [*srm_features, *fft_features, *dct_features, *ela_features, *color_features],
        dtype=np.float32,
    )
    vec = np.nan_to_num(vec, nan=0.0, posinf=10.0, neginf=-10.0)
    return vec


def inspect_metadata(image_path: str | Path) -> dict[str, Any]:
    """Inspect EXIF camera metadata and PNG/text tags for auxiliary provenance cues."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    info: dict[str, Any] = {
        "format": None,
        "width": 0,
        "height": 0,
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "ai_metadata_matches": [],
    }

    with Image.open(path) as img:
        info["format"] = img.format
        info["width"], info["height"] = img.size

        text_blobs: list[str] = []
        for k, v in (img.info or {}).items():
            if isinstance(v, (str, bytes)):
                val = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v
                text_blobs.append(f"{k}: {val[:500]}")

        exif_raw = img.getexif()
        if exif_raw:
            info["has_exif"] = len(exif_raw) > 0
            for tag_id, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value)
                if tag_name == "Make":
                    info["camera_make"] = val_str.strip()
                elif tag_name == "Model":
                    info["camera_model"] = val_str.strip()
                elif tag_name == "Software":
                    info["software"] = val_str.strip()
                text_blobs.append(f"{tag_name}: {val_str[:300]}")

        joined = "\n".join(text_blobs).lower()
        matches = [kw for kw in _AI_SOFTWARE_KEYWORDS if kw in joined]
        info["ai_metadata_matches"] = sorted(set(matches))

    return info


def summarize_forensics(
    rgb: np.ndarray,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a structured dictionary of named forensic signals and optional metadata."""
    vec = extract_forensic_features(rgb)
    named = {name: round(float(val), 6) for name, val in zip(FORENSIC_FEATURE_NAMES, vec, strict=True)}
    summary: dict[str, Any] = {
        "highlights": {
            "fft_high_freq_ratio": named["fft_high_freq_ratio"],
            "fft_spectral_slope": named["fft_spectral_slope"],
            "fft_grid_peak_ratio": named["fft_grid_peak_ratio"],
            "ela_mean": named["ela_mean"],
            "ela_p95": named["ela_p95"],
            "rgb_rg_residual_corr": named["rgb_rg_residual_corr"],
            "rgb_gb_residual_corr": named["rgb_gb_residual_corr"],
            "chroma_luma_noise_ratio": named["chroma_luma_noise_ratio"],
            "laplacian_log_var": named["laplacian_log_var"],
        },
        "features": named,
    }
    if image_path is not None:
        summary["metadata"] = inspect_metadata(image_path)
    return summary
