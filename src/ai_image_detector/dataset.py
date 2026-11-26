"""Dataset discovery, splitting, augmentation, and multi-branch tensor preparation."""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ai_image_detector.config import DataConfig
from ai_image_detector.forensics import (
    FORENSIC_DIM,
    compute_fft_spectrum,
    compute_srm_residuals,
    extract_forensic_features,
    load_rgb_image,
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)

REAL_DIR_NAMES: frozenset[str] = frozenset({"real", "human", "authentic", "natural", "0"})
AI_DIR_NAMES: frozenset[str] = frozenset(
    {"ai", "ai_generated", "fake", "generated", "synthetic", "1"}
)


@dataclass(frozen=True)
class ImageSample:
    """A single labeled image entry."""

    path: Path
    label: int  # 0 = real, 1 = ai_generated


def _parse_label(raw: str | int) -> int:
    text = str(raw).strip().lower()
    if text in REAL_DIR_NAMES:
        return 0
    if text in AI_DIR_NAMES:
        return 1
    raise ValueError(f"Unrecognized label value: {raw!r}. Expected 'real'/'0' or 'ai'/'1'.")


def _collect_class_dir(class_dir: Path, label: int) -> list[ImageSample]:
    if not class_dir.is_dir():
        return []
    samples: list[ImageSample] = []
    for file_path in sorted(class_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append(ImageSample(path=file_path, label=label))
    return samples


def discover_split_dirs(root_dir: str | Path) -> dict[str, list[ImageSample]] | None:
    """Detect pre-split ``train/val/test`` layout if present."""
    root = Path(root_dir)
    train_dir = root / "train"
    if not train_dir.is_dir():
        return None

    splits: dict[str, list[ImageSample]] = {}
    for split_name, aliases in (
        ("train", ("train",)),
        ("val", ("val", "valid", "validation")),
        ("test", ("test", "eval")),
    ):
        split_samples: list[ImageSample] = []
        for alias in aliases:
            candidate = root / alias
            if candidate.is_dir():
                for child in sorted(candidate.iterdir()):
                    if not child.is_dir():
                        continue
                    cname = child.name.lower()
                    if cname in REAL_DIR_NAMES:
                        split_samples.extend(_collect_class_dir(child, 0))
                    elif cname in AI_DIR_NAMES:
                        split_samples.extend(_collect_class_dir(child, 1))
                break
        splits[split_name] = split_samples

    if not splits["train"]:
        return None
    return splits


def discover_samples(source: str | Path) -> list[ImageSample]:
    """Discover labeled images from a directory (``real/`` + ``ai/``) or CSV manifest."""
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".csv":
        samples: list[ImageSample] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV manifest must contain 'path' and 'label' columns.")
            for row in reader:
                img_path = Path(row["path"])
                if not img_path.is_absolute():
                    img_path = (path.parent / img_path).resolve()
                samples.append(ImageSample(path=img_path, label=_parse_label(row["label"])))
        if not samples:
            raise ValueError(f"No samples found in CSV manifest: {path}")
        return samples

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    samples = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        cname = child.name.lower()
        if cname in REAL_DIR_NAMES:
            samples.extend(_collect_class_dir(child, 0))
        elif cname in AI_DIR_NAMES:
            samples.extend(_collect_class_dir(child, 1))

    if not samples:
        raise ValueError(
            f"No labeled images found under {path}. Expected 'real/' and 'ai/' (or 'fake/') subdirectories."
        )
    return samples


def stratified_split_samples(
    samples: Sequence[ImageSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ImageSample]]:
    """Split samples into stratified train/val/test subsets."""
    if len(samples) < 6:
        raise ValueError(f"Need at least 6 images to split train/val/test, found {len(samples)}.")

    labels = [s.label for s in samples]
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError("Dataset must contain both 'real' (0) and 'ai' (1) classes.")

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0.0 or holdout_ratio >= 0.8:
        raise ValueError("val_ratio + test_ratio must be between 0 and 0.8.")

    indices = list(range(len(samples)))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=labels,
    )

    holdout_labels = [labels[i] for i in holdout_idx]
    if test_ratio > 0 and len(holdout_idx) >= 2 and len(set(holdout_labels)) > 1:
        relative_test = test_ratio / holdout_ratio
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=relative_test,
            random_state=seed,
            stratify=holdout_labels,
        )
    else:
        val_idx = holdout_idx
        test_idx = []

    return {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }


def load_dataset_splits(
    dataset_path: str | Path,
    config: DataConfig,
) -> dict[str, list[ImageSample]]:
    """Load pre-split directory or create stratified splits automatically."""
    path = Path(dataset_path)
    if path.is_dir():
        presplit = discover_split_dirs(path)
        if presplit is not None:
            if not presplit["val"]:
                # Carve validation split from train if only train/test were provided.
                split_tv = stratified_split_samples(
                    presplit["train"],
                    val_ratio=config.val_ratio,
                    test_ratio=0.0,
                    seed=config.seed,
                )
                presplit["train"] = split_tv["train"]
                presplit["val"] = split_tv["val"]
            return presplit

    all_samples = discover_samples(path)
    return stratified_split_samples(
        all_samples,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize RGB image to ``(size, size, 3)`` using quality-appropriate interpolation."""
    h, w = rgb.shape[:2]
    if h == size and w == size:
        return rgb
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def _augment_rgb(
    rgb: np.ndarray,
    rng: random.Random,
    jpeg_aug_prob: float,
    jpeg_quality_min: int,
    jpeg_quality_max: int,
) -> np.ndarray:
    out = rgb
    if rng.random() < 0.5:
        out = np.ascontiguousarray(out[:, ::-1, :])

    h, w = out.shape[:2]
    if min(h, w) >= 64 and rng.random() < 0.6:
        scale = rng.uniform(0.82, 1.0)
        crop_h = max(32, int(round(h * scale)))
        crop_w = max(32, int(round(w * scale)))
        top = rng.randint(0, max(0, h - crop_h))
        left = rng.randint(0, max(0, w - crop_w))
        out = out[top : top + crop_h, left : left + crop_w, :]

    if rng.random() < jpeg_aug_prob:
        quality = rng.randint(jpeg_quality_min, jpeg_quality_max)
        buf = io.BytesIO()
        Image.fromarray(out, mode="RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as recomp:
            out = np.asarray(recomp.convert("RGB"), dtype=np.uint8)

    return out


def prepare_model_inputs(
    rgb: np.ndarray,
    image_size: int = 224,
    fft_size: int = 128,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an RGB uint8 image into ``(spatial_6ch, fft_1ch, forensic_vec)`` tensors."""
    resized = resize_rgb(rgb, image_size)
    rgb_norm = (resized.astype(np.float32) / 127.5) - 1.0
    rgb_chw = np.transpose(rgb_norm, (2, 0, 1))
    srm_chw = compute_srm_residuals(resized)
    spatial = np.concatenate([rgb_chw, srm_chw], axis=0).astype(np.float32)

    fft_map = compute_fft_spectrum(resized, fft_size=fft_size)
    forensic_vec = extract_forensic_features(resized)

    if feature_mean is not None and feature_std is not None:
        safe_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        forensic_vec = (forensic_vec - feature_mean) / safe_std
        forensic_vec = np.clip(forensic_vec, -10.0, 10.0).astype(np.float32)

    return (
        torch.from_numpy(spatial),
        torch.from_numpy(fft_map),
        torch.from_numpy(forensic_vec),
    )


def compute_forensic_normalization_stats(
    samples: Sequence[ImageSample],
    image_size: int = 224,
    max_samples: int = 512,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std on training images for ForensicMLP standardization."""
    if not samples:
        return np.zeros(FORENSIC_DIM, dtype=np.float32), np.ones(FORENSIC_DIM, dtype=np.float32)

    selected = list(samples)
    if len(selected) > max_samples:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_samples)

    rows = []
    for sample in selected:
        rgb = resize_rgb(load_rgb_image(sample.path), image_size)
        rows.append(extract_forensic_features(rgb))

    mat = np.stack(rows, axis=0).astype(np.float32)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


class AIImageDataset(Dataset):
    """PyTorch Dataset yielding ``(spatial_6ch, fft_1ch, forensic_vec, label)``."""

    def __init__(
        self,
        samples: Sequence[ImageSample],
        image_size: int = 224,
        fft_size: int = 128,
        train: bool = False,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        jpeg_aug_prob: float = 0.35,
        jpeg_quality_min: int = 65,
        jpeg_quality_max: int = 98,
        seed: int = 42,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.fft_size = fft_size
        self.train = train
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.jpeg_aug_prob = jpeg_aug_prob
        self.jpeg_quality_min = jpeg_quality_min
        self.jpeg_quality_max = jpeg_quality_max
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        rgb = load_rgb_image(sample.path)
        if self.train:
            rgb = _augment_rgb(
                rgb,
                rng=self._rng,
                jpeg_aug_prob=self.jpeg_aug_prob,
                jpeg_quality_min=self.jpeg_quality_min,
                jpeg_quality_max=self.jpeg_quality_max,
            )

        spatial, fft_map, forensic_vec = prepare_model_inputs(
            rgb=rgb,
            image_size=self.image_size,
            fft_size=self.fft_size,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return spatial, fft_map, forensic_vec, label
