"""Training pipeline, temperature calibration, and checkpoint persistence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from ai_image_detector.config import DataConfig, ModelConfig, TrainConfig
from ai_image_detector.dataset import (
    AIImageDataset,
    ImageSample,
    compute_forensic_normalization_stats,
    load_dataset_splits,
)
from ai_image_detector.evaluator import (
    collect_predictions,
    compute_binary_metrics,
    evaluate_model,
    find_optimal_threshold,
)
from ai_image_detector.model import HybridAIDetector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_spec: str = "auto") -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def _fit_temperature(raw_logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit post-hoc temperature scaling ``T`` minimizing validation NLL."""
    if len(raw_logits) < 4 or len(np.unique(y_true.astype(int))) < 2:
        return 1.0

    targets = y_true.astype(np.float64)
    logits = raw_logits.astype(np.float64)

    def nll_loss(temp: float) -> float:
        scaled = np.clip(logits / temp, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs)))

    res = minimize_scalar(nll_loss, bounds=(0.25, 5.0), method="bounded")
    if res.success and np.isfinite(res.x):
        return float(res.x)
    return 1.0


def save_checkpoint(
    path: str | Path,
    model: HybridAIDetector,
    model_config: ModelConfig,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    optimal_threshold: float,
    metrics: dict[str, Any],
) -> Path:
    """Save a self-contained checkpoint dictionary."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "optimal_threshold": float(optimal_threshold),
        "temperature": float(model.temperature.detach().cpu().item()),
        "metrics": metrics,
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[HybridAIDetector, dict[str, Any]]:
    """Load a saved :class:`HybridAIDetector` and metadata from a checkpoint."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    resolved_device = resolve_device(str(device)) if isinstance(device, str) else device
    payload = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)

    model_config = ModelConfig.from_dict(payload["model_config"])
    model = HybridAIDetector(model_config).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    if "temperature" in payload:
        model.temperature.fill_(float(payload["temperature"]))
    model.eval()
    return model, payload


@dataclass
class TrainingSummary:
    checkpoint_path: Path
    metrics_path: Path
    best_epoch: int
    optimal_threshold: float
    temperature: float
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None
    history: list[dict[str, Any]]


def train_detector(
    dataset_path: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig | None = None,
    data_config: DataConfig | None = None,
    train_config: TrainConfig | None = None,
    verbose: bool = True,
) -> TrainingSummary:
    """Train, calibrate, and evaluate :class:`HybridAIDetector` on a dataset."""
    m_cfg = model_config or ModelConfig()
    d_cfg = data_config or DataConfig(image_size=m_cfg.image_size, fft_size=m_cfg.fft_size)
    t_cfg = train_config or TrainConfig()

    seed_everything(t_cfg.seed)
    device = resolve_device(t_cfg.device)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_dataset_splits(dataset_path, d_cfg)
    train_samples: list[ImageSample] = splits["train"]
    val_samples: list[ImageSample] = splits["val"]
    test_samples: list[ImageSample] = splits["test"]

    feat_mean, feat_std = compute_forensic_normalization_stats(
        train_samples,
        image_size=m_cfg.image_size,
        seed=t_cfg.seed,
    )

    train_ds = AIImageDataset(
        train_samples,
        image_size=m_cfg.image_size,
        fft_size=m_cfg.fft_size,
        train=True,
        feature_mean=feat_mean,
        feature_std=feat_std,
        jpeg_aug_prob=d_cfg.jpeg_aug_prob,
        jpeg_quality_min=d_cfg.jpeg_quality_min,
        jpeg_quality_max=d_cfg.jpeg_quality_max,
        seed=t_cfg.seed,
    )
    val_ds = AIImageDataset(
        val_samples,
        image_size=m_cfg.image_size,
        fft_size=m_cfg.fft_size,
        train=False,
        feature_mean=feat_mean,
        feature_std=feat_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=d_cfg.batch_size,
        shuffle=True,
        num_workers=d_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=d_cfg.batch_size,
        shuffle=False,
        num_workers=d_cfg.num_workers,
    )

    model = HybridAIDetector(m_cfg).to(device)

    n_pos = sum(1 for s in train_samples if s.label == 1)
    n_neg = len(train_samples) - n_pos
    pos_weight_val = float(n_neg / max(1, n_pos))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.learning_rate,
        weight_decay=t_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_cfg.epochs),
        eta_min=t_cfg.learning_rate * 0.05,
    )

    best_score = -1.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, t_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for spatial, fft_map, forensic_vec, labels in train_loader:
            spatial = spatial.to(device)
            fft_map = fft_map.to(device)
            forensic_vec = forensic_vec.to(device)
            labels = labels.to(device)

            smooth_targets = labels * (1.0 - t_cfg.label_smoothing) + 0.5 * t_cfg.label_smoothing

            optimizer.zero_grad(set_to_none=True)
            logits = model(spatial, fft_map, forensic_vec)
            loss = criterion(logits, smooth_targets)
            loss.backward()
            if t_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip_norm)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)

        val_metrics = evaluate_model(model, val_loader, device, threshold=0.5)
        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_record)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{t_cfg.epochs:02d} | "
                f"loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['roc_auc']}"
            )

        if score > best_score + 1e-4 or best_state is None:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= t_cfg.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Post-hoc temperature calibration and threshold tuning on validation set.
    y_val, _, val_logits = collect_predictions(model, val_loader, device)
    temperature = _fit_temperature(val_logits, y_val) if t_cfg.calibrate_temperature else 1.0
    model.temperature.fill_(temperature)

    val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits / temperature, -40.0, 40.0)))
    optimal_threshold = find_optimal_threshold(y_val, val_probs) if t_cfg.tune_threshold else 0.5
    final_val_metrics = compute_binary_metrics(y_val, val_probs, threshold=optimal_threshold)

    test_metrics: dict[str, Any] | None = None
    if test_samples:
        test_ds = AIImageDataset(
            test_samples,
            image_size=m_cfg.image_size,
            fft_size=m_cfg.fft_size,
            train=False,
            feature_mean=feat_mean,
            feature_std=feat_std,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=d_cfg.batch_size,
            shuffle=False,
            num_workers=d_cfg.num_workers,
        )
        test_metrics = evaluate_model(model, test_loader, device, threshold=optimal_threshold)

    summary_metrics = {
        "best_epoch": best_epoch,
        "temperature": round(temperature, 4),
        "optimal_threshold": optimal_threshold,
        "dataset_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }

    ckpt_path = save_checkpoint(
        out_dir / "best_model.pt",
        model=model,
        model_config=m_cfg,
        feature_mean=feat_mean,
        feature_std=feat_std,
        optimal_threshold=optimal_threshold,
        metrics=summary_metrics,
    )
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")

    return TrainingSummary(
        checkpoint_path=ckpt_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        optimal_threshold=optimal_threshold,
        temperature=round(temperature, 4),
        val_metrics=final_val_metrics,
        test_metrics=test_metrics,
        history=history,
    )
