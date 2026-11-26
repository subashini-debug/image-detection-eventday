"""Evaluation metrics, threshold optimization, and calibration assessment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ai_image_detector.model import HybridAIDetector


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0

    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        acc_bin = float(np.mean(y_true[mask]))
        conf_bin = float(np.mean(y_prob[mask]))
        ece += (count / total) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive binary classification and calibration metrics."""
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true_int, y_pred))
    precision = float(precision_score(y_true_int, y_pred, zero_division=0))
    recall = float(recall_score(y_true_int, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_int, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true_int, np.clip(y_prob, 0.0, 1.0)))
    ece = compute_ece(y_true_int, y_prob)

    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    has_both_classes = len(np.unique(y_true_int)) > 1
    roc_auc = float(roc_auc_score(y_true_int, y_prob)) if has_both_classes else None
    pr_auc = float(average_precision_score(y_true_int, y_prob)) if has_both_classes else None

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {"real": int(np.sum(y_true_int == 0)), "ai": int(np.sum(y_true_int == 1))},
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability threshold in ``[0.2, 0.8]`` maximizing F1 (tie-broken near 0.5)."""
    y_true_int = y_true.astype(int)
    candidates = np.linspace(0.20, 0.80, 61)
    best_thr = 0.5
    best_score = -1.0

    for thr in candidates:
        preds = (y_prob >= thr).astype(int)
        score = float(f1_score(y_true_int, preds, zero_division=0))
        if score > best_score + 1e-6:
            best_score = score
            best_thr = float(thr)
        elif abs(score - best_score) <= 1e-6:
            if abs(thr - 0.5) < abs(best_thr - 0.5):
                best_thr = float(thr)
    return round(best_thr, 4)


@torch.no_grad()
def collect_predictions(
    model: HybridAIDetector,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and return ``(y_true, y_prob, raw_logits)``."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    for spatial, fft_map, forensic_vec, labels in loader:
        spatial = spatial.to(device)
        fft_map = fft_map.to(device)
        forensic_vec = forensic_vec.to(device)

        logits = model(spatial, fft_map, forensic_vec)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    if not all_logits:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    y_true = np.concatenate(all_targets, axis=0)
    raw_logits = np.concatenate(all_logits, axis=0)
    temp = float(torch.clamp(model.temperature.detach().cpu(), min=0.1, max=10.0).item())
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(raw_logits / temp, -40.0, 40.0)))
    return y_true, y_prob, raw_logits


def evaluate_model(
    model: HybridAIDetector,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a model on a DataLoader and return metrics dictionary."""
    y_true, y_prob, _ = collect_predictions(model, loader, device)
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate on an empty DataLoader.")
    return compute_binary_metrics(y_true, y_prob, threshold=threshold)
