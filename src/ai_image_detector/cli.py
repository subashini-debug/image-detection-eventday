"""Command-line interface for training, evaluating, inspecting, and predicting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader

from ai_image_detector.benchmark import generate_sample_dataset
from ai_image_detector.config import DataConfig, ModelConfig, TrainConfig
from ai_image_detector.dataset import AIImageDataset, discover_samples
from ai_image_detector.evaluator import evaluate_model
from ai_image_detector.forensics import load_rgb_image, summarize_forensics
from ai_image_detector.predictor import AIDetectorPredictor
from ai_image_detector.trainer import load_checkpoint, resolve_device, train_detector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-detect",
        description="Hybrid deep learning + forensic detector for real vs. AI-generated images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. generate-sample
    gen_p = sub.add_parser(
        "generate-sample",
        help="Generate a synthetic sample benchmark dataset (real/ and ai/) for smoke testing.",
    )
    gen_p.add_argument("--output", required=True, help="Output directory for real/ and ai/ folders.")
    gen_p.add_argument("--num-per-class", type=int, default=60, help="Images per class.")
    gen_p.add_argument("--image-size", type=int, default=224, help="Image width/height in pixels.")
    gen_p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # 2. train
    train_p = sub.add_parser("train", help="Train the hybrid detector on a labeled dataset.")
    train_p.add_argument(
        "--data",
        required=True,
        help="Dataset directory (real/ + ai/, or train/val/test/) or CSV manifest (path,label).",
    )
    train_p.add_argument("--output", required=True, help="Directory for checkpoints and metrics.")
    train_p.add_argument("--epochs", type=int, default=15, help="Maximum training epochs.")
    train_p.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    train_p.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    train_p.add_argument("--image-size", type=int, default=224, help="Spatial input size.")
    train_p.add_argument("--base-channels", type=int, default=32, help="Base CNN channel width.")
    train_p.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    train_p.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_p.add_argument("--quiet", action="store_true", help="Suppress per-epoch logs.")

    # 3. evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate a trained checkpoint on a labeled dataset.")
    eval_p.add_argument("--checkpoint", required=True, help="Path to saved .pt checkpoint.")
    eval_p.add_argument("--data", required=True, help="Evaluation dataset directory or CSV.")
    eval_p.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    eval_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (defaults to checkpoint's tuned threshold).",
    )
    eval_p.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    eval_p.add_argument("--json", action="store_true", help="Emit JSON output only.")

    # 4. predict
    pred_p = sub.add_parser("predict", help="Predict whether image(s) are real or AI-generated.")
    pred_p.add_argument("--checkpoint", required=True, help="Path to saved .pt checkpoint.")
    pred_p.add_argument("--input", required=True, help="Path to an image file or directory.")
    pred_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override classification threshold.",
    )
    pred_p.add_argument("--no-tta", action="store_true", help="Disable multi-view test-time augmentation.")
    pred_p.add_argument("--explain", action="store_true", help="Include forensic & EXIF breakdown.")
    pred_p.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    pred_p.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 5. inspect
    insp_p = sub.add_parser(
        "inspect",
        help="Extract frequency, noise-residual, ELA, and metadata forensics without a checkpoint.",
    )
    insp_p.add_argument("--input", required=True, help="Path to an image file.")
    insp_p.add_argument("--json", action="store_true", help="Emit full JSON forensic vector.")

    # 6. serve
    srv_p = sub.add_parser(
        "serve",
        help="Host the web UI and REST API locally using Uvicorn.",
    )
    srv_p.add_argument("--checkpoint", default=None, help="Optional path to .pt checkpoint.")
    srv_p.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    srv_p.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    srv_p.add_argument("--threshold", type=float, default=None, help="Override decision threshold.")
    srv_p.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-sample":
        paths = generate_sample_dataset(
            output_dir=args.output,
            num_per_class=args.num_per_class,
            image_size=args.image_size,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "root": str(paths["root"]),
                    "real_dir": str(paths["real_dir"]),
                    "ai_dir": str(paths["ai_dir"]),
                    "num_per_class": args.num_per_class,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train":
        m_cfg = ModelConfig(image_size=args.image_size, base_channels=args.base_channels)
        d_cfg = DataConfig(image_size=args.image_size, batch_size=args.batch_size, seed=args.seed)
        t_cfg = TrainConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            early_stopping_patience=args.patience,
            device=args.device,
            seed=args.seed,
        )
        summary = train_detector(
            dataset_path=args.data,
            output_dir=args.output,
            model_config=m_cfg,
            data_config=d_cfg,
            train_config=t_cfg,
            verbose=not args.quiet,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(summary.checkpoint_path),
                    "metrics_file": str(summary.metrics_path),
                    "best_epoch": summary.best_epoch,
                    "temperature": summary.temperature,
                    "optimal_threshold": summary.optimal_threshold,
                    "val_metrics": summary.val_metrics,
                    "test_metrics": summary.test_metrics,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate":
        device = resolve_device(args.device)
        model, payload = load_checkpoint(args.checkpoint, device=device)
        m_cfg = ModelConfig.from_dict(payload["model_config"])
        feat_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        feat_std = np.asarray(payload["feature_std"], dtype=np.float32)
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(payload.get("optimal_threshold", 0.5))
        )

        samples = discover_samples(args.data)
        ds = AIImageDataset(
            samples,
            image_size=m_cfg.image_size,
            fft_size=m_cfg.fft_size,
            train=False,
            feature_mean=feat_mean,
            feature_std=feat_std,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        metrics = evaluate_model(model, loader, device, threshold=threshold)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print("Evaluation Metrics:")
            print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "predict":
        predictor = AIDetectorPredictor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
        )
        results = predictor.predict_path(
            args.input,
            tta=not args.no_tta,
            explain=args.explain,
        )
        if args.json:
            payload = [r.to_dict() for r in results]
            print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))
        else:
            for r in results:
                print(
                    f"{Path(r.image_path).name}: {r.label.upper()} "
                    f"(ai_prob={r.ai_probability:.4f}, real_prob={r.real_probability:.4f}, "
                    f"confidence={r.confidence:.4f}, threshold={r.threshold:.4f})"
                )
                if r.forensics:
                    print("  Forensic highlights:", json.dumps(r.forensics["highlights"]))
                    if "metadata" in r.forensics:
                        print("  Metadata:", json.dumps(r.forensics["metadata"]))
        return 0

    if args.command == "inspect":
        rgb = load_rgb_image(args.input)
        report = summarize_forensics(rgb, image_path=args.input)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Forensic summary for {args.input}:")
            print("Highlights:", json.dumps(report["highlights"], indent=2))
            print("Metadata:", json.dumps(report.get("metadata", {}), indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        from ai_image_detector.server import create_app

        app = create_app(
            checkpoint_path=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
