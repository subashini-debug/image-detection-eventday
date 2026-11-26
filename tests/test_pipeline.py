"""End-to-end integration tests for dataset generation, training, evaluation, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from ai_image_detector.benchmark import generate_sample_dataset
from ai_image_detector.cli import main as cli_main
from ai_image_detector.config import DataConfig, ModelConfig, TrainConfig
from ai_image_detector.predictor import AIDetectorPredictor
from ai_image_detector.trainer import train_detector


def test_end_to_end_training_and_prediction(tmp_path: Path) -> None:
    data_dir = tmp_path / "sample_dataset"
    out_dir = tmp_path / "run_out"

    generate_sample_dataset(data_dir, num_per_class=40, image_size=128, seed=7)

    m_cfg = ModelConfig(image_size=128, fft_size=64, base_channels=16, freq_channels=12)
    d_cfg = DataConfig(image_size=128, fft_size=64, batch_size=8, jpeg_aug_prob=0.25, seed=7)
    t_cfg = TrainConfig(epochs=8, learning_rate=1e-3, early_stopping_patience=5, device="cpu", seed=7)

    summary = train_detector(
        dataset_path=data_dir,
        output_dir=out_dir,
        model_config=m_cfg,
        data_config=d_cfg,
        train_config=t_cfg,
        verbose=False,
    )

    assert summary.checkpoint_path.is_file()
    assert summary.metrics_path.is_file()
    assert summary.val_metrics["roc_auc"] is not None
    assert summary.test_metrics is not None
    assert summary.val_metrics["roc_auc"] >= 0.75

    predictor = AIDetectorPredictor(summary.checkpoint_path, device="cpu")
    real_files = sorted((data_dir / "real").iterdir())
    ai_files = sorted((data_dir / "ai").iterdir())

    res_real = predictor.predict_image(real_files[0], tta=True, explain=True)
    res_ai = predictor.predict_image(ai_files[0], tta=True, explain=True)

    assert res_real.label in {"real", "ai_generated"}
    assert 0.0 <= res_real.ai_probability <= 1.0
    assert abs((res_real.ai_probability + res_real.real_probability) - 1.0) < 1e-3
    assert res_real.forensics is not None
    assert res_ai.forensics is not None


def test_cli_workflow(tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "cli_data"
    out_dir = tmp_path / "cli_run"

    assert cli_main(["generate-sample", "--output", str(data_dir), "--num-per-class", "16", "--image-size", "96"]) == 0
    capsys.readouterr()

    assert (
        cli_main(
            [
                "train",
                "--data",
                str(data_dir),
                "--output",
                str(out_dir),
                "--epochs",
                "3",
                "--batch-size",
                "8",
                "--image-size",
                "96",
                "--base-channels",
                "16",
                "--quiet",
            ]
        )
        == 0
    )
    train_stdout = capsys.readouterr().out
    train_payload = json.loads(train_stdout)
    ckpt = train_payload["checkpoint"]
    assert Path(ckpt).is_file()

    assert cli_main(["evaluate", "--checkpoint", ckpt, "--data", str(data_dir), "--json"]) == 0
    eval_payload = json.loads(capsys.readouterr().out)
    assert "accuracy" in eval_payload and "roc_auc" in eval_payload

    sample_ai = next((data_dir / "ai").iterdir())
    assert cli_main(["predict", "--checkpoint", ckpt, "--input", str(sample_ai), "--explain", "--json"]) == 0
    pred_payload = json.loads(capsys.readouterr().out)
    assert pred_payload["label"] in {"real", "ai_generated"}
    assert "forensics" in pred_payload

    assert cli_main(["inspect", "--input", str(sample_ai), "--json"]) == 0
    insp_payload = json.loads(capsys.readouterr().out)
    assert "highlights" in insp_payload
