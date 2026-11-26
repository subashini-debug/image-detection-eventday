"""FastAPI web UI and REST API for real vs. AI-generated image detection."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from ai_image_detector.forensics import inspect_metadata, load_rgb_image, summarize_forensics
from ai_image_detector.predictor import AIDetectorPredictor

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Image Detector</title>
  <style>
    :root {
      --bg: #0b1020;
      --card: #131a2e;
      --border: #253052;
      --text: #e6ecff;
      --muted: #94a3c8;
      --accent: #6366f1;
      --real: #10b981;
      --ai: #ef4444;
      --warn: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at top, #151f3d 0%, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      padding: 32px 20px;
    }
    .container { max-width: 960px; margin: 0 auto; }
    header { margin-bottom: 24px; }
    h1 { margin: 0 0 8px; font-size: 1.9rem; letter-spacing: -0.02em; }
    .subtitle { color: var(--muted); margin: 0; font-size: 0.96rem; }
    .banner {
      margin-top: 14px;
      padding: 10px 14px;
      border-radius: 10px;
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.35);
      color: #c7d2fe;
      font-size: 0.88rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1.15fr;
      gap: 20px;
      margin-top: 20px;
    }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }
    .dropzone {
      border: 2px dashed #394875;
      border-radius: 12px;
      padding: 28px 16px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    .dropzone.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.08);
    }
    .dropzone p { margin: 6px 0; color: var(--muted); font-size: 0.92rem; }
    .btn {
      margin-top: 14px;
      width: 100%;
      padding: 12px 16px;
      border: none;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      font-size: 0.96rem;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .preview-wrap {
      margin-top: 16px;
      display: none;
      text-align: center;
    }
    .preview-wrap img {
      max-width: 100%;
      max-height: 280px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.9rem;
      letter-spacing: 0.04em;
    }
    .badge.real { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge.ai { background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge.inspect { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .meter-row { margin: 14px 0; }
    .meter-labels { display: flex; justify-content: space-between; font-size: 0.88rem; color: var(--muted); margin-bottom: 6px; }
    .bar { height: 10px; border-radius: 999px; background: #1e2946; overflow: hidden; }
    .bar-fill { height: 100%; width: 0%; transition: width 0.35s ease; }
    .metrics-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.88rem; }
    .metrics-table td { padding: 7px 4px; border-bottom: 1px solid #1e2845; }
    .metrics-table td:first-child { color: var(--muted); }
    .metrics-table td:last-child { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .muted { color: var(--muted); font-size: 0.9rem; }
    .toggle { display: flex; align-items: center; gap: 8px; margin-top: 12px; font-size: 0.9rem; color: var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>AI Image Detector</h1>
      <p class="subtitle">Hybrid Spatial + SRM Noise-Residual + 2D FFT Spectrum + Forensic Analyzer</p>
      <div class="banner" id="statusBanner">Loading model status...</div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="dropzone" id="dropzone">
          <strong>Drop an image here or click to browse</strong>
          <p>Supports JPG, PNG, WEBP, BMP, TIFF</p>
          <input type="file" id="fileInput" accept="image/*" hidden />
        </div>

        <label class="toggle">
          <input type="checkbox" id="ttaToggle" checked />
          Use multi-view Test-Time Augmentation (TTA)
        </label>

        <button class="btn" id="analyzeBtn" disabled>Analyze Image</button>

        <div class="preview-wrap" id="previewWrap">
          <img id="previewImg" alt="Uploaded preview" />
          <p class="muted" id="fileName"></p>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Detection Result</h3>
        <div id="resultArea">
          <p class="muted">Upload an image and click <strong>Analyze Image</strong> to inspect authenticity and forensic signals.</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const previewWrap = document.getElementById("previewWrap");
    const previewImg = document.getElementById("previewImg");
    const fileName = document.getElementById("fileName");
    const resultArea = document.getElementById("resultArea");
    const statusBanner = document.getElementById("statusBanner");
    const ttaToggle = document.getElementById("ttaToggle");

    let selectedFile = null;
    let modelLoaded = false;

    fetch("/api/health")
      .then(r => r.json())
      .then(data => {
        modelLoaded = !!data.model_loaded;
        if (modelLoaded) {
          statusBanner.textContent = `Model loaded: ${data.checkpoint} (threshold=${data.threshold})`;
        } else {
          statusBanner.textContent = "No trained checkpoint loaded — running in pure forensic inspection mode.";
        }
      })
      .catch(() => {
        statusBanner.textContent = "Server ready.";
      });

    function setFile(file) {
      if (!file) return;
      selectedFile = file;
      analyzeBtn.disabled = false;
      fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      const reader = new FileReader();
      reader.onload = e => {
        previewImg.src = e.target.result;
        previewWrap.style.display = "block";
      };
      reader.readAsDataURL(file);
    }

    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", e => setFile(e.target.files[0]));
    dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", e => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });

    analyzeBtn.addEventListener("click", async () => {
      if (!selectedFile) return;
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyzing...";
      resultArea.innerHTML = `<p class="muted">Running spatial, SRM noise-residual, FFT spectrum, and ELA checks...</p>`;

      const form = new FormData();
      form.append("file", selectedFile);
      const endpoint = modelLoaded
        ? `/api/predict?explain=true&tta=${ttaToggle.checked}`
        : `/api/inspect`;

      try {
        const res = await fetch(endpoint, { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Request failed");
        renderResult(data);
      } catch (err) {
        resultArea.innerHTML = `<p style="color:#f87171;">Error: ${err.message}</p>`;
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Image";
      }
    });

    function renderResult(data) {
      let headerHtml = "";
      let forensics = data.forensics || data;

      if (data.label) {
        const isAi = data.label === "ai_generated";
        const badgeCls = isAi ? "ai" : "real";
        const badgeText = isAi ? "AI-GENERATED" : "REAL";
        const aiPct = (data.ai_probability * 100).toFixed(1);
        const realPct = (data.real_probability * 100).toFixed(1);
        const barColor = isAi ? "#ef4444" : "#10b981";
        headerHtml = `
          <span class="badge ${badgeCls}">${badgeText}</span>
          <div class="meter-row">
            <div class="meter-labels">
              <span>AI Probability: <strong>${aiPct}%</strong></span>
              <span>Real Probability: <strong>${realPct}%</strong></span>
            </div>
            <div class="bar"><div class="bar-fill" style="width:${aiPct}%; background:${barColor};"></div></div>
            <p class="muted" style="margin-top:8px;">Confidence: ${(data.confidence * 100).toFixed(1)}% (threshold=${data.threshold})</p>
          </div>
        `;
      } else {
        headerHtml = `<span class="badge inspect">FORENSIC REPORT ONLY</span>`;
      }

      const h = forensics.highlights || {};
      const m = forensics.metadata || {};
      const rows = Object.entries(h)
        .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
        .join("");

      const aiTags = (m.ai_metadata_matches && m.ai_metadata_matches.length)
        ? m.ai_metadata_matches.join(", ")
        : "None detected";

      resultArea.innerHTML = `
        ${headerHtml}
        <h4 style="margin:14px 0 6px;">Forensic Highlights</h4>
        <table class="metrics-table">${rows}</table>
        <h4 style="margin:14px 0 6px;">Metadata & Provenance</h4>
        <table class="metrics-table">
          <tr><td>Format / Size</td><td>${m.format || "?"} (${m.width || 0}x${m.height || 0})</td></tr>
          <tr><td>EXIF Present</td><td>${m.has_exif ? "Yes" : "No"}</td></tr>
          <tr><td>Camera</td><td>${[m.camera_make, m.camera_model].filter(Boolean).join(" ") || "N/A"}</td></tr>
          <tr><td>AI Metadata Tags</td><td>${aiTags}</td></tr>
        </table>
      `;
    }
  </script>
</body>
</html>
"""


def _decode_upload(raw_bytes: bytes) -> tuple[np.ndarray, str]:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = img.format or "PNG"
            rgb = load_rgb_image(img)
        return rgb, fmt
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def create_app(
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    threshold: float | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="AI Image Detector", version="0.1.0")

    predictor: AIDetectorPredictor | None = None
    ckpt_label: str | None = None
    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
        predictor = AIDetectorPredictor(ckpt_file, device=device, threshold=threshold)
        ckpt_label = ckpt_file.name

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(content=_HTML_PAGE)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": predictor is not None,
            "checkpoint": ckpt_label,
            "threshold": round(predictor.threshold, 4) if predictor is not None else None,
        }

    @app.post("/api/inspect")
    async def inspect_upload(file: UploadFile = File(...)) -> JSONResponse:
        raw_bytes = await file.read()
        rgb, fmt = _decode_upload(raw_bytes)
        suffix = Path(file.filename or "upload.png").suffix or f".{fmt.lower()}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)
        try:
            report = summarize_forensics(rgb, image_path=tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return JSONResponse(content=report)

    @app.post("/api/predict")
    async def predict_upload(
        file: UploadFile = File(...),
        tta: bool = Query(default=True),
        explain: bool = Query(default=True),
    ) -> JSONResponse:
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="No model checkpoint loaded on server. Use /api/inspect or pass --checkpoint.",
            )
        raw_bytes = await file.read()
        rgb, fmt = _decode_upload(raw_bytes)
        suffix = Path(file.filename or "upload.png").suffix or f".{fmt.lower()}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)
        try:
            probs = [predictor._score_rgb(rgb)]
            if tta:
                probs.append(predictor._score_rgb(np.ascontiguousarray(rgb[:, ::-1, :])))
                h, w = rgb.shape[:2]
                if min(h, w) >= 64:
                    ch, cw = int(h * 0.85), int(w * 0.85)
                    top, left = (h - ch) // 2, (w - cw) // 2
                    probs.append(
                        predictor._score_rgb(
                            np.ascontiguousarray(rgb[top : top + ch, left : left + cw, :])
                        )
                    )
            ai_prob = float(np.mean(probs))
            real_prob = 1.0 - ai_prob
            is_ai = ai_prob >= predictor.threshold
            payload: dict[str, Any] = {
                "filename": file.filename or "upload",
                "label": "ai_generated" if is_ai else "real",
                "ai_probability": round(ai_prob, 4),
                "real_probability": round(real_prob, 4),
                "confidence": round(ai_prob if is_ai else real_prob, 4),
                "threshold": round(predictor.threshold, 4),
                "forensics": summarize_forensics(rgb, image_path=tmp_path)
                if explain
                else {"metadata": inspect_metadata(tmp_path)},
            }
        finally:
            tmp_path.unlink(missing_ok=True)
        return JSONResponse(content=payload)

    return app
