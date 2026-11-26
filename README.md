# AI Image Detector

Hybrid deep-learning and frequency/forensic detector for classifying **real** vs. **AI-generated** images.

## Architecture

`HybridAIDetector` fuses three complementary branches before a temperature-calibrated binary classifier:

1. **Spatial + SRM Noise-Residual CNN (`SpatialResidualEncoder`)**:
   - Takes a 6-channel input: normalized RGB (`3` channels) concatenated with 3 Steganalysis Rich Model (SRM) high-pass noise residual maps (`3` channels).
   - Uses residual convolutional blocks with Squeeze-and-Excitation (SE) channel attention and global average + max pooling.
2. **2D FFT Frequency Spectrum CNN (`FrequencyEncoder`)**:
   - Processes the centered 2D log-magnitude FFT power spectrum (with Hann windowing) to detect spectral roll-off differences and periodic upsampling/checkerboard artifacts.
3. **Handcrafted Forensic MLP (`ForensicMLP`)**:
   - Processes a standardized 32-dimensional forensic feature vector covering SRM residual moments/tails, 8 radial FFT frequency bands, spectral slope, axial grid-peak ratio, DCT band energy & AC kurtosis, JPEG Error Level Analysis (ELA) residuals, cross-channel RGB noise correlation, chroma/luma noise ratio, Laplacian sharpness variance, and saturation spread.

## Installation

```bash
cd C:\Users\acer\.gemini\antigravity\scratch\ai_image_detector
pip install -e . --no-build-isolation --no-deps
```

## Quick Start

### 1. Generate a Sample Benchmark Dataset (Smoke Test)

```bash
ai-detect generate-sample --output ./sample_data --num-per-class 60
```

> **Note:** `generate-sample` creates synthetic procedural approximations for pipeline testing and verification. For real-world detection accuracy across modern diffusion/GAN generators (Midjourney, Flux, SDXL, DALL-E), train on a real photographic + AI-generated dataset such as **GenImage**, **CIFAKE**, or your own curated `real/` and `ai/` image folders.

### 2. Train the Model

Supports either `dataset/real/` + `dataset/ai/` (auto-split into stratified train/val/test), pre-split `train/` + `val/` + `test/` folders, or a CSV manifest (`path,label`).

```bash
ai-detect train --data ./sample_data --output ./runs/demo --epochs 12
```

Training automatically:
- Applies flip, crop, and random JPEG recompression augmentations (preventing format leakage between PNG and JPEG)
- Saves the best validation checkpoint (`best_model.pt`)
- Fits post-hoc temperature scaling and F1-optimal decision threshold on the validation set
- Evaluates on the held-out test split and exports `metrics.json`

### 3. Predict on Images

```bash
ai-detect predict --checkpoint ./runs/demo/best_model.pt --input ./sample_data/ai/ai_0000.jpg --explain
```

Or run batch prediction on a folder with JSON output:

```bash
ai-detect predict --checkpoint ./runs/demo/best_model.pt --input ./some_folder --json
```

### 4. Evaluate on a Test Set

```bash
ai-detect evaluate --checkpoint ./runs/demo/best_model.pt --data ./some_test_set
```

### 5. Pure Forensic Inspection (No Checkpoint Needed)

```bash
ai-detect inspect --input ./path/to/image.jpg
```

### 6. Host Local Web UI & API

```bash
ai-detect serve --checkpoint ./runs/demo/best_model.pt --host 127.0.0.1 --port 8000
```

Then open `http://localhost:8000` in your browser.

## Running Tests

```bash
pytest
```
