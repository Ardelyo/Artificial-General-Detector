<div align="center">
  <h1>🛡️ Artificial General Detector (AGD)</h1>
  <p><strong>The Most Advanced, Multi-Modal AI Content Detection Engine</strong></p>
  <p><em>Created by <strong>OurCreativity</strong> | Admin: <strong>Ardelyo</strong></em></p>
</div>

---

## 🚀 Overview

Artificial General Detector (AGD) is a state-of-the-art forensic framework designed to identify AI-generated content across **four modalities**: Text, Image, Audio, and Video.

Unlike simple classification models, AGD employs an **Ultra-Deep Detection Engine** utilizing **27 independent forensic techniques** (15 for image, 12 for text), combining statistical heuristics, spectral analysis, noise residual mapping, and deep learning (RoBERTa & CNNs) into a unified, weighted ensemble scoring system.

## 🧠 Core Features

### 🖼️ Deep Image Forensics (15 Techniques)
- **Error Level Analysis (ELA):** Pixel-perfect JPEG recompression differentials.
- **SRM Multi-Kernel Analysis:** Extracts camera sensor noise residuals (1st, 2nd, and 3rd order) to differentiate real lenses from synthetic diffusion generation.
- **FFT Radial Spectrum & High-Frequency Energy:** Detects rigid GAN grid artifacts vs. over-smoothed diffusion models.
- **Edge Sharpness & Noise Floor Variance:** Sophisticated local block analysis.
- **Chromatic Aberration Test:** Validates lens dispersion effects.

### 📝 Deep Text Forensics (12 Techniques)
- **Token Perplexity (RoBERTa-base):** Transformer-based semantic probability profiling.
- **Burstiness & Sliding-Window Entropy:** Sentence length variation and localized vocabulary diversity.
- **N-Gram Repetition Heatmaps:** Spots syntax loops common in LLMs.
- **Stylometrics & Transition Density:** Analyzes use of formal marker words ("furthermore", "in conclusion").

### 🔊 Audio & 🎬 Video
- **Audio:** MFCC/LFCC extraction, Spectral Flux, and ZCR to catch TTS vocoder artifacts.
- **Video:** Temporal Frame Diff Energy and Spatial Frequency Consistency across frames.

## 📊 Benchmark Performance

In a rigorous 500-sample comparative benchmark against leading global detectors (GPTZero, ZeroGPT, CIFAKE CNN):
- AGD achieved competitive or superior F1 scores while prioritizing **low False Positive Rates**.
- Detailed diagnostics are provided for every single sample, avoiding black-box "AI or Not" guesses.

## ⚙️ Quick Start

**Prerequisites:** Python 3.9+, Node.js (for optional dashboard).

```bash
# Clone the repository
git clone https://github.com/YourUsername/AGD.git
cd AGD

# Install dependencies
pip install fastapi pydantic numpy transformers scipy pillow scikit-learn pandas docx matplotlib

# Run the Ultra-Deep Benchmark
python tools/ultra_benchmark.py

# Start the Core API Server
uvicorn orchestrator.main:app --reload
```

## 📄 Documentation

- Official Research Paper: Provided in the `/docs` directory (generated via benchmark scripts).
- Extensive methodology details are available in the engine source files (`modules/deep_image_forensics.py`, etc.).

---
*© 2026 OurCreativity - Developing the Future of Truth*
