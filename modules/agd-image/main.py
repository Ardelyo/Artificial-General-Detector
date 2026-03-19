"""
AGD Image Module — SOTA Upgrade
=================================
Methods:
1. Error Level Analysis (ELA) — JPEG recompression
2. Frequency Domain Analysis (FFT high-freq energy ratio)
3. SRM (Steganalysis Rich Model) noise residuals
4. Ensemble of all above
"""
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import math

app = FastAPI(title="AGD Image Module — SOTA")

class ImageAnalysisResult(BaseModel):
    score: float
    breakdown: dict

@app.get("/")
def read_root():
    return {"status": "ok", "module": "agd-image", "version": "2.0-SOTA"}

# ── Method 1: Error Level Analysis ─────────────────────────────────

def compute_ela_baseline(image_bytes: bytes, quality=90):
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=quality)
        compressed = Image.open(buf)
        diff = np.abs(np.array(original).astype(np.float32) -
                      np.array(compressed).astype(np.float32))
        avg_diff = float(np.mean(diff))
        normalized = min(max(float((avg_diff - 1.0) / 5.0), 0.0), 1.0)
        return avg_diff, normalized
    except:
        return 0.0, 0.5

# ── Method 2: FFT Frequency Domain ────────────────────────────────

def compute_frequency_score(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
        arr = np.array(img, dtype=np.float32)
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # High-frequency mask: outer 35% radius
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        hf_mask = dist > min(h, w) * 0.35

        hf_energy = float(np.mean(magnitude[hf_mask]))
        total_energy = float(np.mean(magnitude)) + 1e-6
        ratio = hf_energy / total_energy

        # GAN images: high HFR (grid artifacts). Diffusion: low HFR (too smooth)
        # We detect BOTH: deviation from natural range [0.8, 1.5]
        if ratio < 0.8:
            score = min((0.8 - ratio) / 0.5, 1.0)  # Too smooth → diffusion
        elif ratio > 1.5:
            score = min((ratio - 1.5) / 1.0, 1.0)  # Grid artifacts → GAN
        else:
            score = 0.1  # Normal range

        return ratio, float(score)
    except:
        return 0.0, 0.5

# ── Method 3: SRM Noise Residual Analysis ──────────────────────────

def compute_srm_score(image_bytes: bytes):
    """Simplified SRM: extract noise residuals using 3rd-order linear predictor."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
        arr = np.array(img, dtype=np.float64)

        # 3rd-order SRM-like kernel: predict center pixel from neighbors
        # Residual = center - average(4 neighbors)
        padded = np.pad(arr, 1, mode='reflect')
        predicted = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        residual = arr - predicted

        # Statistics of residuals
        residual_std = float(np.std(residual))
        residual_kurtosis = float(np.mean((residual - np.mean(residual))**4) /
                                  (np.std(residual)**4 + 1e-10))

        # AI images have LOWER noise residual variance (cleaner)
        # Real photos have higher noise (sensor noise, compression artifacts)
        # Kurtosis: AI images tend toward lower kurtosis (more Gaussian noise)
        std_score = max(0.0, float(1.0 - (residual_std / 15.0)))  # Low std → AI
        kurt_score = max(0.0, float(1.0 - (residual_kurtosis / 10.0)))

        combined = std_score * 0.6 + kurt_score * 0.4
        return residual_std, residual_kurtosis, float(min(max(combined, 0), 1))
    except:
        return 0.0, 0.0, 0.5

# ── Method 4: Color Histogram Uniformity ───────────────────────────

def compute_color_uniformity(image_bytes: bytes):
    """AI images often have more uniform color distributions."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
        arr = np.array(img)
        scores = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 256))
            hist = hist / hist.sum()
            # Entropy of histogram: uniform → high entropy → AI
            ent = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
            max_ent = np.log2(32)
            scores.append(ent / max_ent)
        uniformity = float(np.mean(scores))
        # Very high uniformity → possibly AI
        score = max(0.0, float((uniformity - 0.7) / 0.3))
        return uniformity, float(min(score, 1.0))
    except:
        return 0.0, 0.5

# ── Ensemble ───────────────────────────────────────────────────────

@app.post("/analyze", response_model=ImageAnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()

    avg_diff, ela_score = compute_ela_baseline(contents)
    freq_ratio, freq_score = compute_frequency_score(contents)
    res_std, res_kurt, srm_score = compute_srm_score(contents)
    uniformity, color_score = compute_color_uniformity(contents)

    # Weighted ensemble
    master = (ela_score * 0.30 +
              freq_score * 0.25 +
              srm_score * 0.25 +
              color_score * 0.20)

    return {
        "score": round(float(min(max(float(master), 0.0), 1.0)), 4),  # type: ignore
        "breakdown": {
            "error_level_analysis": round(float(ela_score), 4),  # type: ignore
            "frequency_domain": round(float(freq_score), 4),  # type: ignore
            "srm_noise_residual": round(float(srm_score), 4),  # type: ignore
            "color_uniformity": round(float(color_score), 4),  # type: ignore
            "raw_ela_diff": round(float(avg_diff), 4),  # type: ignore
            "raw_freq_ratio": round(float(freq_score), 4),  # type: ignore
            "raw_residual_std": round(float(res_std), 4),  # type: ignore
            "raw_residual_kurtosis": round(float(res_kurt), 4),  # type: ignore
        }
    }
