"""
AGD Video Module — SOTA Upgrade
=================================
Methods:
1. Temporal Frame Difference Energy (inter-frame inconsistency)
2. Motion Vector Consistency (simplified optical flow proxy)
3. Spatial Frequency Consistency across frames
4. Color Histogram Temporal Stability
5. Ensemble of all above
"""
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import struct

app = FastAPI(title="AGD Video Module — SOTA")

class VideoAnalysisResult(BaseModel):
    score: float
    breakdown: dict

@app.get("/")
def read_root():
    return {"status": "ok", "module": "agd-video", "version": "2.0-SOTA"}

# ── Frame extraction (supports raw frames or simple video) ─────────

def extract_frames_from_bytes(video_bytes: bytes, max_frames=30):
    """
    Attempt to extract frames. For real deployment, use opencv.
    This lightweight version handles GIF or raw frame concatenation.
    """
    frames = []
    try:
        img = Image.open(io.BytesIO(video_bytes))
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            # Animated GIF or similar
            for i in range(min(img.n_frames, max_frames)):
                img.seek(i)
                frame = np.array(img.convert("RGB").resize((128, 128)), dtype=np.float32)
                frames.append(frame)
        else:
            # Single image — treat as 1-frame "video"
            frames.append(np.array(img.convert("RGB").resize((128, 128)), dtype=np.float32))
    except:
        pass

    return frames

# ── Method 1: Temporal Frame Difference Energy ────────────────────

def compute_frame_diff_energy(frames):
    """
    ΔF_t = ||frame_t - frame_{t-1}||²_F
    Anomaly if Z-score of ΔF_t is high.
    """
    if len(frames) < 2:
        return 0.0, 0.5

    diffs = []
    for i in range(1, len(frames)):
        diff = np.sum((frames[i] - frames[i-1]) ** 2)
        diffs.append(float(diff))

    mu = np.mean(diffs)
    sigma = np.std(diffs) + 1e-10
    cv = sigma / mu if mu > 0 else 0

    # Deepfakes have INCONSISTENT frame differences (high CV)
    # Natural video has smooth temporal evolution (moderate CV)
    if cv > 1.5:
        score = 0.9  # Very suspicious
    elif cv > 0.8:
        score = cv / 2.0
    else:
        score = 0.1

    return float(cv), float(min(max(score, 0), 1))

# ── Method 2: Motion Consistency (gradient-based flow proxy) ──────

def compute_motion_consistency(frames):
    """Simplified optical flow proxy using spatial gradients."""
    if len(frames) < 3:
        return 0.0, 0.5

    flow_magnitudes = []
    for i in range(1, len(frames)):
        gray_curr = np.mean(frames[i], axis=2)
        gray_prev = np.mean(frames[i-1], axis=2)

        # Temporal gradient
        It = gray_curr - gray_prev

        # Spatial gradients
        Ix = np.diff(gray_curr, axis=1, prepend=gray_curr[:, :1])
        Iy = np.diff(gray_curr, axis=0, prepend=gray_curr[:1, :])

        # Flow magnitude proxy: |It| / (|Ix| + |Iy| + epsilon)
        flow_mag = np.mean(np.abs(It) / (np.abs(Ix) + np.abs(Iy) + 1e-6))
        flow_magnitudes.append(float(flow_mag))

    if not flow_magnitudes:
        return 0.0, 0.5

    # Consistency: std of flow magnitudes
    flow_std = float(np.std(flow_magnitudes))
    # Deepfakes: inconsistent flow across frames
    score = min(flow_std / 0.5, 1.0)

    return flow_std, float(max(0, score))

# ── Method 3: Spatial Frequency Consistency ───────────────────────

def compute_frequency_consistency(frames):
    """Check if frequency content is consistent across frames."""
    if len(frames) < 2:
        return 0.0, 0.5

    hf_energies = []
    for f in frames:
        gray = np.mean(f, axis=2)
        fft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(fft))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((Y - cy)**2 + (X - cx)**2) > min(h, w) * 0.35
        hf_energy = float(np.mean(magnitude[mask]))
        hf_energies.append(hf_energy)

    cv = float(np.std(hf_energies)) / (float(np.mean(hf_energies)) + 1e-10)
    # High CV in frequency → inconsistent processing → suspicious
    score = min(cv / 0.5, 1.0)
    return cv, float(max(0, score))

# ── Method 4: Color Histogram Temporal Stability ──────────────────

def compute_color_stability(frames):
    """AI videos may have color distribution shifts between frames."""
    if len(frames) < 2:
        return 0.0, 0.5

    hist_diffs = []
    for i in range(1, len(frames)):
        for c in range(3):
            h1, _ = np.histogram(frames[i-1][:, :, c], bins=32, range=(0, 256))
            h2, _ = np.histogram(frames[i][:, :, c], bins=32, range=(0, 256))
            h1 = h1 / (h1.sum() + 1e-10)
            h2 = h2 / (h2.sum() + 1e-10)
            # Chi-squared distance
            chi2 = float(np.sum((h1 - h2)**2 / (h1 + h2 + 1e-10)))
            hist_diffs.append(chi2)

    mean_diff = float(np.mean(hist_diffs))
    # High chi-squared → unstable colors → suspicious
    score = min(mean_diff / 0.1, 1.0)
    return mean_diff, float(max(0, score))

# ── Ensemble API ───────────────────────────────────────────────────

@app.post("/analyze", response_model=VideoAnalysisResult)
async def analyze_video(file: UploadFile = File(...)):
    contents = await file.read()
    frames = extract_frames_from_bytes(contents)

    if len(frames) < 2:
        return {
            "score": 0.5,
            "breakdown": {"warning": "Could not extract multiple frames. Upload GIF or multi-frame format."}
        }

    diff_cv, diff_score = compute_frame_diff_energy(frames)
    flow_std, flow_score = compute_motion_consistency(frames)
    freq_cv, freq_score = compute_frequency_consistency(frames)
    color_diff, color_score = compute_color_stability(frames)

    master = (diff_score * 0.30 +
              flow_score * 0.25 +
              freq_score * 0.25 +
              color_score * 0.20)

    return {
        "score": round(float(min(max(master, 0), 1)), 4),
        "breakdown": {
            "frame_diff_energy": round(diff_score, 4),
            "motion_consistency": round(flow_score, 4),
            "frequency_consistency": round(freq_score, 4),
            "color_stability": round(color_score, 4),
            "raw_diff_cv": round(diff_cv, 4),
            "raw_flow_std": round(flow_std, 4),
            "raw_freq_cv": round(freq_cv, 4),
            "raw_color_diff": round(color_diff, 4),
            "n_frames_analyzed": len(frames),
        }
    }
