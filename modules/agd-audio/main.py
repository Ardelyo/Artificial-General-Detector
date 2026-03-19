"""
AGD Audio Module — SOTA Upgrade
=================================
Methods:
1. MFCC (Mel-Frequency Cepstral Coefficients) statistics
2. LFCC (Linear Frequency Cepstral Coefficients) statistics
3. Spectral Flux (vocoder artifact detection)
4. Zero-Crossing Rate analysis
5. Ensemble of all above
"""
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import io
import struct
import wave

app = FastAPI(title="AGD Audio Module — SOTA")

class AudioAnalysisResult(BaseModel):
    score: float
    breakdown: dict

@app.get("/")
def read_root():
    return {"status": "ok", "module": "agd-audio", "version": "2.0-SOTA"}

# ── Audio loading ──────────────────────────────────────────────────

def load_audio_bytes(audio_bytes: bytes, sr=16000):
    """Attempt to load raw PCM or WAV audio."""
    try:
        bio = io.BytesIO(audio_bytes)
        with wave.open(bio, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.array(struct.unpack(f'<{n_frames * n_channels}h', raw),
                                   dtype=np.float32) / 32768.0
            elif sampwidth == 1:
                samples = np.array(struct.unpack(f'{n_frames * n_channels}B', raw),
                                   dtype=np.float32) / 128.0 - 1.0
            else:
                return None, 0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return samples, framerate
    except:
        return None, 0

# ── MFCC computation (from scratch) ───────────────────────────────

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def compute_mfcc(signal, sr=16000, n_mfcc=13, n_fft=512, hop=160, n_mels=26):
    """Compute MFCCs from scratch using numpy."""
    if signal is None or len(signal) < n_fft:
        return None

    # Pre-emphasis
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Framing
    n_frames = 1 + (len(emphasized) - n_fft) // hop
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        frames[i] = emphasized[i * hop: i * hop + n_fft]

    # Hamming window
    frames *= np.hamming(n_fft)

    # FFT
    mag = np.abs(np.fft.rfft(frames, n_fft))
    power = (mag ** 2) / n_fft

    # Mel filterbank
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bins[m-1], bins[m]):
            fbank[m-1, k] = (k - bins[m-1]) / max(bins[m] - bins[m-1], 1)
        for k in range(bins[m], bins[m+1]):
            fbank[m-1, k] = (bins[m+1] - k) / max(bins[m+1] - bins[m], 1)

    filter_energies = np.dot(power, fbank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_energies = np.log(filter_energies)

    # DCT (Type-II)
    mfcc = np.zeros((n_frames, n_mfcc))
    for n in range(n_mfcc):
        mfcc[:, n] = np.sum(log_energies * np.cos(np.pi * n * (np.arange(n_mels) + 0.5) / n_mels), axis=1)

    return mfcc

# ── LFCC computation (linear filterbank) ──────────────────────────

def compute_lfcc(signal, sr=16000, n_lfcc=13, n_fft=512, hop=160, n_filters=26):
    """LFCC: same as MFCC but with linear frequency filterbank."""
    if signal is None or len(signal) < n_fft:
        return None

    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    n_frames = 1 + (len(emphasized) - n_fft) // hop
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        frames[i] = emphasized[i * hop: i * hop + n_fft]
    frames *= np.hamming(n_fft)
    mag = np.abs(np.fft.rfft(frames, n_fft))
    power = (mag ** 2) / n_fft

    # Linear filterbank
    hz_points = np.linspace(0, sr / 2, n_filters + 2)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        for k in range(bins[m-1], bins[m]):
            fbank[m-1, k] = (k - bins[m-1]) / max(bins[m] - bins[m-1], 1)
        for k in range(bins[m], bins[m+1]):
            fbank[m-1, k] = (bins[m+1] - k) / max(bins[m+1] - bins[m], 1)

    filter_energies = np.dot(power, fbank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_energies = np.log(filter_energies)

    lfcc = np.zeros((n_frames, n_lfcc))
    for n in range(n_lfcc):
        lfcc[:, n] = np.sum(log_energies * np.cos(np.pi * n * (np.arange(n_filters) + 0.5) / n_filters), axis=1)

    return lfcc

# ── Spectral Flux (vocoder artifact) ──────────────────────────────

def compute_spectral_flux(signal, sr=16000, n_fft=512, hop=160):
    """Measures spectral change between consecutive frames."""
    if signal is None or len(signal) < n_fft * 2:
        return 0.0, 0.5

    n_frames = 1 + (len(signal) - n_fft) // hop
    prev_mag = None
    flux_values = []

    for i in range(n_frames):
        frame = signal[i * hop: i * hop + n_fft]
        mag = np.abs(np.fft.rfft(frame * np.hamming(n_fft)))
        if prev_mag is not None:
            flux = float(np.sum((mag - prev_mag) ** 2))
            flux_values.append(flux)
        prev_mag = mag

    if not flux_values:
        return 0.0, 0.5

    mean_flux = float(np.mean(flux_values))
    std_flux = float(np.std(flux_values))
    cv_flux = std_flux / (mean_flux + 1e-10)

    # AI audio tends to have very consistent spectral flux (low CV)
    score = max(0, 1.0 - (cv_flux / 2.0))
    return cv_flux, float(min(score, 1.0))

# ── Zero-Crossing Rate ────────────────────────────────────────────

def compute_zcr_features(signal):
    """Zero-crossing rate statistics."""
    if signal is None or len(signal) < 1600:
        return 0.0, 0.5

    frame_len = 800
    hop = 400
    n_frames = (len(signal) - frame_len) // hop
    zcrs = []
    for i in range(n_frames):
        frame = signal[i * hop: i * hop + frame_len]
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame))) > 0)) / frame_len
        zcrs.append(zcr)

    if not zcrs:
        return 0.0, 0.5

    zcr_std = float(np.std(zcrs))
    # AI audio: more uniform ZCR (low std)
    score = max(0, 1.0 - (zcr_std / 0.1))
    return zcr_std, float(min(score, 1.0))

# ── Cepstral Feature Scoring ──────────────────────────────────────

def score_cepstral(coefficients):
    """Score based on cepstral coefficient statistics."""
    if coefficients is None:
        return 0.5, {}

    means = np.mean(coefficients, axis=0)
    stds = np.std(coefficients, axis=0)

    # AI audio: lower variance across cepstral coefficients
    avg_std = float(np.mean(stds))
    # Higher-order coeff std ratio (AI has more uniform high coeff)
    if len(stds) > 6:
        high_low_ratio = float(np.mean(stds[6:])) / (float(np.mean(stds[:6])) + 1e-10)
    else:
        high_low_ratio = 1.0

    score = max(0.0, 1.0 - (avg_std / 5.0))
    return float(min(score, 1.0)), {
        "avg_coeff_std": float(round(float(avg_std), 4)),
        "high_low_ratio": float(round(float(high_low_ratio), 4))
    }

# ── API ────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AudioAnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    contents = await file.read()
    signal, sr = load_audio_bytes(contents)

    if signal is None or len(signal) < 1600:
        return {
            "score": 0.5,
            "breakdown": {"error": "Could not decode audio or audio too short"}
        }

    # Compute all features
    mfcc = compute_mfcc(signal, sr)
    lfcc = compute_lfcc(signal, sr)
    mfcc_score, mfcc_detail = score_cepstral(mfcc)
    lfcc_score, lfcc_detail = score_cepstral(lfcc)
    flux_cv, flux_score = compute_spectral_flux(signal, sr)
    zcr_std, zcr_score = compute_zcr_features(signal)

    # Ensemble
    master = (mfcc_score * 0.30 +
              lfcc_score * 0.30 +
              flux_score * 0.25 +
              zcr_score * 0.15)

    return {
        "score": float(round(float(master), 4)),  # type: ignore
        "breakdown": {
            "mfcc_score": float(round(float(mfcc_score), 4)),  # type: ignore
            "lfcc_score": float(round(float(lfcc_score), 4)),  # type: ignore
            "spectral_flux_score": float(round(float(flux_score), 4)),  # type: ignore
            "zcr_score": float(round(float(zcr_score), 4)),  # type: ignore
            "raw_flux_cv": float(round(float(flux_cv), 4)),  # type: ignore
            "raw_zcr_std": float(round(float(zcr_std), 4)),  # type: ignore
            **{f"mfcc_{k}": float(round(float(v), 4)) if isinstance(v, (int, float)) else v for k, v in mfcc_detail.items()},  # type: ignore
            **{f"lfcc_{k}": float(round(float(v), 4)) if isinstance(v, (int, float)) else v for k, v in lfcc_detail.items()},  # type: ignore
        }
    }
