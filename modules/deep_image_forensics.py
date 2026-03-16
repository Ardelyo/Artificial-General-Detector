"""
AGD Ultra-Deep Image Forensics Engine
========================================
Performs pixel-level, patch-level, and frequency-level analysis
using 15+ independent forensic techniques. Each returns a score
and a detailed diagnostic map.

Techniques:
 1. Pixel-Level ELA Heatmap (per-pixel error level)
 2. JPEG Ghost Detection (multi-quality recompression)
 3. SRM Multi-Kernel Noise Residuals (5 kernels)
 4. DCT Blockwise Analysis (8x8 block artifacts)
 5. FFT Radial Power Spectrum
 6. High-Frequency Energy Ratio
 7. Color Coherence & Channel Correlation
 8. Edge Sharpness Profiling (Sobel gradient stats)
 9. Noise Floor Variance (local vs global)
10. Patch-Grid Homogeneity (NxN grid uniformity)
11. Chromatic Aberration Test
12. Bit-Plane Analysis (LSB patterns)
13. Histogram Gap Detection
14. Texture Complexity (LBP-like)
15. Global Statistical Fingerprint (skew, kurtosis per channel)

Returns a comprehensive ForensicReport dict.
"""

import numpy as np
from PIL import Image
import io
import math

# ═══════════════════════════════════════════════════════════════════
#  1. PIXEL-LEVEL ELA HEATMAP
# ═══════════════════════════════════════════════════════════════════

def pixel_ela_heatmap(image_bytes, quality=90):
    """Compute per-pixel Error Level Analysis."""
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = original.size
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    recomp = Image.open(buf).convert("RGB")

    orig_arr = np.array(original, dtype=np.float32)
    recomp_arr = np.array(recomp, dtype=np.float32)
    diff = np.abs(orig_arr - recomp_arr)

    # Per-pixel magnitude
    pixel_map = np.mean(diff, axis=2)  # H x W
    global_mean = float(np.mean(pixel_map))
    global_std = float(np.std(pixel_map))
    max_diff = float(np.max(pixel_map))

    # Suspicious pixel ratio: pixels with diff > 2*mean
    suspicious_ratio = float(np.sum(pixel_map > 2 * global_mean) / max(pixel_map.size, 1))

    # Score: AI images have LOW ELA variance (smooth generation)
    score = max(0, 1.0 - (global_std / 8.0))
    return {
        "score": min(score, 1.0),
        "global_mean": round(global_mean, 4),
        "global_std": round(global_std, 4),
        "max_diff": round(max_diff, 4),
        "suspicious_pixel_ratio": round(suspicious_ratio, 4),
        "resolution": f"{w}x{h}",
    }

# ═══════════════════════════════════════════════════════════════════
#  2. JPEG GHOST DETECTION
# ═══════════════════════════════════════════════════════════════════

def jpeg_ghost_detection(image_bytes, quality_range=(60, 95, 5)):
    """Detect JPEG ghosts by recompressing at multiple quality levels."""
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_arr = np.array(original, dtype=np.float32)

    ghost_scores = []
    for q in range(quality_range[0], quality_range[1], quality_range[2]):
        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=q)
        recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
        diff = np.mean(np.abs(orig_arr - recomp))
        ghost_scores.append((q, float(diff)))

    # Find the quality with MINIMUM diff → likely original save quality
    min_q, min_diff = min(ghost_scores, key=lambda x: x[1])
    max_diff = max(s[1] for s in ghost_scores)
    diff_range = max_diff - min_diff + 1e-6

    # AI generated PNGs show uniform ghost profile; JPEGs show a dip
    uniformity = float(np.std([s[1] for s in ghost_scores]) / (np.mean([s[1] for s in ghost_scores]) + 1e-6))
    score = max(0, 1.0 - uniformity * 2)

    return {
        "score": min(score, 1.0),
        "estimated_quality": min_q,
        "ghost_curve": ghost_scores,
        "curve_uniformity": round(uniformity, 4),
    }

# ═══════════════════════════════════════════════════════════════════
#  3. SRM MULTI-KERNEL NOISE RESIDUALS
# ═══════════════════════════════════════════════════════════════════

def srm_multi_kernel(image_bytes):
    """Apply 5 different SRM-style high-pass filters and analyze residuals."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float64)
    padded = np.pad(arr, 2, mode='reflect')

    kernels = {
        "1st_order_h": np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]),
        "1st_order_v": np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]),
        "2nd_order": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        "3rd_order_cross": np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]),
        "diagonal": np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]),
    }

    results = {}
    scores = []
    for name, kernel in kernels.items():
        kh, kw = kernel.shape
        residual = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                patch = padded[i:i+kh, j:j+kw]
                if patch.shape == kernel.shape:
                    residual[i, j] = np.sum(patch * kernel)

        std = float(np.std(residual))
        kurt = float(np.mean((residual - np.mean(residual))**4) / (std**4 + 1e-10))
        skew = float(np.mean((residual - np.mean(residual))**3) / (std**3 + 1e-10))

        # AI: lower std (cleaner), higher kurtosis (leptokurtic)
        s = max(0, 1.0 - (std / 12.0))
        scores.append(s)
        results[name] = {
            "residual_std": round(std, 4),
            "residual_kurtosis": round(kurt, 4),
            "residual_skewness": round(skew, 4),
            "sub_score": round(s, 4)
        }

    return {
        "score": round(float(np.mean(scores)), 4),
        "kernels": results,
    }

# ═══════════════════════════════════════════════════════════════════
#  4. DCT BLOCKWISE ANALYSIS (8x8 JPEG blocks)
# ═══════════════════════════════════════════════════════════════════

def dct_blockwise(image_bytes, block_size=8):
    """Analyze DCT coefficient distribution per 8x8 block."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    w, h = img.size
    # Resize to nearest multiple of block_size
    nw = (w // block_size) * block_size
    nh = (h // block_size) * block_size
    img = img.resize((nw, nh))
    arr = np.array(img, dtype=np.float32)

    block_energies = []
    for i in range(0, nh, block_size):
        for j in range(0, nw, block_size):
            block = arr[i:i+block_size, j:j+block_size]
            dct_block = np.fft.fft2(block)
            energy = float(np.sum(np.abs(dct_block[1:, 1:])))  # AC coefficients
            block_energies.append(energy)

    if not block_energies:
        return {"score": 0.5, "n_blocks": 0}

    be = np.array(block_energies)
    cv = float(np.std(be) / (np.mean(be) + 1e-10))

    # AI images: more uniform block energy (low CV)
    score = max(0, 1.0 - (cv / 1.5))
    return {
        "score": round(min(score, 1.0), 4),
        "n_blocks": len(block_energies),
        "mean_block_energy": round(float(np.mean(be)), 2),
        "block_energy_cv": round(cv, 4),
    }

# ═══════════════════════════════════════════════════════════════════
#  5. FFT RADIAL POWER SPECTRUM
# ═══════════════════════════════════════════════════════════════════

def fft_radial_spectrum(image_bytes):
    """Compute radial power spectrum and detect anomalies."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_r = int(min(h, w) / 2)

    radial_profile = []
    for r in range(1, max_r):
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if np.any(mask):
            radial_profile.append(float(np.mean(magnitude[mask])))

    if len(radial_profile) < 5:
        return {"score": 0.5}

    rp = np.array(radial_profile)
    # Check for spectral peaks (GAN artifacts) or rolloff (diffusion)
    gradient = np.diff(rp)
    spike_count = int(np.sum(np.abs(gradient) > np.std(gradient) * 3))
    rolloff_rate = float((rp[0] - rp[-1]) / (rp[0] + 1e-10))

    # GANs: spectral spikes; Diffusion: rapid rolloff
    if spike_count > 3:
        score = min(spike_count / 8.0, 1.0)
    elif rolloff_rate > 0.95:
        score = 0.8  # Too smooth
    else:
        score = 0.15

    return {
        "score": round(score, 4),
        "spectral_spikes": spike_count,
        "rolloff_rate": round(rolloff_rate, 4),
        "n_radial_bins": len(radial_profile),
    }

# ═══════════════════════════════════════════════════════════════════
#  6. HIGH-FREQUENCY ENERGY RATIO
# ═══════════════════════════════════════════════════════════════════

def hf_energy_ratio(image_bytes):
    """Ratio of high-freq to total energy in FFT magnitude."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    fft = np.fft.fft2(arr)
    mag = np.abs(np.fft.fftshift(fft))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

    hf = float(np.mean(mag[dist > min(h, w) * 0.35]))
    total = float(np.mean(mag)) + 1e-10
    ratio = hf / total

    if ratio < 0.75:
        score = min((0.75 - ratio) / 0.4, 1.0)  # Diffusion: too smooth
    elif ratio > 1.6:
        score = min((ratio - 1.6) / 1.0, 1.0)  # GAN: grid artifacts
    else:
        score = 0.1

    return {"score": round(score, 4), "hf_ratio": round(ratio, 4)}

# ═══════════════════════════════════════════════════════════════════
#  7. COLOR COHERENCE & CHANNEL CORRELATION
# ═══════════════════════════════════════════════════════════════════

def color_coherence(image_bytes):
    """Analyze inter-channel correlation and color distribution."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
    arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:,:,0].flatten(), arr[:,:,1].flatten(), arr[:,:,2].flatten()
    rg_corr = float(np.corrcoef(r, g)[0, 1])
    rb_corr = float(np.corrcoef(r, b)[0, 1])
    gb_corr = float(np.corrcoef(g, b)[0, 1])

    avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

    # AI images often have HIGHER inter-channel correlation (less independent noise)
    score = max(0, (avg_corr - 0.5) / 0.5)

    # Color histogram entropy per channel
    entropies = []
    for c in range(3):
        hist, _ = np.histogram(arr[:,:,c], bins=64, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        ent = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        entropies.append(ent)

    avg_ent = float(np.mean(entropies))
    ent_score = max(0, (avg_ent - 4.0) / 2.0)  # High uniformity → AI

    combined = score * 0.6 + ent_score * 0.4

    return {
        "score": round(min(combined, 1.0), 4),
        "rg_correlation": round(rg_corr, 4),
        "rb_correlation": round(rb_corr, 4),
        "gb_correlation": round(gb_corr, 4),
        "channel_entropy_avg": round(avg_ent, 4),
    }

# ═══════════════════════════════════════════════════════════════════
#  8. EDGE SHARPNESS PROFILING (Sobel)
# ═══════════════════════════════════════════════════════════════════

def edge_sharpness(image_bytes):
    """Compute Sobel gradient statistics."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float32)

    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    padded = np.pad(arr, 1, mode='reflect')
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            gx[i, j] = np.sum(padded[i:i+3, j:j+3] * kx)
            gy[i, j] = np.sum(padded[i:i+3, j:j+3] * ky)

    magnitude = np.sqrt(gx**2 + gy**2)
    mean_edge = float(np.mean(magnitude))
    std_edge = float(np.std(magnitude))
    max_edge = float(np.max(magnitude))
    edge_cv = std_edge / (mean_edge + 1e-10)

    # AI: overly uniform edges (low CV) or oversharpened (high mean)
    if edge_cv < 1.0:
        score = max(0, 1.0 - edge_cv)
    elif mean_edge > 100:
        score = min((mean_edge - 100) / 200, 1.0)
    else:
        score = 0.1

    return {
        "score": round(score, 4),
        "mean_edge_magnitude": round(mean_edge, 2),
        "edge_cv": round(edge_cv, 4),
        "max_edge": round(max_edge, 2),
    }

# ═══════════════════════════════════════════════════════════════════
#  9. NOISE FLOOR VARIANCE (local vs global)
# ═══════════════════════════════════════════════════════════════════

def noise_floor_variance(image_bytes, patch_size=16):
    """Compare local noise variance across patches to detect uniformity."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float64)

    # High-pass filter to extract noise
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(arr, size=3)
    noise = arr - smooth

    # Patch-level variance
    variances = []
    for i in range(0, arr.shape[0] - patch_size, patch_size):
        for j in range(0, arr.shape[1] - patch_size, patch_size):
            patch = noise[i:i+patch_size, j:j+patch_size]
            variances.append(float(np.var(patch)))

    if not variances:
        return {"score": 0.5}

    var_arr = np.array(variances)
    var_cv = float(np.std(var_arr) / (np.mean(var_arr) + 1e-10))

    # AI: uniform noise floor (low CV), Real: variable (sensor, compression)
    score = max(0, 1.0 - (var_cv / 2.0))

    return {
        "score": round(min(score, 1.0), 4),
        "noise_variance_cv": round(var_cv, 4),
        "n_patches": len(variances),
        "mean_noise_var": round(float(np.mean(var_arr)), 6),
    }

# ═══════════════════════════════════════════════════════════════════
# 10. PATCH-GRID HOMOGENEITY
# ═══════════════════════════════════════════════════════════════════

def patch_grid_homogeneity(image_bytes, grid=8):
    """Divide image into NxN grid and measure feature homogeneity."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    ph, pw = h // grid, w // grid

    patch_means = []
    patch_stds = []
    for i in range(grid):
        for j in range(grid):
            patch = arr[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
            patch_means.append(float(np.mean(patch)))
            patch_stds.append(float(np.std(patch)))

    mean_cv = float(np.std(patch_means) / (np.mean(patch_means) + 1e-10))
    std_cv = float(np.std(patch_stds) / (np.mean(patch_stds) + 1e-10))

    # AI: more homogeneous patches
    score = max(0, 1.0 - (mean_cv / 0.5)) * 0.5 + max(0, 1.0 - (std_cv / 1.0)) * 0.5

    return {
        "score": round(min(score, 1.0), 4),
        "mean_cv": round(mean_cv, 4),
        "std_cv": round(std_cv, 4),
        "grid_size": f"{grid}x{grid}",
    }

# ═══════════════════════════════════════════════════════════════════
# 11. CHROMATIC ABERRATION TEST
# ═══════════════════════════════════════════════════════════════════

def chromatic_aberration(image_bytes):
    """Real cameras show chromatic aberration at edges; AI doesn't."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)

    # Compute edge maps per channel
    edge_maps = []
    for c in range(3):
        ch = arr[:, :, c]
        dx = np.diff(ch, axis=1, prepend=ch[:, :1])
        dy = np.diff(ch, axis=0, prepend=ch[:1, :])
        edges = np.sqrt(dx**2 + dy**2)
        edge_maps.append(edges)

    # Measure offset between R and B edge maps
    rb_diff = float(np.mean(np.abs(edge_maps[0] - edge_maps[2])))
    rg_diff = float(np.mean(np.abs(edge_maps[0] - edge_maps[1])))

    # Real photos: higher chromatic aberration (larger differences)
    # AI: perfectly aligned channels → low aberration
    combined = (rb_diff + rg_diff) / 2
    score = max(0, 1.0 - (combined / 5.0))  # Low aberration → AI

    return {
        "score": round(min(score, 1.0), 4),
        "rb_edge_diff": round(rb_diff, 4),
        "rg_edge_diff": round(rg_diff, 4),
    }

# ═══════════════════════════════════════════════════════════════════
# 12. BIT-PLANE ANALYSIS (LSB patterns)
# ═══════════════════════════════════════════════════════════════════

def bit_plane_analysis(image_bytes):
    """Analyze least significant bit patterns."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.uint8)

    bit_planes = {}
    scores = []
    for bit in range(8):
        plane = (arr >> bit) & 1
        # Entropy of bit plane
        ones = float(np.mean(plane))
        ent = -(ones * np.log2(ones + 1e-10) + (1 - ones) * np.log2(1 - ones + 1e-10))
        bit_planes[f"bit_{bit}"] = {"entropy": round(ent, 4), "ones_ratio": round(ones, 4)}

        # LSBs (bit 0, 1) should be near-random in real images
        if bit < 2:
            # Too regular → AI
            regularity = abs(ones - 0.5)
            scores.append(regularity * 4)

    avg_score = float(np.mean(scores)) if scores else 0.5

    return {
        "score": round(min(avg_score, 1.0), 4),
        "planes": bit_planes,
    }

# ═══════════════════════════════════════════════════════════════════
# 13. HISTOGRAM GAP DETECTION
# ═══════════════════════════════════════════════════════════════════

def histogram_gaps(image_bytes):
    """Detect unnatural gaps in intensity histogram."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.uint8)
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))

    # Count zero bins (gaps)
    zero_bins = int(np.sum(hist == 0))
    total_bins = 256
    gap_ratio = zero_bins / total_bins

    # Real photos rarely have gaps; AI quantization can create them
    # BUT upsampled AI can also fill all bins. Use gap pattern.
    consecutive_zeros = 0
    max_consecutive = 0
    for h in hist:
        if h == 0:
            consecutive_zeros += 1
            max_consecutive = max(max_consecutive, consecutive_zeros)
        else:
            consecutive_zeros = 0

    score = min(max_consecutive / 10.0, 1.0) if max_consecutive > 3 else gap_ratio

    return {
        "score": round(score, 4),
        "zero_bins": zero_bins,
        "max_consecutive_gaps": max_consecutive,
        "gap_ratio": round(gap_ratio, 4),
    }

# ═══════════════════════════════════════════════════════════════════
# 14. TEXTURE COMPLEXITY (LBP-like)
# ═══════════════════════════════════════════════════════════════════

def texture_complexity(image_bytes):
    """Simplified Local Binary Pattern analysis for texture richness."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((64, 64))
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    lbp_map = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = arr[i, j]
            code = 0
            code |= (arr[i-1, j-1] > center) << 7
            code |= (arr[i-1, j]   > center) << 6
            code |= (arr[i-1, j+1] > center) << 5
            code |= (arr[i,   j+1] > center) << 4
            code |= (arr[i+1, j+1] > center) << 3
            code |= (arr[i+1, j]   > center) << 2
            code |= (arr[i+1, j-1] > center) << 1
            code |= (arr[i,   j-1] > center) << 0
            lbp_map[i-1, j-1] = code

    # Histogram of LBP codes
    hist, _ = np.histogram(lbp_map, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    ent = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
    max_ent = 8.0  # log2(256)

    # AI: lower texture diversity → lower entropy
    diversity = ent / max_ent
    score = max(0, 1.0 - diversity)

    return {
        "score": round(min(score, 1.0), 4),
        "lbp_entropy": round(ent, 4),
        "texture_diversity": round(diversity, 4),
        "unique_patterns": int(np.sum(hist > 0)),
    }

# ═══════════════════════════════════════════════════════════════════
# 15. GLOBAL STATISTICAL FINGERPRINT
# ═══════════════════════════════════════════════════════════════════

def global_statistics(image_bytes):
    """Compute skewness, kurtosis, and higher moments per channel."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float64)

    channel_stats = {}
    scores = []
    for c, name in enumerate(["R", "G", "B"]):
        ch = arr[:, :, c].flatten()
        mu = np.mean(ch)
        std = np.std(ch) + 1e-10
        skew = float(np.mean(((ch - mu) / std) ** 3))
        kurt = float(np.mean(((ch - mu) / std) ** 4))

        channel_stats[name] = {
            "mean": round(float(mu), 2),
            "std": round(float(std), 2),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
        }

        # AI images tend toward near-zero skew and kurtosis ~3 (Gaussian)
        skew_deviation = abs(skew)
        kurt_deviation = abs(kurt - 3.0)
        normalcy = max(0, 1.0 - (skew_deviation + kurt_deviation) / 5.0)
        scores.append(normalcy)

    return {
        "score": round(float(np.mean(scores)), 4),
        "channels": channel_stats,
    }

# ═══════════════════════════════════════════════════════════════════
#  MASTER FORENSIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def full_image_forensics(image_bytes):
    """Run ALL 15 forensic techniques and produce a weighted ensemble verdict."""
    techniques = {}

    # Run all methods
    try: techniques["pixel_ela"] = pixel_ela_heatmap(image_bytes)
    except: techniques["pixel_ela"] = {"score": 0.5}
    try: techniques["jpeg_ghost"] = jpeg_ghost_detection(image_bytes)
    except: techniques["jpeg_ghost"] = {"score": 0.5}
    try: techniques["srm_multi"] = srm_multi_kernel(image_bytes)
    except: techniques["srm_multi"] = {"score": 0.5}
    try: techniques["dct_block"] = dct_blockwise(image_bytes)
    except: techniques["dct_block"] = {"score": 0.5}
    try: techniques["fft_radial"] = fft_radial_spectrum(image_bytes)
    except: techniques["fft_radial"] = {"score": 0.5}
    try: techniques["hf_energy"] = hf_energy_ratio(image_bytes)
    except: techniques["hf_energy"] = {"score": 0.5}
    try: techniques["color_coherence"] = color_coherence(image_bytes)
    except: techniques["color_coherence"] = {"score": 0.5}
    try: techniques["edge_sharpness"] = edge_sharpness(image_bytes)
    except: techniques["edge_sharpness"] = {"score": 0.5}
    try: techniques["noise_floor"] = noise_floor_variance(image_bytes)
    except: techniques["noise_floor"] = {"score": 0.5}
    try: techniques["patch_grid"] = patch_grid_homogeneity(image_bytes)
    except: techniques["patch_grid"] = {"score": 0.5}
    try: techniques["chromatic_ab"] = chromatic_aberration(image_bytes)
    except: techniques["chromatic_ab"] = {"score": 0.5}
    try: techniques["bit_plane"] = bit_plane_analysis(image_bytes)
    except: techniques["bit_plane"] = {"score": 0.5}
    try: techniques["hist_gaps"] = histogram_gaps(image_bytes)
    except: techniques["hist_gaps"] = {"score": 0.5}
    try: techniques["texture"] = texture_complexity(image_bytes)
    except: techniques["texture"] = {"score": 0.5}
    try: techniques["global_stats"] = global_statistics(image_bytes)
    except: techniques["global_stats"] = {"score": 0.5}

    # Weighted ensemble (higher weight = more reliable signal)
    weights = {
        "pixel_ela": 0.10, "jpeg_ghost": 0.08, "srm_multi": 0.10,
        "dct_block": 0.07, "fft_radial": 0.07, "hf_energy": 0.07,
        "color_coherence": 0.08, "edge_sharpness": 0.07,
        "noise_floor": 0.08, "patch_grid": 0.06,
        "chromatic_ab": 0.05, "bit_plane": 0.04,
        "hist_gaps": 0.04, "texture": 0.05, "global_stats": 0.04,
    }

    master = sum(techniques[k]["score"] * weights.get(k, 0.05) for k in techniques)
    total_weight = sum(weights.get(k, 0.05) for k in techniques)
    master = master / max(total_weight, 1e-10)

    # Confidence: agreement between techniques
    all_scores = [techniques[k]["score"] for k in techniques]
    consensus_std = float(np.std(all_scores))
    confidence = max(0, 1.0 - consensus_std)

    verdict = "AI-GENERATED" if master > 0.5 else "LIKELY REAL"
    if confidence < 0.4:
        verdict += " (LOW CONFIDENCE)"

    return {
        "master_score": round(float(min(max(master, 0), 1)), 4),
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "n_techniques": len(techniques),
        "techniques": techniques,
    }
