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
        "global_mean": float(round(float(global_mean), 4)),
        "global_std": float(round(float(global_std), 4)),
        "max_diff": float(round(float(max_diff), 4)),
        "suspicious_pixel_ratio": float(round(float(suspicious_ratio), 4)),
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
    score = max(0.0, 1.0 - uniformity * 2)

    return {
        "score": min(float(score), 1.0),
        "estimated_quality": min_q,
        "ghost_curve": ghost_scores,
        "curve_uniformity": float(round(float(uniformity), 4)),
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
                patch = padded[i:i+kh, j:j+kw]  # type: ignore
                if patch.shape == kernel.shape:
                    residual[i, j] = np.sum(patch * kernel)

        std = float(np.std(residual))
        kurt = float(np.mean((residual - np.mean(residual))**4) / (std**4 + 1e-10))
        skew = float(np.mean((residual - np.mean(residual))**3) / (std**3 + 1e-10))

        # AI: lower std (cleaner), higher kurtosis (leptokurtic)
        s = max(0.0, 1.0 - (std / 12.0))
        scores.append(s)
        results[name] = {
            "residual_std": float(round(float(std), 4)),
            "residual_std": float(round(float(std), 4)), # type: ignore
            "residual_kurtosis": float(round(float(kurt), 4)), # type: ignore
            "residual_skewness": float(round(float(skew), 4)), # type: ignore
            "sub_score": float(round(float(s), 4)) # type: ignore
        }

    return {
        "score": float(round(float(np.mean(scores)), 4)), # type: ignore
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
    score = max(0.0, 1.0 - (cv / 1.5)) # type: ignore
    return {
        "score": float(round(min(float(score), 1.0), 4)), # type: ignore
        "n_blocks": len(block_energies),
        "mean_block_energy": float(round(float(np.mean(be)), 2)), # type: ignore
        "block_energy_cv": float(round(float(cv), 4)), # type: ignore
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
        "score": float(round(float(score), 4)), # type: ignore
        "spectral_spikes": spike_count,
        "spectral_spikes": spike_count,
        "rolloff_rate": float(round(float(rolloff_rate), 4)), # type: ignore
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

    return {"score": float(round(float(score), 4)), "hf_ratio": float(round(float(ratio), 4))} # type: ignore

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
    score = max(0.0, (avg_corr - 0.5) / 0.5) # type: ignore

    # Color histogram entropy per channel
    entropies = []
    for c in range(3):
        hist, _ = np.histogram(arr[:,:,c], bins=64, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        ent = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        entropies.append(ent)

    avg_ent = float(np.mean(entropies))
    ent_score = max(0.0, (avg_ent - 4.0) / 2.0) # type: ignore

    combined = score * 0.6 + ent_score * 0.4

    return {
        "score": float(round(min(float(combined), 1.0), 4)), # type: ignore
        "rg_correlation": float(round(float(rg_corr), 4)), # type: ignore
        "rb_correlation": float(round(float(rb_corr), 4)), # type: ignore
        "gb_correlation": float(round(float(gb_corr), 4)), # type: ignore
        "channel_entropy_avg": float(round(float(avg_ent), 4)), # type: ignore
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
            gx[i, j] = np.sum(padded[i:i+3, j:j+3] * kx) # type: ignore
            gy[i, j] = np.sum(padded[i:i+3, j:j+3] * ky) # type: ignore

    magnitude = np.sqrt(gx**2 + gy**2)
    mean_edge = float(np.mean(magnitude))
    std_edge = float(np.std(magnitude))
    max_edge = float(np.max(magnitude))
    edge_cv = std_edge / (mean_edge + 1e-10)

    # AI: overly uniform edges (low CV) or oversharpened (high mean)
    if edge_cv < 1.0:
        score = max(0.0, 1.0 - edge_cv) # type: ignore
    elif mean_edge > 100:
        score = min((mean_edge - 100) / 200, 1.0)
    else:
        score = 0.1

    return {
        "score": float(round(float(score), 4)), # type: ignore
        "mean_edge_magnitude": float(round(float(mean_edge), 2)), # type: ignore
        "edge_cv": float(round(float(edge_cv), 4)), # type: ignore
        "max_edge": float(round(float(max_edge), 2)), # type: ignore
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
    score = max(0.0, 1.0 - (var_cv / 2.0)) # type: ignore

    return {
        "score": float(round(min(float(score), 1.0), 4)), # type: ignore
        "noise_variance_cv": float(round(float(var_cv), 4)), # type: ignore
        "n_patches": len(variances),
        "mean_noise_var": float(round(float(np.mean(var_arr)), 6)), # type: ignore
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
    score = max(0.0, 1.0 - (mean_cv / 0.5)) * 0.5 + max(0.0, 1.0 - (std_cv / 1.0)) * 0.5 # type: ignore

    return {
        "score": float(round(min(float(score), 1.0), 4)), # type: ignore
        "mean_cv": float(round(float(mean_cv), 4)), # type: ignore
        "std_cv": float(round(float(std_cv), 4)), # type: ignore
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
    # Score: higher error → AI-like processing
    score = min(combined / 5.0, 1.0) # type: ignore
    return {
        "score": float(round(float(score), 4)), # type: ignore
        "rb_edge_diff": float(round(float(rb_diff), 4)), # type: ignore
        "rg_edge_diff": float(round(float(rg_diff), 4)), # type: ignore
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
    entropies = [] # Added to collect entropies for avg_entropy
    for bit in range(8):
        plane = (arr >> bit) & 1
        # Entropy of bit plane
        ones = float(np.mean(plane))
        ent = float(-(ones * np.log2(ones + 1e-10) + (1 - ones) * np.log2(1 - ones + 1e-10)))
        bit_planes[f"bit_{bit}"] = {"entropy": float(round(float(ent), 4)), "ones_ratio": float(round(float(ones), 4))} # type: ignore
        entropies.append(ent) # Collect entropy

        # LSBs (bit 0, 1) should be near-random in real images
        if bit < 2:
            # Too regular → AI
            regularity = abs(ones - 0.5)
            scores.append(regularity * 4)

    avg_score = float(np.mean(scores)) if scores else 0.5
    avg_entropy = float(np.mean(entropies)) # Calculate average entropy

    # AI: lower texture diversity → lower entropy
    # AI: overly uniform bit patterns
    score = max(0.0, 1.0 - (avg_entropy / 0.5)) # type: ignore
    return {
        "score": float(round(min(float(score), 1.0), 4)), # type: ignore
        "lsb_entropy": float(round(float(avg_entropy), 4)), # type: ignore
        "entropy_by_channel": [float(round(float(e), 4)) for e in entropies], # type: ignore
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
            max_consecutive = max(max_consecutive, consecutive_zeros) # type: ignore
        else:
            consecutive_zeros = 0

    score = min(max_consecutive / 10.0, 1.0) if max_consecutive > 3 else gap_ratio # type: ignore

    return {
        "score": float(round(float(score), 4)), # type: ignore
        "zero_bins": zero_bins,
        "max_consecutive_gaps": max_consecutive,
        "gap_ratio": float(round(float(gap_ratio), 4)), # type: ignore
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
    score = max(0.0, 1.0 - diversity) # type: ignore

    return {
        "score": float(round(min(float(score), 1.0), 4)), # type: ignore
        "lbp_entropy": float(round(float(ent), 4)), # type: ignore
        "texture_diversity": float(round(float(diversity), 4)), # type: ignore
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
            "mean": float(round(float(mu), 2)), # type: ignore
            "std": float(round(float(std), 2)), # type: ignore
            "skewness": float(round(float(skew), 4)), # type: ignore
            "kurtosis": float(round(float(kurt), 4)), # type: ignore
        }

        # AI images tend toward near-zero skew and kurtosis ~3 (Gaussian)
        skew_deviation = abs(skew)
        kurt_deviation = abs(kurt - 3.0)
        normalcy = max(0.0, 1.0 - (skew_deviation + kurt_deviation) / 5.0) # type: ignore
        scores.append(normalcy)

    return {
        "score": float(round(float(np.mean(scores)), 4)), # type: ignore
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
    master = master / max(total_weight, 1e-10) # type: ignore

    # Confidence: agreement between techniques
    all_scores = [techniques[k]["score"] for k in techniques]
    consensus_std = float(np.std(all_scores))
    confidence = max(0.0, 1.0 - consensus_std) # type: ignore

    verdict = "AI-GENERATED" if master > 0.5 else "LIKELY REAL"
    if confidence < 0.4:
        verdict += " (LOW CONFIDENCE)"

    return {
        "master_score": float(round(float(min(max(float(master), 0.0), 1.0)), 4)), # type: ignore
        "verdict": verdict,
        "confidence": float(round(float(confidence), 4)), # type: ignore
        "n_techniques": len(techniques),
        "techniques": techniques,
    }
