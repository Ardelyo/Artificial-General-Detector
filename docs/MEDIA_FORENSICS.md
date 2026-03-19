# AGD Media Forensics: Theoretical and State-of-the-Art Alignments

While much of the open-source detection industry focuses exclusively on LLM text or simple Image CNNs, AGD treats **Audio** and **Video** as equally critical orthogonal modalities.

This document outlines how the `agd-audio` and `agd-video` heuristic engines benchmark against current State-of-the-Art (SOTA) research.

---

## 1. Audio Deepfakes: Overcoming the ASVspoof 2021 Benchmark

The *Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2021)* heavily informs our approach to Text-To-Speech (TTS) and Voice Conversion (VC) attacks.

**SOTA Context:**
- During ASVspoof 2021, the new Deepfake (DF) task was established to catch compressed speech found online.
- SOTA baseline results for the best countermeasure models currently sit at an **Equal Error Rate (EER) of ~1.43%** (Wizwand, 2021).
- However, pure deep-learning audio detectors often suffer generalization failures across unknown codecs.

**The AGD Audio Approach:**
Instead of a black-box convolutional model over Mel-spectrograms, AGD relies on highly interpretable, codec-resilient physical signals:
*   **LFCC / MFCC Variance**: AI audio often presents unnaturally low variance in its highest cepstral coefficients due to vocoder spectral approximations. AGD scores this stability explicitly.
*   **Spectral Flux & Zero-Crossing Rates**: Perfect waveforms derived from synthetic generators have unusually consistent Spectral Flux. Humans introduce microscopic temporal variance and breath noise, which AGD tracks.

---

## 2. Video Deepfakes: Solving the FaceForensics++ Compression Gap

*FaceForensics++ (FF++)* remains the gold standard for video manipulation datasets, containing 1000 original videos tampered with Deepfakes, Face2Face, FaceSwap, and NeuralTextures.

**SOTA Context:**
- Models like XceptionNet achieve over **95% AUC (Area Under the Curve)** on untouched FaceForensics++ data.
- **The Compression Gap:** This performance drops aggressively to **~70%** under standard c23 compression (e.g., YouTube compression layers). Temporal artifacts are destroyed by standard video compression routines.

**The AGD Video Approach:**
AGD fights the compression gap by ignoring superficial spatial textures (which c23 compression destroys) and evaluating macro-temporal consistency:
*   **Temporal Difference Energy ($\Delta F_t$)**: Deepfakes frequently stitch frames improperly resulting in extreme localized differences (Micro-stutters).
*   **Optical Flow Approximation (Motion Consistency)**: Using simplified spatial gradients, AGD evaluates object motion. FaceSwap deepfakes often cause the replacement patch to slide independently of the subject's skull velocity map.
*   **Spatial Frequency & Color Shift**: Color histograms often snap unexpectedly across generative AI clips due to inconsistent lighting normalization between generation batches. AGD logs the chi-squared difference across frames to catch these generative lighting desyncs.
