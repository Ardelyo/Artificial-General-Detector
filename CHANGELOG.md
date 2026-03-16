# Changelog

All notable changes to AGD will be documented in this file.

## [2.0.0] — 2026-03-16

### Added
- **Ultra-Deep Image Forensics Engine** (`deep_image_forensics.py`) with 15 techniques:
  Pixel ELA, JPEG Ghost, SRM 5-kernel, DCT Blockwise, FFT Radial, HF Energy, Color Coherence,
  Edge Sharpness, Noise Floor, Patch Grid, Chromatic Aberration, Bit-Plane, Histogram Gaps,
  Texture LBP, Global Stats.
- **Ultra-Deep Text Forensics Engine** (`deep_text_forensics.py`) with 12 techniques:
  Token Perplexity, Sentence RoBERTa, Sliding Entropy, Burstiness, N-gram Heatmap,
  Vocab Fingerprint, Stylometrics, Perplexity Proxy, Positional Entropy, Function Words,
  Transition Density, Sentence Starter Diversity.
- Official Research Paper generator (`generate_research_paper.py`).
- AGD Logo.
- Ultra-Deep Benchmark (`ultra_benchmark.py`).

### Changed
- `agd-text`: Now uses RoBERTa-base-openai-detector + 4-heuristic ensemble.
- `agd-image`: Now uses ELA + FFT + SRM + Color Uniformity ensemble.
- `agd-audio`: Rewritten with MFCC, LFCC, Spectral Flux, ZCR.
- `agd-video`: Enhanced with temporal frame diff, motion consistency, frequency consistency.

## [1.0.0] — 2026-03-15

### Added
- Initial release with core modules (text, image, audio, video).
- Orchestrator API Gateway.
- Basic ELA and Burstiness detection.
- Benchmark framework and DOCX report generation.

---
*Maintained by OurCreativity (Admin: Ardelyo)*
