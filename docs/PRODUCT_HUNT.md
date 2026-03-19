# Product Hunt Launch Kit

This document contains everything needed to launch AGD on Product Hunt.

## The Hook
**Tagline**: Orthogonal Multi-Modal AI Detector (Text, Image, Audio, Video)
**Description**: AGD is the ultimate open-source forensic framework. Instead of a black-box probability, it runs 27 independent detection techniques (including Spatial Frequency and Zero-Crossing Rate) across Text, Image, Audio, and Video to give you an interpretable anomaly map.

## Maker's Comment
Hi Hunters! 👋 Ardelyo here.

We built Artificial General Detector (AGD) because we were tired of "black box" AI detectors that just output "95% AI" with zero explanation. 

AGD is different. It’s an open-source, multi-modal forensic engine. When you upload a suspicious video, image, audio clip, or block of text, AGD runs up to 27 independent hardware-level verification heuristics and frequency-domain transforms. It doesn't just guess; it isolates exactly *why* something is synthetic (e.g., JPEG ghosting artifacts, MFCC uniform variance, or spectral flux anomalies).

**What’s under the hood?**
*   **Image**: 15 Techniques ranging from Pixel-Level ELA to SRM Multi-Kernel.
*   **Text**: 12 Techniques from Sliding-Window Entropy to Positional Token Variance.
*   **Audio**: MFCC, LFCC, Spectral Flux, and Zero-Crossing Rate for TTS artifacts.
*   **Video**: Temporal Frame Difference Energy and Motion Consistency (optical flow proxies).

We’d love for you to try breaking it! Upload some deepfakes, ChatGPT essays, or cloned voices and see exactly what signals give them away. Let me know your feedback below!

## Assets Checklist
- [x] **Icon**: A sharp, high-contrast logo `docs/agd_logo.png`
- [x] **Gallery Images**: Screenshots of the new Surveillance Aesthetic UI `ResultsDashboard`.
- [x] **Demo Video**: A 30s screen recording showing an audio or video file upload resulting in the HUD displaying the anomaly report.
- [ ] **First Comment**: Ready (see above).
- [ ] **Makers Tagged**: `@Ardelyo` / `@OurCreativity`

## Social Links
*   **Website / Repo**: [GitHub Link]
*   **Twitter/X**: [...]
