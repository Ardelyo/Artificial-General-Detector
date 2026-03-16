"""
AGD Text Module — SOTA Upgrade
================================
Methods:
1. RoBERTa-based OpenAI Detector (HuggingFace pipeline)
2. Burstiness (Sentence CV)
3. Shannon Entropy (Vocabulary diversity)
4. N-gram repetition
5. Ensemble of all above
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import string
import os

app = FastAPI(title="AGD Text Module — SOTA")

class TextRequest(BaseModel):
    text: str

class TextAnalysisResult(BaseModel):
    score: float
    breakdown: dict

# ── Lazy-loaded transformer pipeline ───────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            _pipeline = pipeline(
                "text-classification",
                model="roberta-base-openai-detector",
                truncation=True,
                max_length=512
            )
            print("[agd-text] Loaded roberta-base-openai-detector")
        except Exception as e:
            print(f"[agd-text] Transformer load failed: {e}")
            _pipeline = "FAILED"
    return _pipeline

# ── Heuristic methods ──────────────────────────────────────────────

def compute_entropy(text: str) -> float:
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    if not words: return 0.0
    _, counts = np.unique(words, return_counts=True)
    probs = counts / len(words)
    return float(-np.sum(probs * np.log2(probs)))

def compute_burstiness(text: str) -> float:
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if len(sentences) < 3: return 0.5
    lengths = [len(s.split()) for s in sentences]
    return float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0.0

def compute_ngram_repetition(text: str, n: int = 2) -> float:
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    if len(words) < n + 1: return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]  # type: ignore
    return 1.0 - len(set(ngrams)) / max(len(ngrams), 1)

def compute_perplexity_proxy(text: str) -> float:
    """Perplexity proxy via word frequency rank distribution."""
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    if len(words) < 10: return 0.5
    _, counts = np.unique(words, return_counts=True)
    sorted_counts = np.sort(counts)[::-1]
    if len(sorted_counts) < 2: return 0.5
    zipf_deviation = float(np.std(sorted_counts / np.arange(1, len(sorted_counts) + 1)))
    return float(min(max(zipf_deviation / 5.0, 0.0), 1.0))

def roberta_score(text: str) -> float:
    pipe = get_pipeline()
    if pipe == "FAILED" or pipe is None:
        return -1.0
    try:
        result = pipe(text[:512])  # type: ignore
        for item in result:
            if item["label"] == "LABEL_0":  # Fake
                return float(item["score"])
            elif item["label"] == "Fake":
                return float(item["score"])
        # Fallback: if "Real" label, invert
        for item in result:
            if item["label"] in ("LABEL_1", "Real"):
                return float(1.0 - item["score"])
        return 0.5
    except Exception as e:
        print(f"[agd-text] RoBERTa error: {e}")
        return -1.0

# ── API ────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "ok", "module": "agd-text", "version": "2.0-SOTA"}

@app.post("/analyze", response_model=TextAnalysisResult)
async def analyze_text(payload: TextRequest):
    text = payload.text
    words = text.split()
    if len(words) < 20:
        return {"score": 0.5, "breakdown": {"warning": "Text too short for reliable analysis"}}

    # Compute all signals
    cv = compute_burstiness(text)
    entropy = compute_entropy(text)
    ngram_rep = compute_ngram_repetition(text, n=2)
    perp_proxy = compute_perplexity_proxy(text)
    roberta = roberta_score(text)

    # Heuristic scores (normalized to 0-1 AI probability)
    cv_score = float(max(0.0, 1.0 - (cv / 0.6)))
    entropy_score = float(max(0.0, 1.0 - (entropy / 9.0)))
    ngram_score = float(min(max(ngram_rep * 2.5, 0.0), 1.0))

    # Ensemble
    if roberta >= 0:
        # Weighted: RoBERTa dominates (60%), heuristics fill gaps
        master = (roberta * 0.60 +
                  cv_score * 0.10 +
                  entropy_score * 0.10 +
                  ngram_score * 0.10 +
                  perp_proxy * 0.10)
    else:
        # Fallback: heuristics only
        master = (cv_score * 0.30 +
                  entropy_score * 0.30 +
                  ngram_score * 0.25 +
                  perp_proxy * 0.15)

    master = float(min(max(master, 0.0), 1.0))

    return {
        "score": round(master, 4),  # type: ignore
        "breakdown": {
            "roberta_openai_detector": round(roberta, 4) if roberta >= 0 else "unavailable",  # type: ignore
            "burstiness_cv": round(cv_score, 4),  # type: ignore
            "entropy": round(entropy_score, 4),  # type: ignore
            "ngram_repetition": round(ngram_score, 4),  # type: ignore
            "perplexity_proxy": round(perp_proxy, 4),  # type: ignore
            "raw_cv": round(cv, 4),  # type: ignore
            "raw_entropy": round(entropy, 4),  # type: ignore
        }
    }
