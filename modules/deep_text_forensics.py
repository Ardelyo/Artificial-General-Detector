"""
AGD Ultra-Deep Text Forensics Engine
========================================
Performs token-level, sentence-level, and document-level analysis
using 12+ independent forensic techniques.

Techniques:
 1. Token-by-Token Perplexity (RoBERTa log-probability per token)
 2. Sentence-Level RoBERTa Scoring (per-sentence AI probability)
 3. Sliding-Window Entropy (10-word windows)
 4. Burstiness Profile (sentence-length distribution)
 5. N-gram Repetition Heatmap (bi/tri/quad-gram)
 6. Vocabulary Fingerprinting (Zipf's law deviation)
 7. Stylometric Features (avg word length, punctuation density, etc.)
 8. Perplexity Proxy (word frequency rank distribution)
 9. Positional Entropy (entropy variation across document thirds)
10. Function Word Distribution (the, is, was, etc.)
11. Conjunction & Transition Density
12. Sentence Starter Diversity
"""

import numpy as np
import string
import re

# ── Lazy RoBERTa pipeline ─────────────────────────────────────────

_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline
            _pipe = pipeline("text-classification",
                             model="roberta-base-openai-detector",
                             truncation=True, max_length=512)
            print("[Deep-Text] Loaded RoBERTa")
        except Exception as e:
            print(f"[Deep-Text] RoBERTa failed: {e}")
            _pipe = "FAILED"
    return _pipe

def _roberta_score(text):
    pipe = _get_pipe()
    if pipe == "FAILED" or pipe is None:
        return -1.0
    try:
        result = pipe(text[:512])
        for item in result:
            if item["label"] in ("LABEL_0", "Fake"):
                return float(item["score"])
            elif item["label"] in ("LABEL_1", "Real"):
                return float(1.0 - item["score"])
        return 0.5
    except:
        return -1.0

# ── Helper ────────────────────────────────────────────────────────

def _tokenize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

def _sentences(text):
    sents = re.split(r'[.!?]+', text)
    return [s.strip() for s in sents if len(s.strip()) > 5]

# ═══════════════════════════════════════════════════════════════════
#  1. TOKEN-BY-TOKEN PERPLEXITY (RoBERTa-based)
# ═══════════════════════════════════════════════════════════════════

def token_perplexity_profile(text):
    """Score consecutive 50-token chunks with RoBERTa."""
    words = text.split()
    chunk_size = 50
    scores = []
    for i in range(0, len(words) - chunk_size + 1, chunk_size // 2):
        chunk = " ".join(words[i:i + chunk_size])
        s = _roberta_score(chunk)
        if s >= 0:
            scores.append(s)

    if not scores:
        return {"score": 0.5, "n_chunks": 0}

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    max_s = float(np.max(scores))
    min_s = float(np.min(scores))

    return {
        "score": round(mean_s, 4),
        "n_chunks": len(scores),
        "max_chunk_score": round(max_s, 4),
        "min_chunk_score": round(min_s, 4),
        "score_std": round(std_s, 4),
        "chunks": [round(s, 4) for s in scores[:20]],  # First 20 for brevity
    }

# ═══════════════════════════════════════════════════════════════════
#  2. SENTENCE-LEVEL ROBERTA SCORING
# ═══════════════════════════════════════════════════════════════════

def sentence_level_scoring(text):
    """Score each sentence individually with RoBERTa."""
    sents = _sentences(text)
    if len(sents) < 3:
        return {"score": 0.5, "n_sentences": len(sents)}

    scores = []
    for s in sents[:30]:  # Cap at 30 sentences for speed
        score = _roberta_score(s)
        if score >= 0:
            scores.append({"sentence": s[:80], "ai_prob": round(score, 4)})

    if not scores:
        return {"score": 0.5}

    avg = float(np.mean([s["ai_prob"] for s in scores]))
    high_ai_count = sum(1 for s in scores if s["ai_prob"] > 0.7)
    high_ai_ratio = high_ai_count / len(scores)

    return {
        "score": round(avg, 4),
        "n_scored": len(scores),
        "high_ai_ratio": round(high_ai_ratio, 4),
        "sentences": scores[:10],  # Show top-10
    }

# ═══════════════════════════════════════════════════════════════════
#  3. SLIDING-WINDOW ENTROPY
# ═══════════════════════════════════════════════════════════════════

def sliding_window_entropy(text, window=10, stride=5):
    """Compute Shannon entropy in sliding windows of N words."""
    words = _tokenize(text)
    if len(words) < window * 2:
        return {"score": 0.5}

    entropies = []
    for i in range(0, len(words) - window, stride):
        chunk = words[i:i + window]
        _, counts = np.unique(chunk, return_counts=True)
        probs = counts / len(chunk)
        ent = float(-np.sum(probs * np.log2(probs)))
        entropies.append(ent)

    ent_arr = np.array(entropies)
    mean_ent = float(np.mean(ent_arr))
    std_ent = float(np.std(ent_arr))
    cv_ent = std_ent / (mean_ent + 1e-10)

    # AI: uniform entropy (low CV), Human: variable
    score = max(0, 1.0 - (cv_ent / 0.8))

    return {
        "score": round(min(score, 1.0), 4),
        "mean_entropy": round(mean_ent, 4),
        "entropy_cv": round(cv_ent, 4),
        "n_windows": len(entropies),
    }

# ═══════════════════════════════════════════════════════════════════
#  4. BURSTINESS PROFILE
# ═══════════════════════════════════════════════════════════════════

def burstiness_profile(text):
    """Detailed sentence-length distribution analysis."""
    sents = _sentences(text)
    if len(sents) < 3:
        return {"score": 0.5}

    lengths = [len(s.split()) for s in sents]
    la = np.array(lengths, dtype=float)

    cv = float(np.std(la) / (np.mean(la) + 1e-10))
    skew = float(np.mean(((la - np.mean(la)) / (np.std(la) + 1e-10)) ** 3))
    kurt = float(np.mean(((la - np.mean(la)) / (np.std(la) + 1e-10)) ** 4))

    # AI: low CV (uniform), near-zero skew, kurt ~3
    score = max(0, 1.0 - (cv / 0.6))

    return {
        "score": round(min(score, 1.0), 4),
        "cv": round(cv, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n_sentences": len(sents),
        "mean_length": round(float(np.mean(la)), 1),
        "min_length": int(np.min(la)),
        "max_length": int(np.max(la)),
    }

# ═══════════════════════════════════════════════════════════════════
#  5. N-GRAM REPETITION HEATMAP
# ═══════════════════════════════════════════════════════════════════

def ngram_repetition_heatmap(text):
    """Measure repetition at bigram, trigram, and quadgram levels."""
    words = _tokenize(text)
    if len(words) < 10:
        return {"score": 0.5}

    results = {}
    scores = []
    for n in [2, 3, 4]:
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        unique = len(set(ngrams))
        total = len(ngrams)
        rep_ratio = 1.0 - unique / total
        results[f"{n}gram"] = {
            "unique": unique, "total": total,
            "repetition_ratio": round(rep_ratio, 4)
        }
        scores.append(min(rep_ratio * 3, 1.0))

    avg = float(np.mean(scores)) if scores else 0.5

    return {
        "score": round(avg, 4),
        "grams": results,
    }

# ═══════════════════════════════════════════════════════════════════
#  6. VOCABULARY FINGERPRINTING (Zipf's Law)
# ═══════════════════════════════════════════════════════════════════

def vocabulary_fingerprint(text):
    """Check deviation from Zipf's law in word frequency distribution."""
    words = _tokenize(text)
    if len(words) < 20:
        return {"score": 0.5}

    _, counts = np.unique(words, return_counts=True)
    sorted_counts = np.sort(counts)[::-1].astype(float)

    if len(sorted_counts) < 5:
        return {"score": 0.5}

    # Zipf's law: frequency ~ 1/rank
    ranks = np.arange(1, len(sorted_counts) + 1).astype(float)
    expected = sorted_counts[0] / ranks

    # Mean absolute deviation from Zipf
    deviation = float(np.mean(np.abs(sorted_counts - expected) / (expected + 1e-10)))

    # Type-Token Ratio
    ttr = len(set(words)) / len(words)

    # AI: lower TTR (less diverse), closer to Zipf (more predictable)
    ttr_score = max(0, 1.0 - (ttr / 0.7))
    zipf_score = max(0, 1.0 - (deviation / 2.0))
    combined = ttr_score * 0.5 + zipf_score * 0.5

    return {
        "score": round(min(combined, 1.0), 4),
        "type_token_ratio": round(ttr, 4),
        "zipf_deviation": round(deviation, 4),
        "vocab_size": len(set(words)),
        "total_words": len(words),
    }

# ═══════════════════════════════════════════════════════════════════
#  7. STYLOMETRIC FEATURES
# ═══════════════════════════════════════════════════════════════════

def stylometric_features(text):
    """Extract readability and stylistic metrics."""
    words = text.split()
    chars = list(text)
    sents = _sentences(text)

    if not words or not sents:
        return {"score": 0.5}

    avg_word_len = float(np.mean([len(w) for w in words]))
    avg_sent_len = float(np.mean([len(s.split()) for s in sents]))

    # Punctuation density
    punct_count = sum(1 for c in chars if c in string.punctuation)
    punct_density = punct_count / max(len(chars), 1)

    # Uppercase ratio
    upper_count = sum(1 for c in chars if c.isupper())
    upper_ratio = upper_count / max(len(chars), 1)

    # Digit ratio
    digit_count = sum(1 for c in chars if c.isdigit())
    digit_ratio = digit_count / max(len(chars), 1)

    # AI text: avg word len ~4.5-5.5, very regular punct, low digit ratio
    word_len_score = max(0, 1.0 - abs(avg_word_len - 5.0) / 2.0)
    punct_score = max(0, 1.0 - abs(punct_density - 0.06) / 0.04) if punct_density < 0.10 else 0.3

    combined = word_len_score * 0.5 + punct_score * 0.5

    return {
        "score": round(min(combined, 1.0), 4),
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sent_len, 1),
        "punctuation_density": round(punct_density, 4),
        "uppercase_ratio": round(upper_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
    }

# ═══════════════════════════════════════════════════════════════════
#  8. PERPLEXITY PROXY
# ═══════════════════════════════════════════════════════════════════

def perplexity_proxy(text):
    """Zipf deviation as proxy for model perplexity."""
    words = _tokenize(text)
    if len(words) < 20:
        return {"score": 0.5}

    _, counts = np.unique(words, return_counts=True)
    sorted_c = np.sort(counts)[::-1]
    if len(sorted_c) < 3:
        return {"score": 0.5}

    zipf_dev = float(np.std(sorted_c / np.arange(1, len(sorted_c) + 1)))
    score = min(max(zipf_dev / 5.0, 0), 1.0)

    return {"score": round(score, 4), "zipf_std": round(zipf_dev, 4)}

# ═══════════════════════════════════════════════════════════════════
#  9. POSITIONAL ENTROPY (thirds)
# ═══════════════════════════════════════════════════════════════════

def positional_entropy(text):
    """Compare entropy in first third, middle third, last third."""
    words = _tokenize(text)
    if len(words) < 30:
        return {"score": 0.5}

    third = len(words) // 3
    sections = [words[:third], words[third:2*third], words[2*third:]]

    entropies = []
    for section in sections:
        _, counts = np.unique(section, return_counts=True)
        probs = counts / len(section)
        ent = float(-np.sum(probs * np.log2(probs)))
        entropies.append(ent)

    ent_cv = float(np.std(entropies) / (np.mean(entropies) + 1e-10))

    # AI: uniform entropy across positions; Human: variable
    score = max(0, 1.0 - (ent_cv / 0.3))

    return {
        "score": round(min(score, 1.0), 4),
        "section_entropies": [round(e, 4) for e in entropies],
        "entropy_cv": round(ent_cv, 4),
    }

# ═══════════════════════════════════════════════════════════════════
# 10. FUNCTION WORD DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════

def function_word_analysis(text):
    """Analyze frequency of common function words."""
    FUNCTION_WORDS = {
        "the", "a", "an", "is", "was", "are", "were", "it", "this", "that",
        "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
        "and", "but", "or", "so", "if", "when", "which", "who", "what",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "not", "no", "more", "very", "also", "however", "furthermore",
        "moreover", "additionally", "consequently", "therefore",
    }

    words = _tokenize(text)
    if len(words) < 20:
        return {"score": 0.5}

    fw_count = sum(1 for w in words if w in FUNCTION_WORDS)
    fw_ratio = fw_count / len(words)

    # AI overuses formal transition words
    formal_markers = {"furthermore", "moreover", "additionally", "consequently",
                      "therefore", "nevertheless", "nonetheless", "subsequently"}
    formal_count = sum(1 for w in words if w in formal_markers)
    formal_ratio = formal_count / max(len(words), 1)

    # High formal marker ratio → AI
    fm_score = min(formal_ratio * 50, 1.0)
    # Overly high function word ratio → AI
    fw_score = max(0, (fw_ratio - 0.35) / 0.15) if fw_ratio > 0.35 else 0

    combined = fm_score * 0.6 + fw_score * 0.4

    return {
        "score": round(min(combined, 1.0), 4),
        "function_word_ratio": round(fw_ratio, 4),
        "formal_marker_count": formal_count,
        "formal_marker_ratio": round(formal_ratio, 6),
    }

# ═══════════════════════════════════════════════════════════════════
# 11. CONJUNCTION & TRANSITION DENSITY
# ═══════════════════════════════════════════════════════════════════

def transition_density(text):
    """Detect overuse of transition phrases (AI hallmark)."""
    TRANSITIONS = [
        "in conclusion", "it is important to note", "it is worth noting",
        "in summary", "on the other hand", "as a result", "for instance",
        "for example", "in addition", "in other words", "that being said",
        "to summarize", "in particular", "it should be noted",
        "it is essential", "plays a vital role", "plays a crucial role",
        "it is widely recognized", "research suggests", "studies have shown",
    ]

    text_lower = text.lower()
    hits = 0
    found = []
    for t in TRANSITIONS:
        count = text_lower.count(t)
        if count > 0:
            hits += count
            found.append({"phrase": t, "count": count})

    word_count = len(text.split())
    density = hits / max(word_count / 100, 1)  # per 100 words

    # High density → very likely AI
    score = min(density / 3.0, 1.0)

    return {
        "score": round(score, 4),
        "total_transitions": hits,
        "density_per_100w": round(density, 4),
        "found_phrases": found[:10],
    }

# ═══════════════════════════════════════════════════════════════════
# 12. SENTENCE STARTER DIVERSITY
# ═══════════════════════════════════════════════════════════════════

def sentence_starter_diversity(text):
    """Check if sentence beginnings are diverse (AI repeats patterns)."""
    sents = _sentences(text)
    if len(sents) < 5:
        return {"score": 0.5}

    starters = [s.split()[0].lower() if s.split() else "" for s in sents]
    unique = len(set(starters))
    total = len(starters)
    diversity = unique / total

    # Low diversity → AI (e.g., "The... The... The... It... It...")
    score = max(0, 1.0 - (diversity / 0.7))

    # Check for common AI patterns
    starter_counts = {}
    for s in starters:
        starter_counts[s] = starter_counts.get(s, 0) + 1

    most_common = max(starter_counts.values()) if starter_counts else 0
    repetition = most_common / total

    return {
        "score": round(min(max(score, repetition), 1.0), 4),
        "starter_diversity": round(diversity, 4),
        "most_repeated_ratio": round(repetition, 4),
        "unique_starters": unique,
        "total_sentences": total,
    }

# ═══════════════════════════════════════════════════════════════════
#  MASTER TEXT FORENSIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def full_text_forensics(text):
    """Run ALL 12 forensic techniques and produce a weighted ensemble verdict."""
    techniques = {}

    try: techniques["token_perplexity"] = token_perplexity_profile(text)
    except: techniques["token_perplexity"] = {"score": 0.5}
    try: techniques["sentence_roberta"] = sentence_level_scoring(text)
    except: techniques["sentence_roberta"] = {"score": 0.5}
    try: techniques["sliding_entropy"] = sliding_window_entropy(text)
    except: techniques["sliding_entropy"] = {"score": 0.5}
    try: techniques["burstiness"] = burstiness_profile(text)
    except: techniques["burstiness"] = {"score": 0.5}
    try: techniques["ngram_heatmap"] = ngram_repetition_heatmap(text)
    except: techniques["ngram_heatmap"] = {"score": 0.5}
    try: techniques["vocab_fingerprint"] = vocabulary_fingerprint(text)
    except: techniques["vocab_fingerprint"] = {"score": 0.5}
    try: techniques["stylometrics"] = stylometric_features(text)
    except: techniques["stylometrics"] = {"score": 0.5}
    try: techniques["perplexity_proxy"] = perplexity_proxy(text)
    except: techniques["perplexity_proxy"] = {"score": 0.5}
    try: techniques["positional_entropy"] = positional_entropy(text)
    except: techniques["positional_entropy"] = {"score": 0.5}
    try: techniques["function_words"] = function_word_analysis(text)
    except: techniques["function_words"] = {"score": 0.5}
    try: techniques["transitions"] = transition_density(text)
    except: techniques["transitions"] = {"score": 0.5}
    try: techniques["starter_diversity"] = sentence_starter_diversity(text)
    except: techniques["starter_diversity"] = {"score": 0.5}

    # Weighted ensemble
    weights = {
        "token_perplexity": 0.18, "sentence_roberta": 0.16,
        "sliding_entropy": 0.08, "burstiness": 0.08,
        "ngram_heatmap": 0.08, "vocab_fingerprint": 0.07,
        "stylometrics": 0.06, "perplexity_proxy": 0.05,
        "positional_entropy": 0.06, "function_words": 0.07,
        "transitions": 0.06, "starter_diversity": 0.05,
    }

    master = sum(techniques[k]["score"] * weights.get(k, 0.05) for k in techniques)
    total_weight = sum(weights.get(k, 0.05) for k in techniques)
    master = master / max(total_weight, 1e-10)

    all_scores = [techniques[k]["score"] for k in techniques]
    consensus_std = float(np.std(all_scores))
    confidence = max(0, 1.0 - consensus_std)

    verdict = "AI-GENERATED" if master > 0.5 else "LIKELY HUMAN"
    if confidence < 0.4:
        verdict += " (LOW CONFIDENCE)"

    return {
        "master_score": round(float(min(max(master, 0), 1)), 4),
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "n_techniques": len(techniques),
        "techniques": techniques,
    }
