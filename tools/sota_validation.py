"""
SOTA Validation Test — Tests all upgraded modules locally.
"""
import sys, os, importlib.util, time

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    if spec is None or spec.loader is None:
        print(f"Error: Could not find module spec or loader for {name} at {path}")
        sys.exit(1)
    m = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(m)  # type: ignore
    return m

print("=" * 60)
print(" AGD SOTA MODULE VALIDATION")
print("=" * 60)

# ── TEST 1: TEXT ──────────────────────────────────────────────────
print("\n▸ Loading agd-text SOTA module...")
txt = load_mod("txt", "modules/agd-text/main.py")

with open("tests/ai_text.txt", "r", encoding="utf-8") as f:
    ai_text = f.read()
with open("tests/human_text.txt", "r", encoding="utf-8") as f:
    human_text = f.read()

print("\n[TEXT] AI-generated sample:")
cv = txt.compute_burstiness(ai_text)
ent = txt.compute_entropy(ai_text)
ng = txt.compute_ngram_repetition(ai_text)
pp = txt.compute_perplexity_proxy(ai_text)
rb = txt.roberta_score(ai_text)
print(f"  Burstiness CV: {cv:.4f}")
print(f"  Entropy: {ent:.4f}")
print(f"  N-gram rep: {ng:.4f}")
print(f"  Perplexity proxy: {pp:.4f}")
print(f"  RoBERTa score: {rb:.4f}")

print("\n[TEXT] Human-written sample:")
cv = txt.compute_burstiness(human_text)
ent = txt.compute_entropy(human_text)
ng = txt.compute_ngram_repetition(human_text)
pp = txt.compute_perplexity_proxy(human_text)
rb = txt.roberta_score(human_text)
print(f"  Burstiness CV: {cv:.4f}")
print(f"  Entropy: {ent:.4f}")
print(f"  N-gram rep: {ng:.4f}")
print(f"  Perplexity proxy: {pp:.4f}")
print(f"  RoBERTa score: {rb:.4f}")

# ── TEST 2: IMAGE ─────────────────────────────────────────────────
print("\n▸ Loading agd-image SOTA module...")
img = load_mod("img", "modules/agd-image/main.py")

home = os.path.expanduser("~")
artifacts_dir = os.path.join(home, '.gemini/antigravity/brain/7763f4e8-7f16-4110-9bc7-c32ba10ba671')
ai_img_path = None
for f in sorted(os.listdir(artifacts_dir), reverse=True):
    if f.startswith("ai_test_photo"):
        ai_img_path = os.path.join(artifacts_dir, f)
        break

if ai_img_path and os.path.exists(str(ai_img_path)):  # type: ignore
    with open(ai_img_path, "rb") as f:
        ai_bytes = f.read()
    avg, ela = img.compute_ela_baseline(ai_bytes)
    fr, fs = img.compute_frequency_score(ai_bytes)
    rs, rk, ss = img.compute_srm_score(ai_bytes)
    un, cs = img.compute_color_uniformity(ai_bytes)
    print(f"\n[IMAGE] AI-generated:")
    print(f"  ELA: {ela:.4f} | Frequency: {fs:.4f} | SRM: {ss:.4f} | Color: {cs:.4f}")
else:
    print("\n[IMAGE] AI-generated sample not found.")

if os.path.exists("tests/real_test_photo.png"):
    with open("tests/real_test_photo.png", "rb") as f:
        real_bytes = f.read()
    avg, ela = img.compute_ela_baseline(real_bytes)
    fr, fs = img.compute_frequency_score(real_bytes)
    rs, rk, ss = img.compute_srm_score(real_bytes)
    un, cs = img.compute_color_uniformity(real_bytes)
    print(f"\n[IMAGE] Real photo:")
    print(f"  ELA: {ela:.4f} | Frequency: {fs:.4f} | SRM: {ss:.4f} | Color: {cs:.4f}")

print("\n" + "=" * 60)
print(" VALIDATION COMPLETE")
print("=" * 60)
