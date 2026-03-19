import sys
import os
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not find module spec or loader for {name} at {path}")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# Load deep_image_forensics module with checks
spec = importlib.util.find_spec("modules.deep_image_forensics")
if spec is None or spec.loader is None:
    print("Error: Could not find modules.deep_image_forensics or loader")
    sys.exit(1)
module = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(module)  # type: ignore
deep_image = module

image_module = load_module("image_main", os.path.abspath("modules/agd-image/main.py"))
text_module = load_module("text_main", os.path.abspath("modules/agd-text/main.py"))

def test_image(filepath, label):
    print(f"\n[IMAGE TEST - {label}] File: {filepath}")
    if not os.path.exists(filepath):
        print("  -> ERROR: File not found.")
        return
        
    with open(filepath, "rb") as f:
        img_bytes = f.read()
        
    avg_diff, ela_score = image_module.compute_ela_baseline(img_bytes)
    print(f"  -> Raw ELA Average Pixel Difference: {avg_diff:.4f}")
    print(f"  -> Predicted AI Confidence Score: {ela_score*100:.2f}%")
    
def test_text(filepath, label):
    print(f"\n[TEXT TEST - {label}] File: {filepath}")
    if not os.path.exists(filepath):
        print("  -> ERROR: File not found.")
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        text_content = f.read()
        
    cv = text_module.compute_burstiness(text_content)
    entropy = text_module.compute_entropy(text_content)
    
    # Replicating the internal module logic for the console report
    cv_ai_score = max(0.0, 1.0 - (cv / 0.6))  # type: ignore
    entropy_ai_score = max(0.0, 1.0 - (entropy / 9.0))  # type: ignore
    master_score = (cv_ai_score * 0.6) + (entropy_ai_score * 0.4)
    master_score = min(max(master_score, 0.0), 1.0)

    print(f"  -> CV (Burstiness): {cv:.4f} (Score: {cv_ai_score:.2f})")
    print(f"  -> Entropy (Vocabulary): {entropy:.4f} (Score: {entropy_ai_score:.2f})")
    print(f"  -> Final Predicted AI Confidence Score: {master_score*100:.2f}%")

def main():
    print("==================================================")
    print(" AGD REAL BASELINE DETECTION TESTING (No Mocking) ")
    print("==================================================")
    
    home_dir = os.path.expanduser("~")
    ai_img_path = None
    artifacts_dir = os.path.join(home_dir, '.gemini/antigravity/brain/7763f4e8-7f16-4110-9bc7-c32ba10ba671')
    if os.path.exists(artifacts_dir):
        files = os.listdir(artifacts_dir)
        # Find the most recent one or any ai_test_photo
        for f in sorted(files, reverse=True):
            if f.startswith("ai_test_photo"):
                ai_img_path = os.path.join(artifacts_dir, f)
                break
                
    if ai_img_path:
        test_image(ai_img_path, "AI GENERATED IMAGE")
    else:
        print("  -> ERROR: AI Test Photo not found.")
        
    test_image("tests/real_test_photo.png", "REAL WIKIMEDIA PHOTO")
    
    test_text("tests/ai_text.txt", "AI GENERATED TEXT")
    test_text("tests/human_text.txt", "REAL HUMAN WRITTEN")

if __name__ == "__main__":
    main()
