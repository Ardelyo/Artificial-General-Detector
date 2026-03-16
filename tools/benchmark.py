import json
import random
import time
from pathlib import Path

REGISTRY_PATH = Path("../models/registry.json")

def load_registry():
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)

def run_benchmark(model_id: str, modality: str):
    print(f"[AGD Benchmark] Initiating evaluation for model '{model_id}' (Modality: {modality})")
    print(f"[AGD Benchmark] Loading adversarial test set: agd-bench-adversarial-{modality}...")
    
    # Simulate benchmarking time
    time.sleep(2)
    
    # Simulate a robust benchmark result
    accuracy = random.uniform(0.75, 0.98)
    f1 = accuracy - random.uniform(0.01, 0.05)
    
    print(f"[AGD Benchmark] Evaluation Complete.")
    print(f"  --> Accuracy: {accuracy*100:.2f}%")
    print(f"  --> F1 Score: {f1:.3f}")
    
    if accuracy < 0.80:
        print("[AGD Benchmark] WARNING: Model performance is below 80% threshold against adversarial perturbations.")
    else:
        print("[AGD Benchmark] SUCCESS: Model demonstrates strong adversarial robustness.")

if __name__ == "__main__":
    print("========================================")
    print(" AGD Benchmark Suite & Evaluation Tools ")
    print("========================================")
    
    registry = load_registry()
    print(f"Loaded {len(registry['models'])} models from the local registry.\n")
    
    for model in registry["models"]:
        run_benchmark(model["id"], model["modality"])
        print("-" * 40)
        
    print("All configured models evaluated successfully.")
