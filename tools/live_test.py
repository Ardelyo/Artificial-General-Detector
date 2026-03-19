import requests
import json
import os
import shutil
import time

CORE_API = "http://localhost:8000/analyze"

def analyze_file(filepath):
    print(f"\nEvaluating Image: {filepath}")
    with open(filepath, "rb") as f:
        response = requests.post(f"{CORE_API}/file", files={"file": f})
    
    if response.status_code == 200:
        res = response.json()
        print(f"  --> Master Score: {res['master_score']:.2f}")
        print(f"  --> Verdict: {res['category']}")
        if "image" in res["results"]:
            print(f"  --> ELA Score: {res['results']['image']['breakdown']['error_level_analysis']:.2f}")
    else:
        print(f"  --> Error: {response.text}")

def analyze_text(text_content):
    print(f"\nEvaluating Text Snippet...")
    print(f"Snippet: '{text_content[:50]}...'")
    response = requests.post(f"{CORE_API}/text", data={"text": text_content})
    
    if response.status_code == 200:
        res = response.json()
        print(f"  --> Master Score: {res['master_score']:.2f}")
        print(f"  --> Verdict: {res['category']}")
        if "text" in res["results"]:
            print(f"  --> Burstiness Score: {res['results']['text']['breakdown']['burstiness']:.2f}")
    else:
        print(f"  --> Error: {response.text}")

def prepare_and_run():
    # Copy the AI generated image to the tests folder
    home_dir = os.path.expanduser("~")
    # Actually the easiest way is to just point to the artifacts dir
    ai_img_path = None
    for file in os.listdir(os.path.join(home_dir, '.gemini/antigravity/brain/7763f4e8-7f16-4110-9bc7-c32ba10ba671')):
        if file.startswith("ai_test_photo"):
            ai_img_path = os.path.join(home_dir, '.gemini/antigravity/brain/7763f4e8-7f16-4110-9bc7-c32ba10ba671', file)
            break
            
    if not ai_img_path:
        print("Warning: Could not find generated AI image.")
        return
        
    print("====================================")
    print(" AGD LIVE INTEGRATION TESTING SUITE ")
    print("====================================")
    time.sleep(1) # wait for fast API cold start if needed
    
    analyze_file(ai_img_path)
    analyze_file("tests/real_test_photo.png")
    
    with open("tests/ai_text.txt", "r") as f:
        analyze_text(f.read())
        
    with open("tests/human_text.txt", "r") as f:
        analyze_text(f.read())
        
if __name__ == "__main__":
    prepare_and_run()
