from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import httpx
import random

app = FastAPI(
    title="AGD Orchestrator API",
    description="Core API Gateway for Artificial General Detector",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    job_id: str
    status: str
    master_score: float | None = None
    category: str | None = None
    modules_analyzed: list[str] = []
    results: dict = {}

def get_category(score: float) -> str:
    if score <= 0.15: return "Highly Likely Human/Authentic"
    elif score <= 0.35: return "Likely Human/Authentic"
    elif score <= 0.65: return "Inconclusive"
    elif score <= 0.85: return "Likely AI-Generated"
    else: return "Highly Likely AI-Generated"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AGD Orchestrator is running"}

@app.post("/analyze/text", response_model=AnalysisResult)
async def analyze_text(text: str = Form(...)):
    job_id = str(uuid.uuid4())
    
    # Simulate calling the text module
    # In a real app we would use httpx.post("http://agd-text:8004/analyze"...)
    
    score = round(random.uniform(0.1, 0.9), 2)
    
    return {
        "job_id": job_id,
        "status": "completed",
        "master_score": score,
        "category": get_category(score),
        "modules_analyzed": ["text"],
        "results": {
            "text": {
                "score": score,
                "breakdown": {
                    "perplexity": round(score * 1.1, 2) % 1,
                    "burstiness": round(score * 0.8, 2),
                    "n_gram_distribution": score
                }
            }
        }
    }

@app.post("/analyze/file", response_model=AnalysisResult)
async def analyze_file(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="MIME type not detected")

    job_id = str(uuid.uuid4())
    mime = file.content_type
    
    modules = []
    results = {}
    master_score = 0.0
    
    score = round(random.uniform(0.1, 0.9), 2)
    master_score = score
    
    if "image" in mime:
        modules.append("image")
        results["image"] = {
            "score": score,
            "breakdown": {
                "frequency_analysis": round(score * 0.9, 2),
                "pixel_level": round(score * 1.1, 2) % 1,
                "cnn_features": score,
                "metadata_inconsistency": round(random.uniform(0.1, 0.9), 2)
            }
        }
    elif "video" in mime:
        modules.append("video")
        results["video"] = {
            "score": score,
            "breakdown": {
                "optical_flow": round(score * 1.2, 2) % 1,
                "temporal_consistency": round(score * 0.8, 2),
                "compression_artifact": score
            }
        }
    elif "audio" in mime:
        modules.append("audio")
        results["audio"] = {
            "score": score,
            "breakdown": {
                "mel_spectrogram": round(score * 1.1, 2) % 1,
                "breathing_pattern": round(score * 0.9, 2),
                "environmental": score
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported MIME type")
        
    return {
        "job_id": job_id,
        "status": "completed",
        "master_score": master_score,
        "category": get_category(master_score),
        "modules_analyzed": modules,
        "results": results
    }
