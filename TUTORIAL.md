# AGD Usage Tutorial

Welcome to the Artificial General Detector (AGD). This tutorial covers how to run the web interface, directly query the API, and run the underlying forensic benchmarks.

---

## 1. System Requirements
- **Python**: 3.9 or higher (for the forensic detection modules)
- **Node.js**: v18+ (for the Next.js frontend)
- **Git**: For cloning the repository

## 2. Installation Setup

First, initialize the Python backend environment:
```bash
# Clone the repository
git clone https://github.com/Ardelyo/Artificial-General-Detector.git
cd "Artificial General Detector"

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python requirements
pip install fastapi pydantic numpy transformers scipy pillow scikit-learn pandas python-docx matplotlib uvicorn python-multipart
```

Next, initialize the Next.js frontend:
```bash
cd web
npm install
cd ..
```

## 3. Running the Stack

The AGD system is composed of a FastAPI orchestrator that routes files to respective forensic engines, and a Next.js web dashboard.

### Start the Backend API
In a new terminal window, start the orchestrator from the repository root:
```bash
# Runs the FastAPI server on http://localhost:8000
python -m uvicorn core.orchestrator:app --reload
```

### Start the Web UI
In another terminal window, start the Next.js development server:
```bash
cd web
npm run dev
```
Navigate to [http://localhost:3000](http://localhost:3000) in your browser. You will see the Surveillance-Aesthetic dashboard. You can drag and drop media files or paste text to receive a live forensic analysis.

## 4. Running Benchmarks & Experiments

AGD includes tools to rigorously test its 27 forensic techniques against synthetic and authentic datasets. These tools generate comprehensive reports.

### The Ultra-Deep Benchmark
To run an exhaustive evaluation on single/small batch samples and generate a highly detailed DOCX report:
```bash
python tools/ultra_benchmark.py
```
*Output*: A detailed forensic document will be saved to `docs/reports/AGD_UltraDeep_Report.docx`.

### The SOTA Mass Benchmark
To evaluate the detector against 500+ text and image samples and plot a comparative F1-score against other State-of-the-Art (SOTA) detectors:
```bash
python tools/mass_benchmark_sota.py
```
*Output*: CSV metrics and PNG plots will be saved to `docs/reports/` and `docs/figures/`.

### Official Research Paper Generation
To auto-generate the complete AGD whitepaper (with methodology, technique breakdowns, and metrics):
```bash
python tools/generate_research_paper.py
```
*Output*: `docs/reports/AGD_Official_Research_Paper.docx`

## 5. Repository Organization

To ensure a clean workspace, the repository is organized as follows:
- `core/`: FastAPI orchestrator and API routing.
- `modules/`: The individual forensic engines (`agd-text`, `agd-image`, `agd-audio`, `agd-video`) along with their respective Deep Forensic wrappers (`deep_text_forensics.py`, `deep_image_forensics.py`).
- `web/`: Next.js React frontend.
- `tools/`: Benchmarking, automated testing, and report generation scripts.
- `docs/`: Stores benchmark output CSVs, generated DOCX reports (`reports/`), and plot figures (`figures/`).
- `tests/`: Raw input sample data (human vs AI) used by the benchmark scripts.

## Advanced Usage

You can also run modality-specific servers if you do not want to run the entire orchestrator:
```bash
# Start only the Text detector API
uvicorn modules.agd-text.main:app --port 8001

# Start only the Image detector API
uvicorn modules.agd-image.main:app --port 8002
```
