# Contributing to AGD

Thank you for your interest in contributing to the Artificial General Detector (AGD) project!

## How to Contribute

1. **Fork** the repository.
2. Create a new **feature branch** (`git checkout -b feature/my-improvement`).
3. Make your changes and ensure all tests pass.
4. Submit a **Pull Request** with a clear description.

## Development Setup

```bash
pip install fastapi pydantic numpy transformers scipy pillow scikit-learn pandas python-docx matplotlib
```

## Code Style

- Use Python type hints consistently.
- Follow PEP 8 guidelines.
- Add docstrings to all public functions.

## Module Architecture

Each detection module lives in `modules/agd-{modality}/main.py` and must expose:
- A `FastAPI` app instance.
- A `/analyze` POST endpoint returning a score between 0.0 and 1.0.

## Reporting Issues

Open a GitHub Issue with:
- Steps to reproduce.
- Expected vs. actual behavior.
- Python version and OS.

---
*Maintained by OurCreativity (Admin: Ardelyo)*
