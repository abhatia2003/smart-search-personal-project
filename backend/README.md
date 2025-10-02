# Backend for Semantic Search

This folder contains the FastAPI backend that embeds page text and answers semantic queries.

Quick start:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn backend:app --reload --port 8000
```
