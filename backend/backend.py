from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import spacy
import faiss
import numpy as np
import threading
import os
import pickle
import torch

# -----------------------
# Configuration
# -----------------------
EMBEDDING_DIM = 768
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "stored_chunks.pkl"

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Semantic Search Backend (M2 Optimized)")

# Allow browser-based access (CORS)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Data models
# -----------------------
class TextInput(BaseModel):
    text: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    source: Optional[str] = None

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

# -----------------------
# Globals
# -----------------------
nlp = None
model = None
index = None
stored_chunks: List[Dict[str, Any]] = []
index_lock = threading.Lock()

# -----------------------
# Helpers
# -----------------------
def detect_device() -> str:
    """Detect the best available device: MPS (Apple GPU) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_resources():
    """Load all heavy resources: spaCy, transformer model, FAISS index"""
    global nlp, model, index, stored_chunks

    # Load spaCy for sentence segmentation
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        raise RuntimeError("spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm")

    # Select device (MPS > CUDA > CPU)
    device = detect_device()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    print(f"[backend] Using device: {device}")

    # Load or initialize FAISS index
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                stored_chunks = pickle.load(f)
            print(f"[backend] Loaded index with {index.ntotal} vectors and {len(stored_chunks)} chunks.")
        except Exception as e:
            print("[backend] Failed to load existing index, creating new one:", e)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            stored_chunks = []
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        stored_chunks = []

def chunk_text(text: str) -> List[str]:
    """Split long text into sentences."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def embed_text(chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed text using SentenceTransformer on MPS/GPU if available."""
    if not chunks:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")

    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings

def persist_index():
    """Save FAISS index and stored chunks to disk."""
    with index_lock:
        try:
            faiss.write_index(index, INDEX_PATH)
            with open(CHUNKS_PATH, "wb") as f:
                pickle.dump(stored_chunks, f)
            print(f"[backend] Persisted index ({index.ntotal}) and {len(stored_chunks)} chunks.")
        except Exception as e:
            print("[backend] Failed to persist index:", e)

# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def on_startup():
    load_resources()

# -----------------------
# API Endpoints
# -----------------------
@app.post("/embed")
async def embed_endpoint(data: TextInput):
    """Receive text, chunk it, embed it, and add to FAISS index."""
    global stored_chunks, index

    # Handle pre-chunked or raw text input
    if data.chunks and len(data.chunks) > 0:
        input_chunks = [c for c in data.chunks if isinstance(c, dict) and c.get("text")]
        texts = [c["text"] for c in input_chunks]
    elif data.text and data.text.strip():
        texts = chunk_text(data.text)
        input_chunks = [{"text": t, "local_index": i} for i, t in enumerate(texts)]
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'chunks'")

    if not texts:
        return {"status": "no_chunks", "chunks_added": 0}

    # Embed and add to FAISS
    embeddings = embed_text(texts)
    with index_lock:
        start_idx = index.ntotal
        index.add(embeddings)
        for i, c in enumerate(input_chunks):
            stored_chunks.append({
                "text": c["text"],
                "source": data.source,
                "global_index": int(start_idx + i),
                "local_index": c.get("local_index"),
                "position": c.get("position"),
            })

    threading.Thread(target=persist_index, daemon=True).start()
    return {"status": "success", "chunks_added": len(input_chunks), "total_vectors": index.ntotal}

@app.post("/query")
async def query_endpoint(data: QueryInput):
    """Embed query and retrieve top-k similar chunks."""
    global index, stored_chunks
    if index is None or index.ntotal == 0:
        return {"error": "No data in index. Please embed text first."}

    query_vec = embed_text([data.query])
    k = min(data.top_k, index.ntotal)
    with index_lock:
        distances, indices = index.search(query_vec, k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(stored_chunks):
            meta = stored_chunks[idx]
            results.append({
                "chunk": meta["text"],
                "source": meta.get("source"),
                "score": float(score),
                "global_index": int(meta["global_index"]),
                "local_index": meta.get("local_index"),
                "position": meta.get("position"),
            })
    return {"query": data.query, "results": results}

@app.get("/status")
async def status():
    """Return number of stored vectors."""
    return {"vectors_in_index": index.ntotal if index else 0, "stored_chunks": len(stored_chunks)}

@app.post("/clear")
async def clear_index():
    """Clear FAISS index and delete persisted files."""
    global index, stored_chunks
    with index_lock:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        stored_chunks = []
        for path in [INDEX_PATH, CHUNKS_PATH]:
            if os.path.exists(path):
                os.remove(path)
    return {"status": "cleared"}

@app.post("/save")
async def save_index():
    persist_index()
    return {"status": "saved"}