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

# -----------------------
# Configuration
# -----------------------
EMBEDDING_DIM = 768
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "stored_chunks.pkl"

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Semantic Search Backend")

# Allow the extension to call the backend from the browser
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Models
# -----------------------
class TextInput(BaseModel):
    # Either provide raw text (text) OR provide pre-chunked content (chunks).
    text: Optional[str] = None
    # chunks: list of dicts {"text": str, "local_index": int, "position": {top,left,width,height}}
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

def load_resources():
    global nlp, model, index, stored_chunks
    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # If the model isn't installed, instruct user to install manually
        raise RuntimeError("spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm")

    # Load sentence-transformers
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Init or load FAISS index
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                stored_chunks = pickle.load(f)
            print(f"[backend] Loaded index with {index.ntotal} vectors and {len(stored_chunks)} stored chunks.")
        except Exception as e:
            print("[backend] Failed to load existing index, creating a new one.", e)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            stored_chunks = []
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        stored_chunks = []


def chunk_text(text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def embed_text(chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """Return L2-normalized embeddings for a list of chunks as a numpy array."""
    if not chunks:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")

    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    # Ensure dtype
    embeddings = embeddings.astype("float32")
    # Normalize for cosine similarity and use IndexFlatIP
    faiss.normalize_L2(embeddings)
    return embeddings


def persist_index():
    """Persist FAISS index and stored chunks to disk."""
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
# Endpoints
# -----------------------

@app.post("/embed")
async def embed_endpoint(data: TextInput):
    """
    Receive raw text from extension → chunk → embed → store in FAISS.
    """
    global stored_chunks, index

    # Determine chunks source: either provided pre-chunked or chunk raw text
    if data.chunks and len(data.chunks) > 0:
        # Validate chunk entries
        input_chunks = []
        for c in data.chunks:
            if not isinstance(c, dict) or not c.get("text"):
                continue
            input_chunks.append(c)
        if not input_chunks:
            return {"status": "no_chunks", "chunks_added": 0}

        texts = [c["text"] for c in input_chunks]
    elif data.text and data.text.strip():
        texts = chunk_text(data.text)
        # convert into chunk dicts with local_index sequential
        input_chunks = [{"text": t, "local_index": i} for i, t in enumerate(texts)]
        if not texts:
            return {"status": "no_chunks", "chunks_added": 0}
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or non-empty 'chunks'")

    # Embed texts
    embeddings = embed_text(texts)

    # Add to FAISS and store metadata
    with index_lock:
        start_idx = index.ntotal
        index.add(embeddings)
        for i, c in enumerate(input_chunks):
            stored_chunks.append({
                "text": c.get("text"),
                "source": data.source,
                "global_index": int(start_idx + i),
                "local_index": c.get("local_index"),
                "position": c.get("position")
            })

    # Persist in background
    threading.Thread(target=persist_index, daemon=True).start()

    return {"status": "success", "chunks_added": len(input_chunks), "total_vectors": index.ntotal}


@app.post("/query")
async def query_endpoint(data: QueryInput):
    """
    Receive a query → embed → retrieve top_k most similar chunks.
    Returns cosine-similarity scores (since embeddings are normalized and we use IndexFlatIP).
    """
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
                "global_index": int(meta.get("global_index")),
                "local_index": meta.get("local_index"),
                "position": meta.get("position")
            })

    return {"query": data.query, "results": results}


@app.get("/status")
async def status():
    return {"vectors_in_index": index.ntotal if index is not None else 0, "stored_chunks": len(stored_chunks)}


@app.post("/clear")
async def clear_index():
    global index, stored_chunks
    with index_lock:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        stored_chunks = []
        # remove persisted files if exist
        try:
            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)
            if os.path.exists(CHUNKS_PATH):
                os.remove(CHUNKS_PATH)
        except Exception:
            pass
    return {"status": "cleared"}


@app.post("/save")
async def save_index():
    persist_index()
    return {"status": "saved"}
