"""Retrieval pipeline (AI-3): load -> chunk -> embed (local, free) -> FAISS index -> search."""
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI
from pypdf import PdfReader

log = logging.getLogger("rag")

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index_store"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small, free, runs locally (384 dimensions)
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

_client = None
_embedder = None


def client() -> OpenAI:
    """Groq exposes an OpenAI-compatible API, so we reuse the openai SDK."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)  # downloads ~130MB on first run
    return _embedder


# ---------- 1. Load ----------
def load_documents() -> list[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


# ---------- 2. Chunk ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())  # normalise whitespace
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to break on a sentence/space boundary
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# ---------- 3. Embed (local, no API key, no cost) ----------
def embed_passages(texts: list[str]) -> np.ndarray:
    arr = np.array(list(embedder().passage_embed(texts)), dtype="float32")
    faiss.normalize_L2(arr)  # so inner product == cosine similarity
    return arr


def embed_query(text: str) -> np.ndarray:
    arr = np.array(list(embedder().query_embed(text)), dtype="float32")
    faiss.normalize_L2(arr)
    return arr


# ---------- 4. Index + search ----------
class KnowledgeBase:
    def __init__(self):
        self.index = None
        self.chunks: list[dict] = []

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def build(self):
        docs = load_documents()
        if not docs:
            raise RuntimeError("No .txt/.md/.pdf files found in the docs/ folder.")
        chunks = []
        for doc in docs:
            for n, piece in enumerate(chunk_text(doc["text"])):
                chunks.append({"source": doc["source"], "chunk_id": n, "text": piece})
        log.info("Embedding %d chunks from %d documents...", len(chunks), len(docs))
        vectors = embed_passages([c["text"] for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index, self.chunks = index, chunks

        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    def load_or_build(self):
        idx_file, chunk_file = INDEX_DIR / "faiss.index", INDEX_DIR / "chunks.json"
        if idx_file.exists() and chunk_file.exists():
            self.index = faiss.read_index(str(idx_file))
            self.chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
            log.info("Loaded saved index with %d chunks.", len(self.chunks))
        else:
            self.build()

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = embed_query(query)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            results.append({**self.chunks[i], "score": float(score)})
        return results