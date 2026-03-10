"""
config.py — Single source of truth for all shared paths and settings.

Place this file at the PROJECT ROOT (one level above backend/ and frontend/).

Directory layout expected:
    project/
    ├── config.py          ← this file
    ├── backend/
    │   ├── chat_db.py
    │   ├── pdf_export.py
    │   ├── full_process_upload.py
    │   ├── step1_neo4j_fetch.py  ... step8
    │   └── (all other backend scripts)
    ├── frontend/
    │   └── app.py
    └── data/              ← all generated data files live here (auto-created)
        ├── faiss_index.bin
        ├── faiss_metadata.json
        ├── chunk_embeddings.json
        ├── embeddings_matrix.npy
        ├── graph_documents.json
        ├── text_chunks.json
        ├── chat_memory.json
        └── rag_chat.db
"""

import os

# ── Project root = directory containing this file ────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR= os.path.join(ROOT_DIR, "frontend")
DATA_DIR    = os.path.join(ROOT_DIR, "data")

# Auto-create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


def data(filename: str) -> str:
    """Return absolute path to a file inside the data/ directory."""
    return os.path.join(DATA_DIR, filename)


# ── Data file paths ───────────────────────────────────────────────────────────
FAISS_INDEX_FILE     = data("faiss_index.bin")
METADATA_FILE        = data("faiss_metadata.json")
CHUNKS_FILE          = data("chunk_embeddings.json")
EMBEDDINGS_FILE      = data("embeddings_matrix.npy")
GRAPH_DOCUMENTS_FILE = data("graph_documents.json")
TEXT_CHUNKS_FILE     = data("text_chunks.json")
MEMORY_FILE          = data("chat_memory.json")
DB_FILE              = data("rag_chat.db")

# ── Model + LLM settings ──────────────────────────────────────────────────────
GROQ_API_KEY  = "gsk_xxxxxxxxxxxxxxxxxxxxxx"        # ← paste your Groq key here
LLM_MODEL     = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
TEMPERATURE   = 0.2
MAX_TOKENS    = 1024

# ── RAG settings ──────────────────────────────────────────────────────────────
TOP_K                = 7
MIN_SCORE            = 0.15
REL_THRESHOLD        = 0.4
IRRELEVANT_THRESHOLD = 0.22
MEMORY_SIMILARITY    = 0.80
MEMORY_MAX           = 50

# ── Neo4j settings ────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "********"     # ← paste your Neo4j password here
NEO4J_DATABASE = "neo4j"