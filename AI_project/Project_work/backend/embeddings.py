"""
Step 3 — Generate embeddings for each text chunk.

Uses sentence-transformers to convert each chunk's text
into a dense vector embedding.

Model: all-MiniLM-L6-v2
  - Fast and lightweight
  - 384-dimensional embeddings
  - Good for semantic similarity search
  - Works well for research paper content

Output: chunk_embeddings.json
  - All chunks with their embeddings
  - Ready for Step 4 (FAISS vector store)
"""

import json
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Input / Output ────────────────────────────────────────────────────────────
INPUT_FILE       = "text_chunks.json"
OUTPUT_FILE      = "chunk_embeddings.json"
EMBEDDINGS_NPY   = "embeddings_matrix.npy"   # raw matrix saved separately for FAISS

# ── Model settings ────────────────────────────────────────────────────────────
MODEL_NAME  = "all-MiniLM-L6-v2"   # fast, 384-dim, good quality
BATCH_SIZE  = 32                    # process chunks in batches

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------

def load_model(model_name):
    """Load sentence transformer model."""
    print(f"Loading model: '{model_name}' ...")
    start = time.time()
    model = SentenceTransformer(model_name)
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    print(f"  Embedding dimension : {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(model, chunks):
    """
    Generate embeddings for all chunks in batches.
    Returns numpy array of shape (num_chunks, embedding_dim).
    """
    texts     = [c["text"] for c in chunks]
    total     = len(texts)

    print(f"\nGenerating embeddings for {total} chunks ...")
    print(f"  Batch size : {BATCH_SIZE}")

    start      = time.time()
    embeddings = model.encode(
        texts,
        batch_size        = BATCH_SIZE,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True   # normalize for cosine similarity
    )
    elapsed = time.time() - start

    print(f"\nEmbeddings generated in {elapsed:.1f}s")
    print(f"  Shape : {embeddings.shape}  "
          f"({embeddings.shape[0]} chunks x {embeddings.shape[1]} dims)")

    return embeddings


def attach_embeddings(chunks, embeddings):
    """
    Attach embedding vector to each chunk dict.
    Also adds embedding_dim for reference.
    """
    enriched = []
    for i, chunk in enumerate(chunks):
        enriched.append({
            **chunk,
            "embedding_dim": int(embeddings.shape[1]),
            "embedding":     embeddings[i].tolist()   # list for JSON serialization
        })
    return enriched


def print_summary(chunks_with_embeddings, embeddings):
    """Print embedding quality summary."""
    print(f"\n── Embedding summary ────────────────────────────────────")
    print(f"  Total chunks embedded : {len(chunks_with_embeddings)}")
    print(f"  Embedding dimension   : {embeddings.shape[1]}")
    print(f"  Matrix size           : {embeddings.nbytes / 1024:.1f} KB")

    # Per paper chunk count
    paper_counts = {}
    for c in chunks_with_embeddings:
        pid = c["paper_id"]
        paper_counts[pid] = paper_counts.get(pid, 0) + 1

    print(f"\n  Chunks per paper:")
    for pid, count in sorted(paper_counts.items()):
        print(f"    {pid} : {count} chunks")

    # Per source type
    type_counts = {}
    for c in chunks_with_embeddings:
        t = c["source_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  Chunks per source type:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {stype:<15} : {count}")

    print(f"─────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load chunks
    chunks = load_json(INPUT_FILE)
    if not chunks:
        print("No chunks found. Run step2_text_conversion.py first.")
        return
    print(f"Loaded {len(chunks)} chunks from '{INPUT_FILE}'")

    # 2. Load model
    model = load_model(MODEL_NAME)

    # 3. Generate embeddings
    embeddings = generate_embeddings(model, chunks)

    # 4. Save raw embeddings matrix separately for FAISS (Step 4)
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"Embeddings matrix saved -> '{EMBEDDINGS_NPY}'")

    # 5. Attach embeddings to chunks and save full JSON
    chunks_with_embeddings = attach_embeddings(chunks, embeddings)
    save_json(chunks_with_embeddings, OUTPUT_FILE)
    print(f"Chunks with embeddings saved -> '{OUTPUT_FILE}'")

    # 6. Summary
    print_summary(chunks_with_embeddings, embeddings)

    print(f"\nStep 3 complete.")
    print(f"  '{EMBEDDINGS_NPY}'  — raw matrix for FAISS index (Step 4)")
    print(f"  '{OUTPUT_FILE}'     — chunks with metadata + embeddings")


if __name__ == "__main__":
    main()