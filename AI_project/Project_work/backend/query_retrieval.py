"""
Step 5 — Accept user query, embed it, retrieve top-k chunks from FAISS.

Covers pipeline steps 5, 6, and 7:
  5. Accept a user query and convert it into an embedding
  6. Retrieve top-k most relevant chunks from FAISS using similarity search
  7. Return retrieved chunks with scores and metadata

Output: retrieved_chunks.json
  - Top-k chunks most relevant to the query
  - Used by Step 6 (LLM answer generation)
"""

import json
import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Input / Output ────────────────────────────────────────────────────────────
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE    = "faiss_metadata.json"
OUTPUT_FILE      = "retrieved_chunks.json"

# ── Retrieval settings ────────────────────────────────────────────────────────
MODEL_NAME       = "all-MiniLM-L6-v2"   # must match Step 3
TOP_K            = 5                     # number of chunks to retrieve
MIN_SCORE        = 0.2                   # minimum similarity score threshold
                                         # chunks below this are filtered out

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
# Load FAISS index and metadata
# ---------------------------------------------------------------------------

def load_faiss_store():
    """Load FAISS index and metadata from disk."""
    if not os.path.exists(FAISS_INDEX_FILE):
        print(f"FAISS index not found: '{FAISS_INDEX_FILE}'")
        print("Run step4_faiss_store.py first.")
        return None, None

    print(f"Loading FAISS index from '{FAISS_INDEX_FILE}' ...")
    index    = faiss.read_index(FAISS_INDEX_FILE)
    metadata = load_json(METADATA_FILE)

    print(f"  Vectors in index : {index.ntotal}")
    print(f"  Metadata entries : {len(metadata)}")
    return index, metadata


def load_embedding_model():
    """Load the same model used in Step 3."""
    print(f"Loading embedding model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded — dim: {model.get_sentence_embedding_dimension()}")
    return model


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query(model, query):
    """
    Convert user query text into a normalized embedding vector.
    Uses the same model and normalization as Step 3.
    """
    print(f"\nEmbedding query ...")
    start = time.time()
    query_embedding = model.encode(
        [query],
        convert_to_numpy    = True,
        normalize_embeddings = True   # must match Step 3 normalization
    )
    print(f"  Query embedded in {time.time() - start:.3f}s")
    return query_embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# FAISS retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(index, metadata, query_embedding, top_k=TOP_K):
    """
    Search FAISS index for top-k most similar chunks.

    Returns list of dicts with:
      rank          — result rank (1 = most relevant)
      score         — cosine similarity score (0.0 to 1.0)
      chunk_id      — unique chunk identifier
      paper_id      — source paper
      section       — section name
      source_type   — metadata / section / relationships / citations
      text          — chunk text content
    """
    print(f"\nSearching FAISS index (top_k={top_k}) ...")
    start = time.time()

    scores, indices = index.search(query_embedding, k=top_k)

    elapsed = time.time() - start
    print(f"  Search completed in {elapsed:.4f}s")

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:   # FAISS returns -1 for empty slots
            continue
        if score < MIN_SCORE:
            print(f"  Rank {rank} skipped — score {score:.4f} below "
                  f"threshold {MIN_SCORE}")
            continue

        chunk_meta = metadata[idx]
        results.append({
            "rank":        rank,
            "score":       float(round(score, 4)),
            "chunk_id":    chunk_meta["chunk_id"],
            "paper_id":    chunk_meta["paper_id"],
            "section":     chunk_meta["section"],
            "source_type": chunk_meta["source_type"],
            "token_count": chunk_meta["token_count"],
            "text":        chunk_meta["text"]
        })

    return results


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def display_results(query, results):
    """Pretty-print retrieved chunks."""
    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print(f"{'='*65}")
    print(f"Retrieved {len(results)} chunks:\n")

    for r in results:
        print(f"Rank {r['rank']} — Score: {r['score']:.4f}")
        print(f"  Chunk    : {r['chunk_id']}")
        print(f"  Paper    : {r['paper_id']}")
        print(f"  Section  : {r['section']}")
        print(f"  Type     : {r['source_type']}")
        print(f"  Tokens   : {r['token_count']}")
        print(f"  Preview  : {r['text'][:150].strip()}...")
        print()

    if not results:
        print("No chunks retrieved above the minimum score threshold.")
        print(f"Try lowering MIN_SCORE (currently {MIN_SCORE}) "
              f"or rephrasing the query.")


# ---------------------------------------------------------------------------
# Main retrieval function — called by Step 6 (LLM)
# ---------------------------------------------------------------------------

def retrieve(query, top_k=TOP_K, save_output=True):
    """
    Full retrieval pipeline:
      1. Embed query
      2. Search FAISS
      3. Return ranked chunks

    Can be imported and called directly by Step 6:
      from step5_query_retrieval import retrieve
      chunks = retrieve("What is blockchain reward system?")
    """
    # Load resources
    index, metadata = load_faiss_store()
    if index is None:
        return []

    model = load_embedding_model()

    # Embed query
    query_embedding = embed_query(model, query)

    # Retrieve
    results = retrieve_chunks(index, metadata, query_embedding, top_k)

    # Display
    display_results(query, results)

    # Save output for Step 6
    if save_output:
        output = {
            "query":           query,
            "top_k":           top_k,
            "total_retrieved": len(results),
            "min_score":       MIN_SCORE,
            "results":         results
        }
        save_json(output, OUTPUT_FILE)
        print(f"Retrieved chunks saved -> '{OUTPUT_FILE}'")

    return results


# ---------------------------------------------------------------------------
# Main — interactive query mode
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("RAG Retrieval System — Research Paper Knowledge Graph")
    print("=" * 65)
    print(f"  FAISS index  : {FAISS_INDEX_FILE}")
    print(f"  Top-k        : {TOP_K}")
    print(f"  Min score    : {MIN_SCORE}")
    print(f"  Model        : {MODEL_NAME}")
    print()

    # Load resources once
    index, metadata = load_faiss_store()
    if index is None:
        return

    model = load_embedding_model()

    # Interactive query loop
    print("\nEnter your query (type 'quit' to exit):\n")

    while True:
        query = input("Query: ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Exiting retrieval system.")
            break

        # Embed query
        query_embedding = embed_query(model, query)

        # Retrieve
        results = retrieve_chunks(index, metadata, query_embedding, TOP_K)

        # Display
        display_results(query, results)

        # Save
        output = {
            "query":           query,
            "top_k":           TOP_K,
            "total_retrieved": len(results),
            "min_score":       MIN_SCORE,
            "results":         results
        }
        save_json(output, OUTPUT_FILE)
        print(f"Saved -> '{OUTPUT_FILE}'")
        print(f"\n{'─'*65}\n")


if __name__ == "__main__":
    main()