"""
Step 4 — Build and save a FAISS vector store from chunk embeddings.

Loads embeddings_matrix.npy and chunk_embeddings.json,
builds a FAISS index, and saves everything locally so
Step 6 (query retrieval) can load and search without re-embedding.

Output files:
  faiss_index.bin      — FAISS index (searchable vector store)
  faiss_metadata.json  — chunk metadata mapped by FAISS index position
"""

import json
import os
import time
import numpy as np
import faiss

# ── Input / Output ────────────────────────────────────────────────────────────
EMBEDDINGS_FILE  = "embeddings_matrix.npy"
CHUNKS_FILE      = "chunk_embeddings.json"
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE    = "faiss_metadata.json"

# ── FAISS settings ────────────────────────────────────────────────────────────
# IndexFlatIP = exact inner product search (cosine similarity when normalized)
# Best choice for < 10,000 vectors — no approximation, perfect accuracy
INDEX_TYPE = "IndexFlatIP"

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
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings):
    """
    Build a FAISS IndexFlatIP (inner product) index.

    Why IndexFlatIP:
      - Embeddings are L2-normalized (done in Step 3)
      - Inner product on normalized vectors = cosine similarity
      - Exact search — no approximation errors
      - Perfect for 165 vectors (approximation only needed for millions+)
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # FAISS requires float32
    embeddings_f32 = embeddings.astype(np.float32)
    index.add(embeddings_f32)

    return index


def build_metadata_store(chunks):
    """
    Build a metadata list aligned with FAISS index positions.
    Position i in this list corresponds to vector i in the FAISS index.

    Strips the embedding field to keep the metadata file lightweight.
    """
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "faiss_id":    i,
            "chunk_id":    chunk["chunk_id"],
            "paper_id":    chunk["paper_id"],
            "section":     chunk["section"],
            "source_type": chunk["source_type"],
            "token_count": chunk["token_count"],
            "text":        chunk["text"]
            # embedding excluded — already in FAISS index
        })
    return metadata


def verify_index(index, embeddings, metadata):
    """
    Run a quick self-search to verify the index works correctly.
    Each vector's nearest neighbour should be itself (score = 1.0).
    """
    print("\nVerifying FAISS index ...")
    test_vectors = embeddings[:5].astype(np.float32)
    scores, indices = index.search(test_vectors, k=1)

    all_correct = True
    for i, (score, idx) in enumerate(zip(scores, indices)):
        correct = idx[0] == i
        if not correct:
            all_correct = False
        chunk_id = metadata[i]["chunk_id"]
        print(f"  [{i}] {chunk_id:<25} "
              f"nearest={idx[0]:>3}  score={score[0]:.4f}  "
              f"{'OK' if correct else 'MISMATCH'}")

    if all_correct:
        print("  Self-search verification passed — index is correct")
    else:
        print("  WARNING: Some vectors did not match themselves")

    return all_correct


def print_summary(index, metadata, embeddings):
    """Print FAISS store summary."""
    print(f"\n── FAISS Store Summary ──────────────────────────────────")
    print(f"  Index type        : {INDEX_TYPE}")
    print(f"  Total vectors     : {index.ntotal}")
    print(f"  Embedding dim     : {embeddings.shape[1]}")
    print(f"  Index size        : {os.path.getsize(FAISS_INDEX_FILE) / 1024:.1f} KB")
    print(f"  Metadata size     : {os.path.getsize(METADATA_FILE) / 1024:.1f} KB")

    # Breakdown by paper
    paper_counts = {}
    for m in metadata:
        pid = m["paper_id"]
        paper_counts[pid] = paper_counts.get(pid, 0) + 1

    print(f"\n  Vectors per paper:")
    for pid, count in sorted(paper_counts.items()):
        print(f"    {pid} : {count:>3} vectors")

    # Breakdown by source type
    type_counts = {}
    for m in metadata:
        t = m["source_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  Vectors per source type:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {stype:<15} : {count:>3}")

    print(f"─────────────────────────────────────────────────────────")


def demo_search(index, metadata, embeddings):
    """
    Demo: search using the first chunk as a query.
    Shows what Step 6 will do with a real user query.
    """
    print(f"\nDemo search (using first chunk as query):")
    query_vec = embeddings[0:1].astype(np.float32)
    scores, indices = index.search(query_vec, k=5)

    print(f"  Query chunk : {metadata[0]['chunk_id']} "
          f"— {metadata[0]['text'][:60]}...")
    print(f"\n  Top 5 results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        m = metadata[idx]
        print(f"    {rank}. [{m['chunk_id']}] "
              f"paper={m['paper_id']} "
              f"section={m['section']:<20} "
              f"score={score:.4f}")
              
              
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load embeddings matrix
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings matrix not found: '{EMBEDDINGS_FILE}'")
        print("Run step3_embeddings.py first.")
        return

    print(f"Loading embeddings from '{EMBEDDINGS_FILE}' ...")
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"  Shape : {embeddings.shape}")

    # 2. Load chunk metadata
    chunks = load_json(CHUNKS_FILE)
    if not chunks:
        print("No chunks found. Run step3_embeddings.py first.")
        return
    print(f"  Chunks loaded : {len(chunks)}")

    # Validate alignment
    if len(chunks) != embeddings.shape[0]:
        print(f"ERROR: Mismatch — {len(chunks)} chunks vs "
              f"{embeddings.shape[0]} embeddings")
        return
    print(f"  Alignment check passed ({len(chunks)} == {embeddings.shape[0]})")

    # 3. Build FAISS index
    print(f"\nBuilding FAISS {INDEX_TYPE} index ...")
    start = time.time()
    index = build_faiss_index(embeddings)
    print(f"  Index built in {time.time() - start:.2f}s")
    print(f"  Vectors in index : {index.ntotal}")

    # 4. Build metadata store
    metadata = build_metadata_store(chunks)

    # 5. Save FAISS index
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"\nFAISS index saved -> '{FAISS_INDEX_FILE}'")

    # 6. Save metadata
    save_json(metadata, METADATA_FILE)
    print(f"Metadata saved    -> '{METADATA_FILE}'")

    # 7. Verify index
    verify_index(index, embeddings, metadata)

    # 8. Demo search
    demo_search(index, metadata, embeddings)

    # 9. Summary
    print_summary(index, metadata, embeddings)

    print(f"\nStep 4 complete.")
    print(f"  '{FAISS_INDEX_FILE}'  — load this in Step 6 for retrieval")
    print(f"  '{METADATA_FILE}'     — maps FAISS positions to chunk text")


if __name__ == "__main__":
    main()