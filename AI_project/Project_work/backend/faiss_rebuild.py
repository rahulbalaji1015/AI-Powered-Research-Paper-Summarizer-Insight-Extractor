import json
import numpy as np
import faiss

# Step 1 — Rebuild embeddings matrix from chunk_embeddings.json
with open("chunk_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Chunks loaded: {len(chunks)}")

# Extract embedding vectors
embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
print(f"Embeddings matrix shape: {embeddings.shape}")

# Save new matrix
np.save("embeddings_matrix.npy", embeddings)
print("embeddings_matrix.npy saved")

# Step 2 — Rebuild FAISS index from new matrix
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")
print(f"FAISS index rebuilt — {index.ntotal} vectors")

# Step 3 — Rebuild metadata aligned with new index
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
    })

with open("faiss_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)
print(f"faiss_metadata.json rebuilt — {len(metadata)} entries")

# Verify alignment
assert index.ntotal == len(metadata) == len(chunks)
print(f"\nAll aligned: {index.ntotal} vectors == {len(metadata)} metadata == {len(chunks)} chunks")