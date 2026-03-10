# Run this once in terminal to remove bad chunks
import json

with open("faiss_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("chunk_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Remove P11 chunks
metadata = [m for m in metadata if m["paper_id"] != "P11"]
chunks   = [c for c in chunks   if c["paper_id"] != "P11"]

with open("faiss_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

with open("chunk_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print(f"Cleaned. Metadata: {len(metadata)} | Chunks: {len(chunks)}")