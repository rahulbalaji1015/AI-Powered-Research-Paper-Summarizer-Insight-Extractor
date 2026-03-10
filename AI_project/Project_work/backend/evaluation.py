"""
Step 7 — Compute confidence score and evaluation metrics.

Covers pipeline steps 10 and 11:
  10. Compute confidence score using cosine similarity of retrieved chunks
  11. Calculate evaluation metrics:
        - Precision@k
        - Recall@k
        - Semantic Similarity
        - Faithfulness

Input:  rag_answer.json  (from Step 6)
Output: evaluation_report.json
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Input / Output ────────────────────────────────────────────────────────────
ANSWER_FILE     = "rag_answer.json"
CHUNKS_FILE     = "chunk_embeddings.json"
OUTPUT_FILE     = "evaluation_report.json"

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "all-MiniLM-L6-v2"   # same model as Step 3
RELEVANCE_THRESHOLD = 0.5              # score above this = relevant chunk
TOP_K           = 5

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Metric 1 — Confidence Score
# ---------------------------------------------------------------------------

def compute_confidence_score(retrieved_chunks):
    """
    Confidence score = weighted average of retrieval similarity scores.

    Higher score = model is more confident the retrieved chunks
    are relevant to the query.

    Scale:
      0.8 - 1.0 : Very high confidence
      0.6 - 0.8 : High confidence
      0.4 - 0.6 : Medium confidence
      0.0 - 0.4 : Low confidence
    """
    if not retrieved_chunks:
        return 0.0

    scores = [c.get("score", 0.0) for c in retrieved_chunks]

    # Weighted average — higher ranked chunks get more weight
    weights = [1 / (i + 1) for i in range(len(scores))]
    weighted_sum   = sum(s * w for s, w in zip(scores, weights))
    weight_total   = sum(weights)
    confidence     = weighted_sum / weight_total if weight_total > 0 else 0.0

    return round(confidence, 4)


def confidence_label(score):
    if score >= 0.8:
        return "Very High"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Metric 2 — Precision@k
# ---------------------------------------------------------------------------

def compute_precision_at_k(retrieved_chunks, k=TOP_K,
                             threshold=RELEVANCE_THRESHOLD):
    """
    Precision@k = number of relevant chunks in top-k / k

    A chunk is considered relevant if its similarity score
    is above the RELEVANCE_THRESHOLD.

    Scale: 0.0 to 1.0 (1.0 = all retrieved chunks are relevant)
    """
    top_k_chunks  = retrieved_chunks[:k]
    relevant      = sum(1 for c in top_k_chunks
                        if c.get("score", 0.0) >= threshold)
    precision     = relevant / k if k > 0 else 0.0
    return round(precision, 4), relevant, k


# ---------------------------------------------------------------------------
# Metric 3 — Recall@k
# ---------------------------------------------------------------------------

def compute_recall_at_k(retrieved_chunks, all_chunks_file,
                         query_embedding, model, k=TOP_K,
                         threshold=RELEVANCE_THRESHOLD):
    """
    Recall@k = relevant chunks retrieved in top-k /
               total relevant chunks in entire corpus

    A chunk in the corpus is considered relevant if its
    embedding similarity to the query is above threshold.
    """
    # Load all chunk embeddings
    all_chunks = load_json(all_chunks_file)
    if not all_chunks:
        return 0.0, 0, 0

    # Count total relevant chunks in corpus
    total_relevant = 0
    for chunk in all_chunks:
        emb = chunk.get("embedding", [])
        if emb:
            sim = cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                total_relevant += 1

    if total_relevant == 0:
        return 0.0, 0, 0

    # Count relevant in top-k retrieved
    retrieved_relevant = sum(
        1 for c in retrieved_chunks[:k]
        if c.get("score", 0.0) >= threshold
    )

    recall = retrieved_relevant / total_relevant
    return round(recall, 4), retrieved_relevant, total_relevant


# ---------------------------------------------------------------------------
# Metric 4 — Semantic Similarity
# ---------------------------------------------------------------------------

def compute_semantic_similarity(query, answer, model):
    """
    Semantic Similarity = cosine similarity between
    query embedding and answer embedding.

    Measures how semantically aligned the answer is
    with the original question.

    Scale: 0.0 to 1.0 (1.0 = perfect semantic match)
    """
    embeddings = model.encode(
        [query, answer],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return round(similarity, 4)


# ---------------------------------------------------------------------------
# Metric 5 — Faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness(answer, retrieved_chunks, model):
    """
    Faithfulness = measures how well the answer is grounded
    in the retrieved context (not hallucinated).

    Method:
      - Split answer into sentences
      - For each sentence, compute max similarity to any chunk
      - Faithfulness = average of max similarities

    Scale:
      0.8 - 1.0 : Highly faithful — answer closely follows context
      0.6 - 0.8 : Faithful — mostly grounded in context
      0.4 - 0.6 : Partially faithful — some content may be outside context
      0.0 - 0.4 : Low faithfulness — possible hallucination
    """
    # Split answer into sentences
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".")
                 if len(s.strip()) > 20]

    if not sentences:
        return 0.0, []

    # Get chunk texts
    chunk_texts = [c.get("text", "") for c in retrieved_chunks]
    if not chunk_texts:
        return 0.0, []

    # Embed all sentences and chunks
    all_texts  = sentences + chunk_texts
    embeddings = model.encode(
        all_texts,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    sent_embeddings  = embeddings[:len(sentences)]
    chunk_embeddings = embeddings[len(sentences):]

    # For each sentence find max similarity to any chunk
    sentence_scores = []
    for i, sent_emb in enumerate(sent_embeddings):
        max_sim = max(
            cosine_similarity(sent_emb, chunk_emb)
            for chunk_emb in chunk_embeddings
        )
        sentence_scores.append({
            "sentence": sentences[i],
            "max_similarity": round(float(max_sim), 4)
        })

    faithfulness = np.mean([s["max_similarity"] for s in sentence_scores])
    return round(float(faithfulness), 4), sentence_scores


def faithfulness_label(score):
    if score >= 0.8:
        return "Highly Faithful"
    elif score >= 0.6:
        return "Faithful"
    elif score >= 0.4:
        return "Partially Faithful"
    else:
        return "Low Faithfulness — possible hallucination"


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(answer_data, model):
    """Run all metrics and return evaluation report."""
    query             = answer_data.get("query", "")
    answer            = answer_data.get("answer", "")
    retrieved_chunks  = answer_data.get("retrieved_chunks", [])

    print(f"\nRunning evaluation for query:")
    print(f"  '{query[:70]}'\n")

    # Embed query for recall computation
    print("  Embedding query ...")
    query_embedding = model.encode(
        [query],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )[0]

    # Metric 1 — Confidence Score
    print("  Computing confidence score ...")
    confidence = compute_confidence_score(retrieved_chunks)

    # Metric 2 — Precision@k
    print("  Computing Precision@k ...")
    precision, rel_retrieved, k = compute_precision_at_k(retrieved_chunks)

    # Metric 3 — Recall@k
    print("  Computing Recall@k ...")
    recall, rel_ret, total_rel = compute_recall_at_k(
        retrieved_chunks, CHUNKS_FILE, query_embedding, model
    )

    # Metric 4 — Semantic Similarity
    print("  Computing semantic similarity ...")
    semantic_sim = compute_semantic_similarity(query, answer, model)

    # Metric 5 — Faithfulness
    print("  Computing faithfulness ...")
    faithfulness, sentence_scores = compute_faithfulness(
        answer, retrieved_chunks, model
    )

    # Build report
    report = {
        "query":  query,
        "answer": answer,
        "model":  answer_data.get("model", ""),
        "metrics": {
            "confidence_score": {
                "score":       confidence,
                "label":       confidence_label(confidence),
                "description": "Weighted similarity of retrieved chunks to query"
            },
            "precision_at_k": {
                "score":             precision,
                "relevant_retrieved": rel_retrieved,
                "k":                 k,
                "threshold":         RELEVANCE_THRESHOLD,
                "description":       f"{rel_retrieved}/{k} chunks above "
                                     f"similarity threshold {RELEVANCE_THRESHOLD}"
            },
            "recall_at_k": {
                "score":            recall,
                "relevant_retrieved": rel_ret,
                "total_relevant":   total_rel,
                "description":      f"{rel_ret} of {total_rel} relevant "
                                    f"corpus chunks retrieved"
            },
            "semantic_similarity": {
                "score":       semantic_sim,
                "description": "Cosine similarity between query and answer embeddings"
            },
            "faithfulness": {
                "score":            faithfulness,
                "label":            faithfulness_label(faithfulness),
                "sentence_scores":  sentence_scores,
                "description":      "How grounded the answer is in retrieved context"
            }
        },
        "sources":           answer_data.get("sources", []),
        "retrieved_chunks":  retrieved_chunks
    }

    return report


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_report(report):
    """Pretty-print evaluation report."""
    metrics = report["metrics"]

    print(f"\n{'='*65}")
    print(f"EVALUATION REPORT")
    print(f"{'='*65}")
    print(f"Query  : {report['query'][:70]}")
    print(f"Model  : {report['model']}")
    print(f"{'─'*65}")

    # Confidence
    c = metrics["confidence_score"]
    print(f"\n1. Confidence Score   : {c['score']:.4f}  [{c['label']}]")
    print(f"   {c['description']}")

    # Precision
    p = metrics["precision_at_k"]
    print(f"\n2. Precision@{p['k']}        : {p['score']:.4f}  "
          f"({p['relevant_retrieved']}/{p['k']} relevant)")
    print(f"   {p['description']}")

    # Recall
    r = metrics["recall_at_k"]
    print(f"\n3. Recall@{p['k']}           : {r['score']:.4f}  "
          f"({r['relevant_retrieved']}/{r['total_relevant']} "
          f"relevant corpus chunks retrieved)")
    print(f"   {r['description']}")

    # Semantic Similarity
    s = metrics["semantic_similarity"]
    print(f"\n4. Semantic Similarity: {s['score']:.4f}")
    print(f"   {s['description']}")

    # Faithfulness
    f = metrics["faithfulness"]
    print(f"\n5. Faithfulness       : {f['score']:.4f}  [{f['label']}]")
    print(f"   {f['description']}")

    print(f"\n{'─'*65}")
    print(f"SOURCES ({len(report['sources'])} papers):")
    for s in report["sources"]:
        print(f"  {s['paper_id']} — {s['section']} "
              f"(relevance: {s['score']:.4f})")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Step 7 — RAG Evaluation Metrics")
    print("=" * 65)

    # Load answer from Step 6
    answer_data = load_json(ANSWER_FILE)
    if not answer_data:
        print(f"No answer found. Run step6_llm_answer.py first.")
        return

    print(f"Loaded answer from '{ANSWER_FILE}'")
    print(f"  Query  : {answer_data.get('query', '')[:70]}")
    print(f"  Model  : {answer_data.get('model', '')}")
    print(f"  Chunks : {len(answer_data.get('retrieved_chunks', []))}")

    # Load embedding model
    print(f"\nLoading embedding model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded.")

    # Run evaluation
    report = evaluate(answer_data, model)

    # Display
    display_report(report)

    # Save
    save_json(report, OUTPUT_FILE)
    print(f"Evaluation report saved -> '{OUTPUT_FILE}'")
   


if __name__ == "__main__":
    main()