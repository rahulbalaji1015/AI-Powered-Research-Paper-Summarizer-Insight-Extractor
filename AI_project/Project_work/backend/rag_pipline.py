"""
Step 8 — Complete RAG Pipeline (End-to-End).

Combines all steps into a single script:
  1. Load FAISS index + metadata
  2. Load embedding model
  3. Accept user query
  4. Embed query
  5. Retrieve top-k chunks from FAISS
  6. Generate answer using Groq (llama-3.1-8b-instant)
  7. Compute confidence score + evaluation metrics
  8. Return final output with answer, sources, confidence, metrics

Output: final_output.json
"""

import json
import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Settings ──────────────────────────────────────────────────────────────────
GROQ_API_KEY  = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx"      # ← paste your Groq API key
LLM_MODEL     = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
TEMPERATURE   = 0.1
MAX_TOKENS    = 1024
TOP_K         = 5
MIN_SCORE     = 0.2
REL_THRESHOLD = 0.5                      # relevance threshold for precision/recall

# ── File paths ────────────────────────────────────────────────────────────────
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE    = "faiss_metadata.json"
CHUNKS_FILE      = "chunk_embeddings.json"
OUTPUT_FILE      = "final_output.json"
HISTORY_FILE     = "query_history.json"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


# ---------------------------------------------------------------------------
# Resource loader — loads once, reused for every query
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Full RAG pipeline loaded once and reused for multiple queries.
    Avoids reloading models and indexes on every query.
    """

    def __init__(self):
        self.embed_model  = None
        self.groq_client  = None
        self.faiss_index  = None
        self.metadata     = None
        self.all_chunks   = None
        self.query_history = []

    def load(self):
        """Load all resources — called once at startup."""
        print("=" * 65)
        print("Loading RAG Pipeline resources ...")
        print("=" * 65)

        # 1. Embedding model
        print(f"\n[1/4] Loading embedding model '{EMBED_MODEL}' ...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"  Done — dim: "
              f"{self.embed_model.get_sentence_embedding_dimension()}")

        # 2. FAISS index
        print(f"\n[2/4] Loading FAISS index '{FAISS_INDEX_FILE}' ...")
        if not os.path.exists(FAISS_INDEX_FILE):
            print(f"  ERROR: FAISS index not found. Run step4_faiss_store.py")
            return False
        self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        self.metadata    = load_json(METADATA_FILE)
        print(f"  Done — {self.faiss_index.ntotal} vectors loaded")

        # 3. All chunks (for recall computation)
        print(f"\n[3/4] Loading chunk embeddings '{CHUNKS_FILE}' ...")
        self.all_chunks = load_json(CHUNKS_FILE)
        if self.all_chunks:
            print(f"  Done — {len(self.all_chunks)} chunks loaded")
        else:
            print(f"  WARNING: chunk embeddings not found — "
                  f"Recall@k will be skipped")

        # 4. Groq client
        print(f"\n[4/4] Initializing Groq client '{LLM_MODEL}' ...")
        try:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"  Done.")
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

        print(f"\nAll resources loaded. Pipeline ready.\n")
        return True

    # ── Step 4: Embed query ───────────────────────────────────────────────────

    def embed_query(self, query):
        embedding = self.embed_model.encode(
            [query],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        return embedding.astype(np.float32)

    # ── Step 5: Retrieve chunks ───────────────────────────────────────────────

    def retrieve(self, query_embedding):
        scores, indices = self.faiss_index.search(query_embedding, k=TOP_K)
        results = []
        for rank, (score, idx) in enumerate(
                zip(scores[0], indices[0]), 1):
            if idx == -1 or score < MIN_SCORE:
                continue
            chunk = self.metadata[idx]
            results.append({
                "rank":        rank,
                "score":       float(round(score, 4)),
                "chunk_id":    chunk["chunk_id"],
                "paper_id":    chunk["paper_id"],
                "section":     chunk["section"],
                "source_type": chunk["source_type"],
                "token_count": chunk["token_count"],
                "text":        chunk["text"]
            })
        return results

    # ── Step 6: Generate answer ───────────────────────────────────────────────

    def build_prompt(self, query, chunks):
        context_lines = []
        for i, chunk in enumerate(chunks, 1):
            context_lines.append(
                f"[Source {i} | Paper: {chunk['paper_id']} | "
                f"Section: {chunk['section']} | "
                f"Relevance: {chunk['score']:.2f}]\n{chunk['text']}"
            )
        context_block = "\n\n".join(context_lines)

        return f"""You are a research assistant specialized in analyzing research papers on content moderation and social media.

Answer the question using ONLY the information provided in the context below.
- If the answer is found in the context, answer clearly and mention which Paper ID(s) the information comes from.
- If the context does not contain enough information, respond with: "The provided papers do not contain enough information to answer this question."
- Do NOT use any outside knowledge beyond what is given.

CONTEXT FROM RESEARCH PAPERS:
{context_block}

QUESTION: {query}

ANSWER (cite Paper IDs where relevant):"""

    def generate_answer(self, query, chunks, retries=3):
        prompt = self.build_prompt(query, chunks)

        for attempt in range(1, retries + 1):
            start = time.time()
            try:
                response = self.groq_client.chat.completions.create(
                    model       = LLM_MODEL,
                    messages    = [{"role": "user", "content": prompt}],
                    temperature = TEMPERATURE,
                    max_tokens  = MAX_TOKENS
                )
                elapsed = time.time() - start
                answer  = response.choices[0].message.content.strip()
                tokens  = response.usage.total_tokens
                return answer, elapsed, tokens, len(prompt.split())

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait = 30 * attempt
                    print(f"  Rate limit — waiting {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"  Groq error: {e}")
                    return None, 0, 0, 0

        return None, 0, 0, 0

    # ── Step 7: Evaluation metrics ────────────────────────────────────────────

    def compute_metrics(self, query, answer, chunks, query_embedding):
        metrics = {}

        # 1. Confidence Score
        scores  = [c["score"] for c in chunks]
        weights = [1 / (i + 1) for i in range(len(scores))]
        conf    = (sum(s * w for s, w in zip(scores, weights))
                   / sum(weights)) if weights else 0.0
        metrics["confidence_score"] = {
            "score": round(conf, 4),
            "label": ("Very High" if conf >= 0.8 else
                      "High"      if conf >= 0.6 else
                      "Medium"    if conf >= 0.4 else "Low")
        }

        # 2. Precision@k
        relevant  = sum(1 for c in chunks
                        if c["score"] >= REL_THRESHOLD)
        precision = relevant / TOP_K if TOP_K > 0 else 0.0
        metrics["precision_at_k"] = {
            "score":              round(precision, 4),
            "relevant_retrieved": relevant,
            "k":                  TOP_K
        }

        # 3. Recall@k
        if self.all_chunks:
            total_relevant = sum(
                1 for ch in self.all_chunks
                if ch.get("embedding") and
                cosine_similarity(query_embedding[0],
                                  ch["embedding"]) >= REL_THRESHOLD
            )
            recall = (relevant / total_relevant
                      if total_relevant > 0 else 0.0)
            metrics["recall_at_k"] = {
                "score":            round(recall, 4),
                "relevant_retrieved": relevant,
                "total_relevant":   total_relevant
            }
        else:
            metrics["recall_at_k"] = {"score": None,
                                       "note":  "chunk_embeddings.json not found"}

        # 4. Semantic Similarity
        embeddings = self.embed_model.encode(
            [query, answer],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        sem_sim = cosine_similarity(embeddings[0], embeddings[1])
        metrics["semantic_similarity"] = {"score": round(sem_sim, 4)}

        # 5. Faithfulness
        sentences = [s.strip() for s in
                     answer.replace("\n", " ").split(".")
                     if len(s.strip()) > 20]
        if sentences:
            chunk_texts  = [c["text"] for c in chunks]
            all_texts    = sentences + chunk_texts
            embs         = self.embed_model.encode(
                all_texts,
                normalize_embeddings = True,
                convert_to_numpy     = True
            )
            sent_embs    = embs[:len(sentences)]
            chunk_embs   = embs[len(sentences):]
            sent_scores  = [
                max(cosine_similarity(se, ce) for ce in chunk_embs)
                for se in sent_embs
            ]
            faith = float(np.mean(sent_scores))
        else:
            faith = 0.0

        metrics["faithfulness"] = {
            "score": round(faith, 4),
            "label": ("Highly Faithful"    if faith >= 0.8 else
                      "Faithful"           if faith >= 0.6 else
                      "Partially Faithful" if faith >= 0.4 else
                      "Low — possible hallucination")
        }

        return metrics

    # ── Main query function ───────────────────────────────────────────────────

    def query(self, user_query):
        """
        Run the full RAG pipeline for one query.
        Returns structured final output.
        """
        print(f"\n{'='*65}")
        print(f"Processing query: {user_query[:65]}")
        print(f"{'='*65}")
        pipeline_start = time.time()

        # Step 4: Embed query
        print("  [4] Embedding query ...")
        query_embedding = self.embed_query(user_query)

        # Step 5: Retrieve
        print(f"  [5] Retrieving top-{TOP_K} chunks ...")
        chunks = self.retrieve(query_embedding)
        if not chunks:
            print("  No relevant chunks found. Try rephrasing.")
            return None
        print(f"      Retrieved {len(chunks)} chunks")

        # Step 6: Generate answer
        print(f"  [6] Generating answer with {LLM_MODEL} ...")
        answer, gen_time, total_tokens, prompt_tokens = \
            self.generate_answer(user_query, chunks)
        if not answer:
            return None
        print(f"      Done in {gen_time:.1f}s ({total_tokens} tokens)")

        # Step 7: Evaluate
        print(f"  [7] Computing evaluation metrics ...")
        metrics = self.compute_metrics(
            user_query, answer, chunks, query_embedding
        )

        total_time = round(time.time() - pipeline_start, 2)

        # Step 8: Build final output
        sources = list({c["paper_id"]: c for c in chunks}.values())
        output  = {
            "query":           user_query,
            "answer":          answer,
            "model":           LLM_MODEL,
            "total_time_sec":  total_time,
            "generation_time": round(gen_time, 2),
            "prompt_tokens":   prompt_tokens,
            "total_tokens":    total_tokens,
            "sources":         [
                {
                    "paper_id":    s["paper_id"],
                    "section":     s["section"],
                    "source_type": s["source_type"],
                    "score":       s["score"]
                }
                for s in sources
            ],
            "metrics":          metrics,
            "retrieved_chunks": chunks
        }

        # Save output
        save_json(output, OUTPUT_FILE)

        # Append to history
        self.query_history.append({
            "query":      user_query,
            "answer":     answer[:200] + "..." if len(answer) > 200 else answer,
            "confidence": metrics["confidence_score"]["score"],
            "faithfulness": metrics["faithfulness"]["score"],
            "time":       total_time
        })
        save_json(self.query_history, HISTORY_FILE)

        return output

    # ── Display ───────────────────────────────────────────────────────────────

    def display(self, output):
        if not output:
            return
        m = output["metrics"]
        print(f"\n{'='*65}")
        print(f"ANSWER:")
        print(f"{output['answer']}")
        print(f"\n{'─'*65}")
        print(f"SOURCES ({len(output['sources'])} papers):")
        for s in output["sources"]:
            print(f"  {s['paper_id']} — {s['section']:<20} "
                  f"score={s['score']:.4f}")
        print(f"\n{'─'*65}")
        print(f"METRICS:")
        print(f"  Confidence Score   : "
              f"{m['confidence_score']['score']:.4f}  "
              f"[{m['confidence_score']['label']}]")
        print(f"  Precision@{TOP_K}        : "
              f"{m['precision_at_k']['score']:.4f}  "
              f"({m['precision_at_k']['relevant_retrieved']}/{TOP_K} relevant)")
        r = m["recall_at_k"]
        if r.get("score") is not None:
            print(f"  Recall@{TOP_K}           : "
                  f"{r['score']:.4f}  "
                  f"({r['relevant_retrieved']}/{r['total_relevant']} retrieved)")
        print(f"  Semantic Similarity: "
              f"{m['semantic_similarity']['score']:.4f}")
        print(f"  Faithfulness       : "
              f"{m['faithfulness']['score']:.4f}  "
              f"[{m['faithfulness']['label']}]")
        print(f"\n  Total pipeline time: {output['total_time_sec']}s")
        print(f"  Output saved      : '{OUTPUT_FILE}'")
        print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main — interactive loop
# ---------------------------------------------------------------------------

def main():
    pipeline = RAGPipeline()
    if not pipeline.load():
        print("Pipeline failed to load. Check errors above.")
        return

    print("RAG Pipeline ready. Enter your queries below.")
    print("Commands: 'history' — show past queries | 'quit' — exit\n")

    while True:
        query = input("Query: ").strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print(f"Session ended. {len(pipeline.query_history)} "
                  f"queries saved to '{HISTORY_FILE}'")
            break

        if query.lower() == "history":
            if not pipeline.query_history:
                print("No queries yet.\n")
            else:
                print(f"\nQuery history ({len(pipeline.query_history)}):")
                for i, h in enumerate(pipeline.query_history, 1):
                    print(f"  {i}. {h['query'][:60]}")
                    print(f"     confidence={h['confidence']} | "
                          f"faithfulness={h['faithfulness']} | "
                          f"time={h['time']}s")
                print()
            continue

        output = pipeline.query(query)
        pipeline.display(output)


if __name__ == "__main__":
    main()