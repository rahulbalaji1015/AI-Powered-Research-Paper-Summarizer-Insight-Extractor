"""
Step 6 — Generate answers using Groq API (llama3 quality, free tier).

Takes retrieved chunks from Step 5 and passes them to
Groq with a carefully designed prompt to generate
answers strictly from the research paper content.

Covers pipeline steps 8 and 9:
  8. Pass retrieved chunks + query to LLM for answer generation
  9. Generate answer strictly from retrieved research paper content
"""

import json
import os
import time
from groq import Groq

# ── Settings ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx"     
MODEL_NAME   = "llama-3.1-8b-instant"         # free, fast, high quality
TEMPERATURE  = 0.1                      # low = more factual answers
MAX_TOKENS   = 1024                     # max response tokens

# ── Input / Output ────────────────────────────────────────────────────────────
RETRIEVED_FILE = "retrieved_chunks.json"
OUTPUT_FILE    = "rag_answer.json"

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K     = 5
MIN_SCORE = 0.2

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


# ---------------------------------------------------------------------------
# Groq setup
# ---------------------------------------------------------------------------

def setup_groq():
    """Initialize Groq client."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq client initialized — model: '{MODEL_NAME}'")
        return client
    except Exception as e:
        print(f"ERROR setting up Groq: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query, retrieved_chunks):
    """
    Build a RAG prompt that:
      - Provides retrieved context with source labels
      - Instructs LLM to answer ONLY from provided context
      - Requests paper ID references in the answer
    """
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        paper_id = chunk.get("paper_id",   "")
        section  = chunk.get("section",    "")
        text     = chunk.get("text",       "").strip()
        score    = chunk.get("score",      0.0)

        context_lines.append(
            f"[Source {i} | Paper: {paper_id} | "
            f"Section: {section} | Relevance: {score:.2f}]\n{text}"
        )

    context_block = "\n\n".join(context_lines)

    prompt = f"""You are a research assistant specialized in analyzing research papers on content moderation and social media.

Answer the question using ONLY the information provided in the context below.
- If the answer is found in the context, answer clearly and mention which Paper ID(s) the information comes from.
- If the context does not contain enough information, respond with: "The provided papers do not contain enough information to answer this question."
- Do NOT use any outside knowledge beyond what is given.

CONTEXT FROM RESEARCH PAPERS:
{context_block}

QUESTION: {query}

ANSWER (cite Paper IDs where relevant):"""

    return prompt


# ---------------------------------------------------------------------------
# Groq call
# ---------------------------------------------------------------------------

def call_groq(client, prompt, retries=3):
    """
    Send prompt to Groq and return generated response.
    Automatically retries on rate limit errors with backoff.
    """
    print(f"Calling Groq ({MODEL_NAME}) ...")

    for attempt in range(1, retries + 1):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS
            )
            elapsed = time.time() - start
            answer  = response.choices[0].message.content.strip()
            print(f"  Response generated in {elapsed:.1f}s")

            # Groq returns token usage — helpful for monitoring
            usage = response.usage
            print(f"  Tokens — prompt: {usage.prompt_tokens} | "
                  f"completion: {usage.completion_tokens} | "
                  f"total: {usage.total_tokens}")
            return answer, elapsed

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait = 30 * attempt
                print(f"  Rate limit hit (attempt {attempt}/{retries}). "
                      f"Waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"ERROR calling Groq: {e}")
                print("  Possible causes:")
                print("    - Invalid API key")
                print("    - No internet connection")
                return None, 0

    print(f"  All {retries} attempts failed.")
    return None, 0


# ---------------------------------------------------------------------------
# Source extractor
# ---------------------------------------------------------------------------

def extract_sources(retrieved_chunks):
    """Build deduplicated list of source papers used in the answer."""
    sources = []
    seen    = set()
    for chunk in retrieved_chunks:
        pid = chunk.get("paper_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            sources.append({
                "paper_id":    pid,
                "section":     chunk.get("section",     ""),
                "source_type": chunk.get("source_type", ""),
                "score":       chunk.get("score",        0.0)
            })
    return sources


# ---------------------------------------------------------------------------
# Main answer generation function
# ---------------------------------------------------------------------------

def generate_answer(query, retrieved_chunks, groq_client, save_output=True):
    """
    Full answer generation pipeline:
      1. Build RAG prompt
      2. Call Groq LLM
      3. Return structured result with answer + sources

    Can be imported by full RAG pipeline:
      from step6_llm_answer import generate_answer
    """
    if not retrieved_chunks:
        print("No retrieved chunks provided.")
        return None

    print(f"\nBuilding prompt from {len(retrieved_chunks)} chunks ...")
    prompt        = build_prompt(query, retrieved_chunks)
    prompt_tokens = len(prompt.split())
    print(f"  Prompt length : {prompt_tokens} tokens")

    # Call Groq
    answer, elapsed = call_groq(groq_client, prompt)
    if not answer:
        return None

    # Build structured result
    sources = extract_sources(retrieved_chunks)
    result  = {
        "query":            query,
        "answer":           answer,
        "model":            MODEL_NAME,
        "generation_time":  round(elapsed, 2),
        "prompt_tokens":    prompt_tokens,
        "sources":          sources,
        "retrieved_chunks": retrieved_chunks
    }

    if save_output:
        save_json(result, OUTPUT_FILE)
        print(f"Answer saved -> '{OUTPUT_FILE}'")

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_answer(result):
    """Pretty-print the generated answer."""
    if not result:
        return

    print(f"\n{'='*65}")
    print(f"QUERY  : {result['query']}")
    print(f"MODEL  : {result['model']}")
    print(f"TIME   : {result['generation_time']}s")
    print(f"{'='*65}")
    print(f"\nANSWER :\n{result['answer']}")
    print(f"\n{'─'*65}")
    print(f"SOURCES ({len(result['sources'])} papers):")
    for s in result["sources"]:
        print(f"  {s['paper_id']} — {s['section']} "
              f"(relevance: {s['score']:.4f})")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main — interactive mode
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("RAG Answer Generation — Groq API (llama3-8b)")
    print("=" * 65)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Temperature : {TEMPERATURE}")
    print(f"  Max tokens  : {MAX_TOKENS}")
    print()

    # Setup Groq client
    groq_client = setup_groq()
    if not groq_client:
        return

    # Load FAISS + embedding model for interactive mode
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from query_retrieval import load_faiss_store, embed_query, retrieve_chunks

        index, metadata = load_faiss_store()
        if index is None:
            return
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print()

        # Interactive query loop
        print("Enter your query (type 'quit' to exit):\n")
        while True:
            query = input("Query: ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Exiting.")
                break

            # Embed + retrieve
            query_vec = embed_query(embed_model, query)
            chunks    = retrieve_chunks(index, metadata, query_vec, TOP_K)

            if not chunks:
                print("No relevant chunks found. Try rephrasing.\n")
                continue

            # Generate answer
            result = generate_answer(query, chunks, groq_client)
            display_answer(result)

    except ImportError:
        # Fallback: use saved retrieved_chunks.json from Step 5
        print("Running in offline mode — using saved retrieved_chunks.json\n")
        saved = load_json(RETRIEVED_FILE)
        if not saved:
            print("No retrieved chunks found. Run step5_query_retrieval.py first.")
            return

        result = generate_answer(
            saved["query"],
            saved["results"],
            groq_client
        )
        display_answer(result)


if __name__ == "__main__":
    main()