"""
Research Paper RAG Assistant — v4
New in this version:
  1. Memory — remembers past Q&A, references them when similar questions asked
  2. Output mode selector — RAG / AI Analysis / Additional / About Paper / All Combined
  3. Full combined output metrics — confidence + faithfulness on entire response
  4. Vibrant UI — faster, more responsive, better colors
  5. Strict irrelevant detection — general knowledge questions always flagged

Folder structure:
    project/
    ├── config.py          ← edit API keys + Neo4j password here
    ├── backend/           ← chat_db.py, pdf_export.py, upload.py, steps ...
    ├── frontend/          ← this file (app.py)
    └── data/              ← all .bin .json .db .npy files (auto-created)
"""

import streamlit as st
import json
import os
import re
import sys
import time
import uuid
import numpy as np
import faiss
import tempfile

# ── Path setup: add project root and backend/ to sys.path ────────────────────
# Works correctly no matter which directory you run `streamlit run` from.
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))  # frontend/
_ROOT         = os.path.dirname(_FRONTEND_DIR)               # project root
_BACKEND_DIR  = os.path.join(_ROOT, "backend")

for _p in [_ROOT, _BACKEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── All imports now resolve from root and backend/ ────────────────────────────
import config

from sentence_transformers import SentenceTransformer
from upload import full_process_paper
from groq import Groq
import chat_db as db
from pdf_export import generate_pdf

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# ── Settings — pulled from config.py (edit keys there, not here) ──────────────
GROQ_API_KEY         = config.GROQ_API_KEY
LLM_MODEL            = config.LLM_MODEL
EMBED_MODEL          = config.EMBED_MODEL
TEMPERATURE          = config.TEMPERATURE
MAX_TOKENS           = config.MAX_TOKENS
TOP_K                = config.TOP_K
MIN_SCORE            = config.MIN_SCORE
REL_THRESHOLD        = config.REL_THRESHOLD
IRRELEVANT_THRESHOLD = config.IRRELEVANT_THRESHOLD
MEMORY_SIMILARITY    = config.MEMORY_SIMILARITY
MEMORY_MAX           = config.MEMORY_MAX

# ── File paths — all absolute, from config.py ────────────────────────────────
FAISS_INDEX_FILE = config.FAISS_INDEX_FILE
METADATA_FILE    = config.METADATA_FILE
CHUNKS_FILE      = config.CHUNKS_FILE
MEMORY_FILE      = config.MEMORY_FILE

# ── Irrelevant topic keywords — always flagged regardless of score ─────────────
GENERAL_KNOWLEDGE_PATTERNS = [
    r"\bwho is (the )?(father|mother|founder|inventor|creator|president|"
    r"prime minister|king|queen|leader|ceo|director)\b",
    r"\bwhat is (the )?(capital|currency|population|area|flag|national)\b",
    r"\bwhen (was|did|were|is)\b.*(born|died|founded|invented|established|war|battle)\b",
    r"\bhow (tall|old|far|long|much|many) (is|are|was|were)\b",
    r"\bwho (won|lost|scored|played|invented|discovered|ruled)\b",
    r"\bwhat (year|date|day|month) (was|did|is|are)\b",
    r"\brecipe\b|\bcooking\b|\bfood\b|\bweather\b|\bsports\b|\bcricket\b"
    r"|\bfootball\b|\bbasketball\b|\bmovie\b|\bsong\b|\bmusic\b",
    r"\bfunny\b|\bjoke\b|\blaugh\b|\bpolitics\b|\belection\b",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Research RAG Assistant",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""

<style>
    /* ── Global — light background for readability ── */
    .stApp { background: #f0f2f8; }
    .stApp p, .stApp li, .stApp span { color: #1a1a2e; }

    section[data-testid="stSidebar"] { background: #1e1b4b !important; }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #e0e7ff !important; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #c4b5fd !important; }

    /* ── Answer blocks — WHITE background, dark text ── */

    .rag-box {
        background: #ffffff;
        border-left: 6px solid #7c3aed;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
        color: #1a1a2e;
        line-height: 1.8;
        font-size: 15px;
        box-shadow: 0 2px 16px #7c3aed22;
    }

    .ai-box {
        background: #ffffff;
        border-left: 6px solid #0891b2;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
        color: #1a1a2e;
        line-height: 1.8;
        font-size: 15px;
        box-shadow: 0 2px 16px #0891b222;
    }

    .extra-box {
        background: #ffffff;
        border-left: 6px solid #d97706;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
        color: #1a1a2e;
        line-height: 1.8;
        font-size: 15px;
        box-shadow: 0 2px 16px #d9770622;
    }

    .about-box {
        background: #ffffff;
        border-left: 6px solid #059669;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
        color: #1a1a2e;
        line-height: 1.8;
        font-size: 15px;
        box-shadow: 0 2px 16px #05966922;
    }

    .combined-box {
        background: #ffffff;
        border: 2px solid #7c3aed;
        border-radius: 12px;
        padding: 22px 26px;
        margin: 10px 0;
        color: #1a1a2e;
        line-height: 1.9;
        font-size: 15px;
        box-shadow: 0 4px 20px #7c3aed22;
    }

    .combined-box h3 {
        color: #7c3aed;
        margin-top: 18px;
        margin-bottom: 6px;
        font-size: 15px;
        font-weight: 700;
        border-bottom: 1px solid #e9e3ff;
        padding-bottom: 4px;
    }

    .memory-box {
        background: #fdf4ff;
        border-left: 6px solid #a855f7;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #4a1d6b;
        font-size: 13px;
        box-shadow: 0 2px 8px #a855f722;
    }

    .irrelevant-box {
        background: #fff1f2;
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #7f1d1d;
        font-size: 14px;
        box-shadow: 0 2px 8px #ef444422;
    }

    .warning-box {
        background: #fffbeb;
        border-left: 6px solid #f59e0b;
        border-radius: 10px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #78350f;
        font-size: 13px;
    }

    /* ── Section labels ── */

    .section-label {
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 16px 0 5px 0;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }

    .label-rag   { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; }
    .label-ai    { background:#e0f7fa; color:#0e7490; border:1px solid #67e8f9; }
    .label-extra { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
    .label-about { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
    .label-mem   { background:#fce7f3; color:#831843; border:1px solid #f9a8d4; }
    .label-combined { background:#ede9fe; color:#4c1d95; border:1px solid #7c3aed; }

    /* ── Source chips ── */

    .source-chip {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px 3px;
        border: 1px solid #c4b5fd;
        font-weight: 600;
    }

    /* ── Chat input ── */

    div[data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: 2px solid #7c3aed44 !important;
        color: #1a1a2e !important;
        border-radius: 12px !important;
        font-size: 14px !important;
    }

    /* ── Text input ── */

    div[data-testid="stTextInput"] input {
        background: #f5f3ff !important;
        border: 1px solid #c4b5fd !important;
        color: #1a1a2e !important;
        border-radius: 8px !important;
    }

    /* ── Chat messages ── */

    div[data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #f0ebff !important;
        border-radius: 12px !important;
        margin: 6px 0 !important;
        padding: 4px 0 !important;
    }

    /* ── Title gradient ── */

    h1 {
        background: linear-gradient(90deg, #7c3aed, #0891b2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* ───────────────── VISIBILITY FIXES FOR SIDEBAR CONTROLS ───────────────── */

    /* Select paper / Load session dropdown */

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #e5e7eb !important;
        color: #000000 !important;
        border: 1px solid #9ca3af !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* Dropdown menu */

    div[role="listbox"] {
        background: #f3f4f6 !important;
    }

    div[role="option"] {
        color: #000000 !important;
    }

    /* File uploader */

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        background: #e5e7eb !important;
        border-radius: 10px !important;
        padding: 8px !important;
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] span {
        color: #000000 !important;
    }

    /* Rename session input */

    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
        background: #e5e7eb !important;
        color: #000000 !important;
        border: 1px solid #9ca3af !important;
    }

    /* Delete paper / session buttons */

    section[data-testid="stSidebar"] .stButton > button {
        background: #d1d5db !important;
        color: #000000 !important;
        border: 1px solid #9ca3af !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #9ca3af !important;
        color: #000000 !important;
    }

    hr { border-color: #e9d5ff !important; }

</style>

""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_FILE):
        return None, None
    return faiss.read_index(FAISS_INDEX_FILE), load_json(METADATA_FILE)


def load_all_chunks():
    return load_json(CHUNKS_FILE) or []


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


def cosine_sim(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0


def get_available_papers():
    meta = load_json(METADATA_FILE)
    if not meta:
        return []
    return sorted(set(m["paper_id"] for m in meta))


def get_paper_context(metadata, paper_id, max_chars=5000):
    chunks   = [m for m in metadata if m["paper_id"] == paper_id]
    priority = {"section": 0, "uploaded": 0, "relationships": 1,
                "metadata": 2, "citations": 3}
    chunks   = sorted(chunks, key=lambda c: priority.get(c["source_type"], 9))
    ctx      = ""
    for c in chunks:
        block = f"\n[{c['section']} | {c['source_type']}]\n{c['text']}\n"
        if len(ctx) + len(block) > max_chars:
            break
        ctx += block
    return ctx.strip()


def call_groq(client, prompt, max_tokens=MAX_TOKENS, stop_key=None):
    if stop_key and st.session_state.get(stop_key):
        return None
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
        if stop_key and st.session_state.get(stop_key):
            return None
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"


# ---------------------------------------------------------------------------
# Memory system
# ---------------------------------------------------------------------------

def load_memory():
    return load_json(MEMORY_FILE) or []


def save_memory(memory):
    save_json(memory[-MEMORY_MAX:], MEMORY_FILE)


def add_to_memory(query, combined_answer, paper_id, embed_model):
    """Add a Q&A pair to persistent memory with embedding."""
    memory = load_memory()
    emb    = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].tolist()
    memory.append({
        "query":      query,
        "answer":     combined_answer[:800],
        "paper_id":   paper_id or "all",
        "embedding":  emb,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M")
    })
    save_memory(memory)


def find_similar_memory(query, embed_model, threshold=MEMORY_SIMILARITY):
    """Find past Q&A pairs similar to the current query."""
    memory = load_memory()
    if not memory:
        return []

    q_emb = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]

    matches = []
    for item in memory:
        if not item.get("embedding"):
            continue
        sim = cosine_sim(q_emb, item["embedding"])
        if sim >= threshold:
            matches.append({**item, "similarity": round(sim, 4)})

    return sorted(matches, key=lambda x: x["similarity"], reverse=True)[:3]


# ---------------------------------------------------------------------------
# Irrelevant detection
# ---------------------------------------------------------------------------

def is_general_knowledge(query):
    """Check if query matches known off-topic patterns."""
    q = query.lower().strip()
    for pattern in GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def is_irrelevant(chunks, query):
    """Two-stage irrelevance check."""
    # Stage 1: pattern match for general knowledge
    if is_general_knowledge(query):
        return True, "general_knowledge"
    # Stage 2: low FAISS score
    if not chunks:
        return True, "no_results"
    if max(c["score"] for c in chunks) < IRRELEVANT_THRESHOLD:
        return True, "low_score"
    return False, None


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def embed_query(model, query):
    return model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


def retrieve_chunks(index, metadata, query_vec,
                    top_k=TOP_K, paper_filter=None):
    fetch_k  = min(top_k * 6 if paper_filter else top_k * 2, index.ntotal)
    scores, indices = index.search(query_vec, k=fetch_k)
    results  = []
    seen     = set()
    for sc, idx in zip(scores[0], indices[0]):
        if idx == -1 or sc < MIN_SCORE:
            continue
        chunk = metadata[idx]
        if paper_filter and chunk["paper_id"] != paper_filter:
            continue
        cid = chunk["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        results.append({
            "score":       float(round(sc, 4)),
            "chunk_id":    cid,
            "paper_id":    chunk["paper_id"],
            "section":     chunk["section"],
            "source_type": chunk["source_type"],
            "token_count": chunk["token_count"],
            "text":        chunk["text"]
        })
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_rag_prompt(query, chunks, paper_filter, memory_context=""):
    context = "\n\n".join(
        f"[Source {i} | Paper:{c['paper_id']} | "
        f"Section:{c['section']} | Score:{c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    note = f" Focus on paper {paper_filter}." if paper_filter else ""
    mem  = (f"\nRELEVANT PREVIOUS ANSWER:\n{memory_context}\n"
            if memory_context else "")
    return f"""You are a research assistant.{note}
Answer using ONLY the research context below. Cite Paper IDs.{mem}
If not found say: "The papers do not contain enough information on this."

RESEARCH CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def build_ai_prompt(query, paper_ctx, paper_id, memory_context=""):
    mem = (f"\nPREVIOUS RELATED ANSWER:\n{memory_context}\n"
           if memory_context else "")
    return f"""You are an expert analyst for paper {paper_id}.{mem}
Using the paper below, provide deep analytical insight on the question.
Go beyond the text — explain mechanisms, implications, and connections.

PAPER CONTENT:
{paper_ctx[:3000]}

QUESTION: {query}

Provide 2-3 analytical paragraphs with academic depth."""


def build_additional_prompt(query, chunks, paper_id, memory_context=""):
    ctx = "\n".join(c["text"][:300] for c in chunks[:4])
    mem = (f"\nPREVIOUS CONTEXT:\n{memory_context}\n"
           if memory_context else "")
    scope = f"paper {paper_id}" if paper_id else "these papers"
    return f"""Based on {scope} research content, provide broader knowledge.{mem}

CONTEXT:
{ctx}

TOPIC: {query}

Provide:
1. Related concepts and theories
2. Real-world implications
3. Broader field connections
4. Key fact or statistic relevant to this topic

3-4 concise paragraphs."""


def build_about_prompt(paper_ctx, paper_id, query, memory_context=""):
    mem = (f"\nPREVIOUS RELATED INFO:\n{memory_context}\n"
           if memory_context else "")
    return f"""Provide information about paper {paper_id} relevant to this question.{mem}

PAPER CONTENT:
{paper_ctx[:3500]}

QUESTION: {query}

Provide:
**Overview**: 2-sentence summary of the paper
**Relevance to question**: How this paper addresses the topic (2-3 sentences)
**Key details**: Specific methods, datasets, results mentioned (3-4 sentences)
**Authors & context**: Publication context if mentioned"""


def build_overview_prompt(ctx, pid):
    return f"""Comprehensive overview of paper {pid}.

{ctx}

**Title & Authors**: (from content)
**Objective**: (2 sentences)
**Problem**: (2 sentences)
**Approach**: (3 sentences)
**Findings**: (3 sentences)
**Significance**: (2 sentences)"""


def build_terms_prompt(ctx, pid):
    return f"""Extract 10 key technical terms from paper {pid}.

{ctx}

Format EXACTLY:
TERM: [term]
DEFINITION: [1-2 sentence simple definition]"""


def build_insights_prompt(ctx, pid):
    return f"""Deep insights for paper {pid}.

{ctx}

**KEY CONTRIBUTIONS**: (4 items)
**LIMITATIONS**: (3-4 items)
**FUTURE DIRECTIONS**: (4 items)
**REAL-WORLD APPLICATIONS**: (4 items)
**VS PRIOR WORK**: (2-3 sentences)"""


def build_summary_prompt(ctx, pid):
    return f"""Academic summary of paper {pid}.

{ctx}

Write 4-5 flowing paragraphs:
1. Introduction & motivation
2. Methodology
3. Results & findings
4. Conclusions & contributions
5. Limitations & future work"""


def build_single_combined_prompt(query, chunks, paper_ctx,
                                  paper_id, memory_context=""):
    """One unified prompt → one flowing answer with 4 clear sections."""
    rag_ctx = "\n\n".join(
        f"[Paper:{c['paper_id']} | {c['section']} | score:{c['score']:.2f}]\n{c['text']}"
        for c in chunks[:5]
    )
    mem  = f"\nPREVIOUS RELATED ANSWER:\n{memory_context}\n" if memory_context else ""
    note = f" You are analysing paper {paper_id}." if paper_id else ""
    return f"""You are an expert research assistant.{note}{mem}

Write ONE comprehensive flowing answer to the question using ALL sources below.
Use ### headings to separate the four parts naturally:

### Answer from the Research Papers
(Answer directly from retrieved content — cite Paper IDs)

### In-Depth Analysis
(Expert analytical insight — explain mechanisms, implications, context)

### Broader Knowledge & Connections
(Related concepts, real-world applications, field trends)

### About This Paper
(Paper objective, approach, key findings relevant to the question)

RETRIEVED CHUNKS:
{rag_ctx}

FULL PAPER CONTEXT:
{paper_ctx[:2000] if paper_ctx else "Not available"}

QUESTION: {query}

Write one unified, thorough, academic answer:"""


def build_compare_prompt(ctxs, pids):
    secs = "\n\n".join(
        f"=== {pid} ===\n{ctx[:1200]}" for pid, ctx in zip(pids, ctxs)
    )
    return f"""Compare papers side by side:

{secs}

**Objectives** | **Approaches** | **Datasets** |
**Results** | **Strengths** | **Weaknesses** | **Best Paper & Why**"""


# ---------------------------------------------------------------------------
# Metrics — computed on FULL combined output
# ---------------------------------------------------------------------------

def compute_metrics(query, full_answer, chunks, query_vec, all_chunks):
    """
    Compute all 5 metrics on the FULL combined answer.
    This gives higher faithfulness since the full answer
    draws from more sources.
    """
    em = load_embed_model()

    # 1. Confidence — weighted average of retrieval scores
    scores  = [c["score"] for c in chunks]
    weights = [1 / (i + 1) for i in range(len(scores))]
    conf    = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # 2. Precision@k
    relevant  = sum(1 for c in chunks if c["score"] >= REL_THRESHOLD)
    precision = relevant / TOP_K

    # 3. Recall@k
    recall_score = None
    recall_rel   = 0
    recall_total = 0
    if all_chunks:
        recall_total = sum(
            1 for ch in all_chunks
            if ch.get("embedding") and
            cosine_sim(query_vec[0], ch["embedding"]) >= REL_THRESHOLD
        )
        if recall_total > 0:
            recall_score = round(relevant / recall_total, 4)
            recall_rel   = relevant

    # 4. Semantic Similarity — query vs full combined answer
    embs = em.encode(
        [query, full_answer[:1000]],
        normalize_embeddings=True, convert_to_numpy=True
    )
    sem_sim = cosine_sim(embs[0], embs[1])

    # 5. Faithfulness — on FULL combined answer sentences
    # More sentences from more sources → higher faithfulness
    sentences = [s.strip() for s in
                 full_answer.replace("\n", " ").split(".")
                 if len(s.strip()) > 20][:25]   # cap at 25 sentences

    if sentences and chunks:
        chunk_texts = [c["text"] for c in chunks]
        all_texts   = sentences + chunk_texts
        embs2       = em.encode(all_texts,
                         normalize_embeddings=True, convert_to_numpy=True)
        s_embs = embs2[:len(sentences)]
        c_embs = embs2[len(sentences):]
        faith  = float(np.mean([
            max(cosine_sim(se, ce) for ce in c_embs)
            for se in s_embs
        ]))
    else:
        faith = 0.0

    return {
        "confidence": {
            "score": round(conf, 4),
            "label": ("Very High" if conf >= 0.8 else
                      "High"      if conf >= 0.6 else
                      "Medium"    if conf >= 0.4 else "Low")
        },
        "precision": {
            "score": round(precision, 4),
            "relevant": relevant, "k": TOP_K
        },
        "recall": {
            "score": recall_score,
            "relevant": recall_rel,
            "total": recall_total
        },
        "semantic_similarity": {"score": round(sem_sim, 4)},
        "faithfulness": {
            "score": round(faith, 4),
            "label": ("Highly Faithful"    if faith >= 0.8 else
                      "Faithful"           if faith >= 0.6 else
                      "Partially Faithful" if faith >= 0.4 else
                      "Low — check answer")
        }
    }


def render_metrics(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🎯 Confidence",
                  f"{m['confidence']['score']:.2f}",
                  m['confidence']['label'])
    with c2:
        p = m["precision"]
        st.metric(f"📌 Precision@{p['k']}",
                  f"{p['score']:.2f}",
                  f"{p['relevant']}/{p['k']} relevant")
    with c3:
        r = m["recall"]
        if r.get("score") is not None:
            st.metric(f"🔁 Recall@{TOP_K}",
                      f"{r['score']:.2f}",
                      f"{r['relevant']}/{r['total']}")
        else:
            st.metric(f"🔁 Recall@{TOP_K}", "N/A")
    with c4:
        st.metric("🔗 Semantic Sim.",
                  f"{m['semantic_similarity']['score']:.2f}")
    with c5:
        st.metric("✅ Faithfulness",
                  f"{m['faithfulness']['score']:.2f}",
                  m['faithfulness']['label'])


def render_sources(chunks):
    seen, html = set(), ""
    for c in chunks:
        if c["paper_id"] not in seen:
            seen.add(c["paper_id"])
            html += (f'<span class="source-chip">'
                     f'📄 {c["paper_id"]} — {c["section"]} '
                     f'({c["score"]:.2f})</span>')
    if html:
        st.markdown(f'<div style="margin:8px 0 4px 0">{html}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Delete paper
# ---------------------------------------------------------------------------

def delete_paper(paper_id):
    meta   = load_json(METADATA_FILE) or []
    chunks = load_json(CHUNKS_FILE)   or []
    m2     = [x for x in meta   if x["paper_id"] != paper_id]
    c2     = [x for x in chunks if x["paper_id"] != paper_id]
    if len(m2) == len(meta):
        return False, "Paper not found."
    removed = len(meta) - len(m2)
    if c2:
        embs  = np.array([x["embedding"] for x in c2], dtype=np.float32)
        idx   = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        faiss.write_index(idx, FAISS_INDEX_FILE)
        for i, x in enumerate(m2):
            x["faiss_id"] = i
        np.save(config.EMBEDDINGS_FILE, embs)   # ← absolute path from config
    else:
        faiss.write_index(faiss.IndexFlatIP(384), FAISS_INDEX_FILE)
    save_json(m2, METADATA_FILE)
    save_json(c2, CHUNKS_FILE)
    return True, f"Removed {removed} chunks for {paper_id}."


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

COMMANDS = {
    "/overview": "Full structured overview of selected paper",
    "/terms":    "10 key terms with definitions",
    "/insights": "Contributions, limitations, future directions",
    "/summary":  "Full academic paragraph summary",
    "/compare":  "Compare all available papers",
    "/memory":   "Show remembered Q&A pairs",
    "/help":     "List all commands"
}

def detect_command(q):
    q = q.strip().lower()
    for cmd in COMMANDS:
        if q.startswith(cmd):
            return cmd
    return None


# ---------------------------------------------------------------------------
# Core query processor
# ---------------------------------------------------------------------------

def process_query(query, index, metadata, all_chunks,
                  embed_model, groq_client,
                  paper_filter, output_mode, stop_key):
    """
    Runs full pipeline and returns structured result dict.
    output_mode: "🔀 All Combined" | "📄 RAG Only" | "🤖 AI Analysis" |
                 "💡 Additional"  | "📋 About Paper"
    """
    command = detect_command(query)

    # /memory command
    if command == "/memory":
        mem = load_memory()
        return {"type": "memory_list", "memory": mem}

    # /help
    if command == "/help":
        return {"type": "help"}

    # Paper commands
    if command in ("/overview", "/terms", "/insights", "/summary", "/compare"):
        if command != "/compare" and not paper_filter:
            return {"type": "error",
                    "message": "Select a specific paper from the sidebar."}
        if command == "/compare":
            papers = get_available_papers()[:4]
            ctxs   = [get_paper_context(metadata, p) for p in papers]
            ans    = call_groq(groq_client, build_compare_prompt(ctxs, papers),
                               max_tokens=1400, stop_key=stop_key)
            return {"type": "compare", "answer": ans, "papers": papers}

        ctx = get_paper_context(metadata, paper_filter, max_chars=5000)
        prompts = {
            "/overview": build_overview_prompt(ctx, paper_filter),
            "/terms":    build_terms_prompt(ctx, paper_filter),
            "/insights": build_insights_prompt(ctx, paper_filter),
            "/summary":  build_summary_prompt(ctx, paper_filter),
        }
        ans = call_groq(groq_client, prompts[command],
                        max_tokens=1400, stop_key=stop_key)
        if command == "/terms":
            return {"type": "terms", "raw": ans, "paper_id": paper_filter}
        return {"type": "command", "command": command,
                "answer": ans, "paper_id": paper_filter}

    # ── Normal question ──────────────────────────────────────────────────────

    # Stage 1: Check for general knowledge / off-topic
    irrelevant, reason = is_irrelevant([], query)
    if reason == "general_knowledge":
        return {
            "type":    "irrelevant",
            "reason":  "general_knowledge",
            "message": (
                "This question is not related to research papers. "
                "Please ask something specific to the papers in the knowledge base."
            )
        }

    # Stage 2: Retrieve chunks
    query_vec = embed_query(embed_model, query)
    chunks    = retrieve_chunks(index, metadata, query_vec,
                                paper_filter=paper_filter)

    # Stage 3: Check FAISS score irrelevance
    irrelevant, reason = is_irrelevant(chunks, query)
    if reason in ("no_results", "low_score") and reason != "general_knowledge":
        # Still try to answer but flag it
        pass

    # Stage 4: Check memory for similar past questions
    memory_matches = find_similar_memory(query, embed_model)
    memory_context = ""
    if memory_matches:
        best = memory_matches[0]
        if best["similarity"] >= MEMORY_SIMILARITY:
            memory_context = (
                f"[From previous session — similarity {best['similarity']:.0%}]\n"
                f"Q: {best['query']}\nA: {best['answer']}"
            )

    if not chunks and not memory_context:
        return {
            "type":    "irrelevant",
            "reason":  "no_results",
            "message": "No relevant content found in the knowledge base."
        }

    if st.session_state.get(stop_key):
        return {"type": "stopped"}

    # ── Generate outputs based on mode ────────────────────────────────────────
    rag_answer  = None
    ai_context  = None
    additional  = None
    about_paper = None

    needs_rag        = output_mode in ("🔀 All Combined", "📄 RAG Only")
    needs_ai         = output_mode in ("🔀 All Combined", "🤖 AI Analysis") and paper_filter
    needs_additional = output_mode in ("🔀 All Combined", "💡 Additional")
    needs_about      = output_mode in ("🔀 All Combined", "📋 About Paper") and paper_filter
    needs_single     = output_mode == "📝 Single Combined"

    # ── Single Combined — one LLM call, one flowing answer ────────────────────
    if needs_single:
        paper_ctx = get_paper_context(metadata, paper_filter, max_chars=2500) \
                    if paper_filter else ""
        single_answer = call_groq(
            groq_client,
            build_single_combined_prompt(
                query, chunks, paper_ctx,
                paper_filter, memory_context
            ),
            max_tokens=1400, stop_key=stop_key
        )
        if not single_answer or st.session_state.get(stop_key):
            return {"type": "stopped"}
        metrics = compute_metrics(
            query, single_answer, chunks, query_vec, all_chunks
        )
        if single_answer:
            add_to_memory(query, single_answer, paper_filter, embed_model)
        return {
            "type":           "single_combined",
            "query":          query,
            "single_answer":  single_answer,
            "chunks":         chunks,
            "metrics":        metrics,
            "paper_id":       paper_filter,
            "memory_context": memory_context,
            "memory_matches": memory_matches,
            "low_relevance":  irrelevant
        }

    if chunks and needs_rag:
        rag_answer = call_groq(
            groq_client,
            build_rag_prompt(query, chunks, paper_filter, memory_context),
            stop_key=stop_key
        )
        if st.session_state.get(stop_key):
            return {"type": "stopped"}

    if needs_ai and paper_filter:
        paper_ctx = get_paper_context(metadata, paper_filter, max_chars=3500)
        ai_context = call_groq(
            groq_client,
            build_ai_prompt(query, paper_ctx, paper_filter, memory_context),
            max_tokens=700, stop_key=stop_key
        )
        if st.session_state.get(stop_key):
            return {"type": "stopped"}

    if chunks and needs_additional:
        additional = call_groq(
            groq_client,
            build_additional_prompt(query, chunks, paper_filter, memory_context),
            max_tokens=600, stop_key=stop_key
        )
        if st.session_state.get(stop_key):
            return {"type": "stopped"}

    if needs_about and paper_filter:
        paper_ctx2 = get_paper_context(metadata, paper_filter, max_chars=3500)
        about_paper = call_groq(
            groq_client,
            build_about_prompt(paper_ctx2, paper_filter, query, memory_context),
            max_tokens=600, stop_key=stop_key
        )
        if st.session_state.get(stop_key):
            return {"type": "stopped"}

    # Build full combined answer for metrics
    parts = [p for p in [rag_answer, ai_context, additional, about_paper] if p]
    full_answer = " ".join(parts) if parts else "No answer generated."

    # Compute metrics on FULL combined output
    metrics = compute_metrics(
        query, full_answer, chunks or [], query_vec, all_chunks
    ) if chunks else None

    # Save to memory
    if full_answer and full_answer != "No answer generated.":
        add_to_memory(query, full_answer, paper_filter, embed_model)

    return {
        "type":           "full",
        "query":          query,
        "rag_answer":     rag_answer,
        "ai_context":     ai_context,
        "additional":     additional,
        "about_paper":    about_paper,
        "memory_context": memory_context,
        "memory_matches": memory_matches,
        "chunks":         chunks,
        "metrics":        metrics,
        "paper_id":       paper_filter,
        "output_mode":    output_mode,
        "low_relevance":  irrelevant
    }


# ---------------------------------------------------------------------------
# Render result
# ---------------------------------------------------------------------------

def render_result(result):
    rtype = result.get("type")

    if rtype == "help":
        st.markdown("**📖 Available Commands**")
        for cmd, desc in COMMANDS.items():
            st.markdown(
                f'<span class="cmd-chip">{cmd}</span> — {desc}',
                unsafe_allow_html=True
            )
        return

    if rtype == "memory_list":
        mem = result.get("memory", [])
        if not mem:
            st.info("No memory stored yet. Ask some questions first.")
        else:
            st.markdown(f"**🧠 Memory — {len(mem)} stored Q&A pairs**")
            for i, item in enumerate(reversed(mem[-10:]), 1):
                with st.expander(f"{i}. {item['query'][:70]} "
                                 f"[{item.get('timestamp','')}]"):
                    st.caption(f"Paper: {item.get('paper_id','all')}")
                    st.write(item["answer"])
        return

    if rtype == "error":
        st.warning(result["message"])
        return

    if rtype == "stopped":
        st.info("⏹ Generation stopped.")
        return

    if rtype == "irrelevant":
        reason = result.get("reason", "")
        if reason == "general_knowledge":
            st.markdown(
                '<div class="irrelevant-box">'
                '🚫 <b>Off-Topic Question</b><br>'
                'This is a general knowledge question, not related to the '
                'research papers in the system. '
                'Please ask about the research topics, methods, findings, '
                'or content within the papers.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="irrelevant-box">'
                '⚠️ <b>No Relevant Content Found</b><br>'
                f'{result["message"]}'
                '</div>',
                unsafe_allow_html=True
            )
        return

    if rtype == "single_combined":
        # Memory reference
        if result.get("memory_context") and result.get("memory_matches"):
            best = result["memory_matches"][0]
            st.markdown(
                f'<div class="memory-box">🧠 <b>Memory Reference</b> — '
                f'Similar question found ({best["similarity"]:.0%} match) '
                f'from {best.get("timestamp","earlier")}:<br>'
                f'<i>"{best["query"][:100]}"</i></div>',
                unsafe_allow_html=True
            )
        if result.get("low_relevance"):
            st.markdown(
                '<div class="warning-box">⚠️ Low relevance — '
                'showing best available content.</div>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<span class="section-label label-combined">'
            '📝 Single Combined Answer</span>',
            unsafe_allow_html=True
        )
        answer = result.get("single_answer", "")
        st.markdown(
            f'<div class="combined-box">{answer}</div>',
            unsafe_allow_html=True
        )
        if result.get("chunks"):
            render_sources(result["chunks"])
        if result.get("metrics"):
            with st.expander("📊 Evaluation Metrics", expanded=False):
                render_metrics(result["metrics"])
        if result.get("chunks"):
            with st.expander(
                    f"🔍 {len(result['chunks'])} Retrieved Chunks",
                    expanded=False):
                for i, c in enumerate(result["chunks"], 1):
                    st.markdown(
                        f"**{i}. {c['paper_id']} — {c['section']} "
                        f"| score: {c['score']:.3f}**"
                    )
                    st.caption(c["text"][:250] + "...")
                    if i < len(result["chunks"]):
                        st.divider()
        return

    if rtype == "compare":
        st.markdown("### 📊 Paper Comparison")
        st.markdown(
            f'<div class="ai-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )
        return

    if rtype == "terms":
        st.markdown(f"### 📚 Key Terms — {result['paper_id']}")
        blocks = re.findall(
            r"TERM:\s*(.+?)\nDEFINITION:\s*(.+?)(?=\nTERM:|\Z)",
            result["raw"], re.DOTALL
        )
        if blocks:
            for term, defn in blocks:
                st.markdown(
                    f'<div class="term-card">'
                    f'<div class="term-word">🔑 {term.strip()}</div>'
                    f'<div class="term-def">{defn.strip()}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.write(result["raw"])
        return

    if rtype in ("command", "full"):
        # Memory reference
        mem_ctx = result.get("memory_context", "")
        if mem_ctx and result.get("memory_matches"):
            best = result["memory_matches"][0]
            st.markdown(
                f'<div class="memory-box">'
                f'🧠 <b>Memory Reference</b> — '
                f'Similar question found ({best["similarity"]:.0%} match) '
                f'from {best.get("timestamp","earlier")}:<br>'
                f'<i>"{best["query"][:100]}"</i>'
                f'</div>',
                unsafe_allow_html=True
            )

        if result.get("low_relevance"):
            st.markdown(
                '<div class="warning-box">'
                '⚠️ Low relevance score — answer may be partially outside '
                'paper scope. Showing best available content.'
                '</div>',
                unsafe_allow_html=True
            )

        if result.get("rag_answer"):
            st.markdown(
                '<span class="section-label label-rag">'
                '📄 Answer from Research Papers (RAG)'
                '</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="rag-box">{result["rag_answer"]}</div>',
                unsafe_allow_html=True
            )
            if result.get("chunks"):
                render_sources(result["chunks"])

        if result.get("ai_context"):
            st.markdown(
                '<span class="section-label label-ai">'
                '🤖 AI Analysis & Deep Context'
                '</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="ai-box">{result["ai_context"]}</div>',
                unsafe_allow_html=True
            )

        if result.get("additional"):
            st.markdown(
                '<span class="section-label label-extra">'
                '💡 Additional Knowledge & Field Connections'
                '</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="extra-box">{result["additional"]}</div>',
                unsafe_allow_html=True
            )

        if result.get("about_paper"):
            st.markdown(
                '<span class="section-label label-about">'
                '📋 About This Paper'
                '</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="about-box">{result["about_paper"]}</div>',
                unsafe_allow_html=True
            )

        if result.get("metrics"):
            with st.expander("📊 Evaluation Metrics (on full combined output)",
                             expanded=False):
                render_metrics(result["metrics"])

        if result.get("chunks"):
            with st.expander(
                    f"🔍 {len(result['chunks'])} Retrieved Chunks",
                    expanded=False):
                for i, c in enumerate(result["chunks"], 1):
                    st.markdown(
                        f"**{i}. {c['paper_id']} — {c['section']} "
                        f"| {c['source_type']} | score: {c['score']:.3f}**"
                    )
                    st.caption(c["text"][:250] + "...")
                    if i < len(result["chunks"]):
                        st.divider()


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _make_pdf_bytes(session_id, message_ids):
    """Build PDF bytes. Returns (bytes, filename) or (None, error_string)."""
    session_info = db.get_session(session_id)
    if not session_info:
        return None, "Session not found."
    msgs = (db.get_selected_messages(message_ids)
            if message_ids
            else db.get_session_messages(session_id))
    if not msgs:
        return None, "No messages to export yet."
    try:
        pdf_bytes = generate_pdf(session_info, msgs)
        safe  = re.sub(r"[^a-zA-Z0-9_-]", "_", session_info["title"][:30])
        fname = f"chat_{safe}_{time.strftime('%Y%m%d_%H%M')}.pdf"
        return pdf_bytes, fname
    except Exception as e:
        return None, str(e)


def pdf_download_button(session_id, message_ids,
                        label="📥 Export as PDF", key_suffix=""):
    """Render a PDF download button.
    key_suffix MUST be unique across all calls on the same page render.
    """
    pdf_bytes, result = _make_pdf_bytes(session_id, message_ids)
    if pdf_bytes is None:
        st.caption(f"PDF: {result}")
        return
    # Fully unique key: session + suffix + hash of first 64 bytes of content
    uid = abs(hash(pdf_bytes[:64])) % 999983
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=result,
        mime="application/pdf",
        use_container_width=True,
        key=f"dl_{session_id[:8]}_{key_suffix}_{uid}"
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # ── DB init ───────────────────────────────────────────────────────────────
    db.init_db()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔬 Research Paper RAG Assistant")
    st.caption("Knowledge Graph · FAISS Retrieval · Groq AI · Memory · SQLite")

    # ── Session state init ────────────────────────────────────────────────────
    defaults = {
        "chat_history":    [],
        "stop_generation": False,
        "is_generating":   False,
        "session_id":      None,
        "session_title":   "New Chat",
        "msg_ids":         [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    stop_key = "stop_generation"

    # ── Auto-create a DB session if none exists ───────────────────────────────
    if not st.session_state.session_id:
        sid = str(uuid.uuid4())[:8]
        st.session_state.session_id    = sid
        st.session_state.session_title = f"Chat {time.strftime('%b %d %H:%M')}"
        db.create_session(sid, st.session_state.session_title)

    current_sid = st.session_state.session_id

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")

        # Paper selection
        available    = get_available_papers()
        selected     = st.selectbox("📂 Select paper:", ["All Papers"] + available)
        paper_filter = None if selected == "All Papers" else selected

        st.divider()

        # Output mode
        st.markdown("**🎛 Output Mode**")
        output_mode = st.radio(
            "mode",
            ["🔀 All Combined", "📝 Single Combined",
             "📄 RAG Only", "🤖 AI Analysis",
             "💡 Additional", "📋 About Paper"],
            label_visibility="collapsed"
        )

        st.divider()

        # ── Session management ────────────────────────────────────────────────
        st.markdown("**💬 Chat Sessions**")

        if st.button("➕ New Chat", use_container_width=True):
            sid   = str(uuid.uuid4())[:8]
            title = f"Chat {time.strftime('%b %d %H:%M')}"
            db.create_session(sid, title, paper_filter)
            st.session_state.session_id    = sid
            st.session_state.session_title = title
            st.session_state.chat_history  = []
            st.session_state.msg_ids       = []
            st.rerun()

        all_sessions = db.get_all_sessions()
        if all_sessions:
            session_labels = {
                s["session_id"]: (
                    f"{s['title']} "
                    f"({s['message_count']} msgs · {s['updated_at'][:10]})"
                )
                for s in all_sessions
            }
            chosen_label = st.selectbox(
                "Load session:",
                options=list(session_labels.values()),
                index=list(session_labels.keys()).index(current_sid)
                      if current_sid in session_labels else 0,
                key="session_select"
            )
            chosen_sid = next(
                (k for k, v in session_labels.items() if v == chosen_label),
                current_sid
            )
            if chosen_sid != current_sid:
                msgs = db.get_session_messages(chosen_sid)
                sess = db.get_session(chosen_sid)
                st.session_state.session_id    = chosen_sid
                st.session_state.session_title = sess["title"] if sess else "Loaded"
                st.session_state.msg_ids       = [m["message_id"] for m in msgs]
                st.session_state.chat_history  = [
                    {
                        "query":      m["query"],
                        "message_id": m["message_id"],
                        "result": {
                            "type":          "full" if m.get("rag_answer") else
                                             "single_combined" if m.get("single_answer") else
                                             "command",
                            "query":         m["query"],
                            "rag_answer":    m.get("rag_answer"),
                            "ai_context":    m.get("ai_context"),
                            "additional":    m.get("additional"),
                            "about_paper":   m.get("about_paper"),
                            "single_answer": m.get("single_answer"),
                            "answer":        m.get("command_answer"),
                            "chunks":        [],
                            "metrics":       m.get("metrics"),
                            "paper_id":      m.get("paper_id"),
                            "output_mode":   m.get("output_mode"),
                        }
                    }
                    for m in msgs
                ]
                st.rerun()

        with st.expander("✏️ Rename / Delete Session"):
            new_name = st.text_input("New name:",
                                     value=st.session_state.session_title,
                                     key="rename_inp")
            if st.button("💾 Save Name", use_container_width=True):
                db.rename_session(current_sid, new_name)
                st.session_state.session_title = new_name
                st.success("Renamed!")
                st.rerun()
            if st.button("🗑️ Delete This Session",
                         type="primary", use_container_width=True):
                db.delete_session(current_sid)
                st.session_state.session_id    = None
                st.session_state.chat_history  = []
                st.session_state.msg_ids       = []
                st.rerun()

        st.divider()

        # ── PDF Export (sidebar) ──────────────────────────────────────────────
        st.markdown("**📄 Export as PDF**")
        export_scope = st.radio(
            "Export scope:",
            ["Entire Session", "Select Messages"],
            key="export_scope",
            label_visibility="collapsed"
        )

        if export_scope == "Select Messages" and st.session_state.msg_ids:
            all_msgs    = db.get_session_messages(current_sid)
            msg_options = {
                m["message_id"]: f"Q{i}: {m['query'][:50]}..."
                for i, m in enumerate(all_msgs, 1)
            }
            selected_labels = st.multiselect(
                "Pick messages:",
                options=list(msg_options.values()),
                key="pdf_msg_sel"
            )
            selected_mids = [
                mid for mid, lbl in msg_options.items()
                if lbl in selected_labels
            ]
            if selected_mids:
                pdf_download_button(
                    current_sid, selected_mids,
                    label="📥 Download Selected PDF",
                    key_suffix="sidebar_sel"
                )
            else:
                st.caption("Select messages above then download.")
        else:
            if st.session_state.msg_ids:
                pdf_download_button(
                    current_sid, None,
                    label="📥 Download Full Session PDF",
                    key_suffix="sidebar_full"
                )
            else:
                st.caption("Ask some questions first.")

        st.divider()

        # Quick actions
        if paper_filter:
            st.markdown("**⚡ Quick Actions**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📋 Overview",  use_container_width=True):
                    st.session_state["pending_query"] = "/overview"
                if st.button("💡 Insights",  use_container_width=True):
                    st.session_state["pending_query"] = "/insights"
            with c2:
                if st.button("📚 Key Terms", use_container_width=True):
                    st.session_state["pending_query"] = "/terms"
                if st.button("📝 Summary",   use_container_width=True):
                    st.session_state["pending_query"] = "/summary"
            if st.button("📊 Compare All",  use_container_width=True):
                st.session_state["pending_query"] = "/compare"
            if st.button("🧠 Show Memory",  use_container_width=True):
                st.session_state["pending_query"] = "/memory"
            st.divider()

        # Upload paper
        st.markdown("**📤 Upload New Paper**")
        pid_in   = st.text_input("Paper ID:", placeholder="P11", key="uid")
        title_in = st.text_input("Title (optional):", key="utitle")
        up_file  = st.file_uploader("File:", type=["pdf","docx","txt"], key="ufile")
        if up_file and pid_in:
            if st.button("➕ Add to Knowledge Base", use_container_width=True):
                if not os.path.exists(FAISS_INDEX_FILE):
                    st.error("FAISS index not found.")
                else:
                    pid   = pid_in.strip().upper()
                    title = title_in.strip() or up_file.name
                    log   = []
                    pbox  = st.empty()
                    pbar  = st.progress(0)
                    steps = ["Load","Extract","Sections","Entities",
                             "Relations","Neo4j","Chunks","Embed","Done"]
                    def cb(msg):
                        log.append(msg)
                        pbar.progress(int(
                            min(len(log), len(steps)-1)/len(steps)*100))
                        pbox.info("\n".join(log[-3:]))
                    em = load_embed_model()
                    ok, summ, err = full_process_paper(up_file, pid, title, em, cb)
                    pbar.progress(100)
                    if ok:
                        pbox.empty()
                        st.success(f"✅ {pid} added! {summ['chunks']} chunks indexed.")
                        st.rerun()
                    else:
                        st.error(f"Failed: {err}")

        st.divider()

        # Delete paper
        st.markdown("**🗑️ Delete Paper**")
        del_sel = st.selectbox("Select:", ["— select —"] + available, key="dsel")
        if del_sel != "— select —":
            n = len([m for m in (load_json(METADATA_FILE) or [])
                     if m["paper_id"] == del_sel])
            st.caption(f"{n} chunks will be removed")
            if st.button(f"Delete {del_sel}", type="primary",
                         use_container_width=True):
                with st.spinner("Deleting ..."):
                    ok, msg = delete_paper(del_sel)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

        # Stats
        st.markdown("**📈 Stats**")
        db_stats = db.get_db_stats()
        mem      = load_memory()
        c1, c2 = st.columns(2)
        c1.metric("DB Sessions",  db_stats["sessions"])
        c2.metric("DB Messages",  db_stats["messages"])
        c1.metric("Memories",     len(mem))
        c2.metric("This Session", len(st.session_state.chat_history))

        full_h = [h for h in st.session_state.chat_history
                  if h["result"].get("metrics")]
        if full_h:
            scores = [h["result"]["metrics"]["confidence"]["score"]
                      for h in full_h]
            st.metric("Avg Confidence", f"{np.mean(scores):.2f}")

        ca, cb_ = st.columns(2)
        with ca:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with cb_:
            if st.button("🧹 Clear Memory", use_container_width=True):
                save_json([], MEMORY_FILE)
                st.rerun()

        with st.expander("📖 Commands"):
            for cmd, desc in COMMANDS.items():
                st.markdown(
                    f'<span class="cmd-chip">{cmd}</span> {desc}<br>',
                    unsafe_allow_html=True
                )

    # ── Load resources ────────────────────────────────────────────────────────
    if not os.path.exists(FAISS_INDEX_FILE):
        st.error("⚠️ FAISS index not found. Run steps 1–4 first.")
        return

    embed_model     = load_embed_model()
    index, metadata = load_faiss_index()
    all_chunks      = load_all_chunks()
    groq_client     = load_groq_client()

    if index is None:
        st.error("Failed to load FAISS index.")
        return

    # ── Mode + session indicator ──────────────────────────────────────────────
    mode_colors = {
        "🔀 All Combined":    "#7c3aed",
        "📝 Single Combined": "#4f46e5",
        "📄 RAG Only":        "#3b82f6",
        "🤖 AI Analysis":     "#0891b2",
        "💡 Additional":      "#d97706",
        "📋 About Paper":     "#059669",
    }
    color = mode_colors.get(output_mode, "#7c3aed")
    scope = f"paper {paper_filter}" if paper_filter else "all papers"
    st.markdown(
        f'<div style="background:{color}15;border:1px solid {color}44;'
        f'border-radius:8px;padding:8px 16px;margin-bottom:10px;'
        f'display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:{color};font-weight:700;font-size:13px;">'
        f'{output_mode} · {scope}</span>'
        f'<span style="color:#6b7280;font-size:12px;">'
        f'Session: {st.session_state.session_title}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Quick suggestions ─────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("**💬 Quick Start — click or type:**")
        suggestions = [
            "What methods are used for content moderation?",
            "What are the key findings?",
            "What datasets were used?",
            "How does this compare to prior work?",
        ]
        cols = st.columns(4)
        for i, s in enumerate(suggestions):
            if cols[i].button(s, use_container_width=True, key=f"sug{i}"):
                st.session_state["pending_query"] = s

    # ── Chat history ──────────────────────────────────────────────────────────
    # PDF buttons are rendered OUTSIDE st.chat_message() to avoid Streamlit
    # widget-inside-container key collision bugs.
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            render_result(chat["result"])
        # PDF button lives outside the chat bubble
        mid = chat.get("message_id")
        if mid:
            col_pdf, col_gap = st.columns([1, 5])
            with col_pdf:
                pdf_download_button(
                    current_sid, [mid],
                    label="📥 PDF",
                    key_suffix=f"hist_{idx}_{mid[:6]}"
                )

    # ── Stop button ───────────────────────────────────────────────────────────
    if st.session_state.get("is_generating"):
        if st.button("⏹ Stop Generation", type="primary"):
            st.session_state[stop_key] = True
            st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    pending = st.session_state.pop("pending_query", None)
    scope_l = f"paper {paper_filter}" if paper_filter else "all papers"
    query   = st.chat_input(
        f"Ask about {scope_l} — or type /help"
    ) or pending

    if query:
        st.session_state[stop_key]       = False
        st.session_state["is_generating"] = True

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("⚡ Processing ..."):
                result = process_query(
                    query, index, metadata, all_chunks,
                    embed_model, groq_client,
                    paper_filter, output_mode, stop_key
                )
            render_result(result)

        # ── Save to DB ────────────────────────────────────────────────────────
        message_id     = None
        saveable_types = {"full", "single_combined", "command",
                          "compare", "terms"}
        if result.get("type") in saveable_types:
            result["query"]       = query
            result["output_mode"] = output_mode
            message_id = str(uuid.uuid4())[:12]
            db.save_message(current_sid, message_id, result)
            if "msg_ids" not in st.session_state:
                st.session_state.msg_ids = []
            st.session_state.msg_ids.append(message_id)

        st.session_state["is_generating"] = False
        st.session_state[stop_key]        = False

        st.session_state.chat_history.append({
            "query":      query,
            "result":     result,
            "message_id": message_id,
        })

        # ── PDF button for the new answer (outside chat bubble) ───────────────
        if message_id:
            col_pdf, col_gap = st.columns([1, 5])
            with col_pdf:
                pdf_download_button(
                    current_sid, [message_id],
                    label="📥 Export as PDF",
                    key_suffix=f"new_{message_id}"
                )


if __name__ == "__main__":
    main()