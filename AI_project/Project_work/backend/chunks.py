"""
Step 2 — Convert graph documents into plain text chunks.

For each paper, builds structured text from:
  - Metadata    (title, authors, year, venue, domain, keywords)
  - Abstract entities
  - Each section's raw_text + entities
  - Relationship triples
  - Citations

Then splits into chunks of 300-500 tokens with:
  - chunk_id   : unique identifier  (P01_chunk_001)
  - paper_id   : source paper
  - section    : which section the chunk came from
  - source_type: metadata / section / relationships / citations
  - text        : the actual chunk text
"""

import json
import os
import re

# ── Input / Output ────────────────────────────────────────────────────────────
INPUT_FILE  = "graph_documents.json"
OUTPUT_FILE = "text_chunks.json"

# Chunk size controls
TARGET_TOKENS    = 400   # target chunk size in tokens
MAX_TOKENS       = 500   # hard max — split if exceeded
MIN_TOKENS       = 50    # minimum — don't create tiny orphan chunks
OVERLAP_TOKENS   = 50    # overlap between consecutive chunks from same section

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_tokens(text):
    """Approximate token count — split on whitespace."""
    return len(text.split())


def split_into_chunks(text, paper_id, section_name, source_type, chunk_counter):
    """
    Split a long text into overlapping chunks of TARGET_TOKENS.
    Returns list of chunk dicts and updated chunk_counter.
    """
    words  = text.split()
    chunks = []

    if len(words) == 0:
        return chunks, chunk_counter

    # If text is short enough, keep as single chunk
    if len(words) <= MAX_TOKENS:
        if len(words) >= MIN_TOKENS:
            chunk_counter += 1
            chunks.append({
                "chunk_id":    f"{paper_id}_chunk_{chunk_counter:03d}",
                "paper_id":    paper_id,
                "section":     section_name,
                "source_type": source_type,
                "token_count": len(words),
                "text":        text.strip()
            })
        return chunks, chunk_counter

    # Split into overlapping chunks
    start = 0
    while start < len(words):
        end        = min(start + TARGET_TOKENS, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) >= MIN_TOKENS:
            chunk_counter += 1
            chunks.append({
                "chunk_id":    f"{paper_id}_chunk_{chunk_counter:03d}",
                "paper_id":    paper_id,
                "section":     section_name,
                "source_type": source_type,
                "token_count": len(chunk_words),
                "text":        " ".join(chunk_words).strip()
            })

        # Move forward with overlap
        start += TARGET_TOKENS - OVERLAP_TOKENS
        if start >= len(words):
            break

    return chunks, chunk_counter


# ---------------------------------------------------------------------------
# Text builders — one per data type
# ---------------------------------------------------------------------------

def build_metadata_text(doc):
    """Build a clean text block from paper metadata."""
    lines = []
    lines.append(f"Paper ID: {doc['paper_id']}")
    lines.append(f"Title: {doc['title']}")

    if doc.get("authors"):
        lines.append(f"Authors: {', '.join(doc['authors'])}")
    if doc.get("year"):
        lines.append(f"Year: {doc['year']}")
    if doc.get("venue"):
        lines.append(f"Published in: {doc['venue']}")
    if doc.get("domain"):
        lines.append(f"Domain: {doc['domain']}")
    if doc.get("keywords"):
        lines.append(f"Keywords: {', '.join(doc['keywords'])}")

    return "\n".join(lines)


def build_section_text(doc, section):
    """
    Build text for one section.
    Uses raw_text as primary content.
    Appends entity list as supplementary context.
    """
    section_name = section.get("name", "")
    raw_text     = section.get("raw_text", "").strip()
    lines        = []

    if not raw_text:
        return ""

    # Section header
    lines.append(f"Section: {section_name}")
    lines.append(f"Paper: {doc['title']}")
    lines.append("")

    # Raw text content — primary source for RAG
    lines.append(raw_text)

    # Entity context — helps RAG with named entity queries
    section_entities = [
        e for e in doc.get("entities", [])
        if e.get("section_name") == section_name
    ]
    if section_entities:
        unique_types = {}
        for e in section_entities:
            t = e.get("entity_type", "")
            if t not in unique_types:
                unique_types[t] = []
            unique_types[t].append(e.get("entity_text", ""))

        lines.append("")
        lines.append("Key entities:")
        for etype, enames in unique_types.items():
            # Deduplicate and limit to 10 per type
            unique_names = list(dict.fromkeys(enames))[:10]
            lines.append(f"  {etype}: {', '.join(unique_names)}")

    return "\n".join(lines)


def build_relationships_text(doc):
    """
    Build text from NLP relationship triples.
    Groups by section for better context.
    """
    rels = doc.get("relationships", [])
    if not rels:
        return ""

    lines = []
    lines.append(f"Relationships in: {doc['title']}")
    lines.append("")

    # Group by section
    by_section = {}
    for r in rels:
        sec = r.get("source", "general")
        # Clean up source label
        if "::" in sec:
            sec = sec.split("::")[-1]
        by_section.setdefault(sec, []).append(r)

    for sec_name, sec_rels in by_section.items():
        lines.append(f"[{sec_name}]")
        for r in sec_rels[:30]:   # limit per section to avoid bloat
            subj     = r.get("subject",   "")
            relation = r.get("relation",  "")
            obj      = r.get("object",    "")
            sentence = r.get("sentence",  "")
            if sentence:
                lines.append(f"  {sentence}")
            elif subj and relation and obj:
                lines.append(f"  {subj} {relation} {obj}")
        lines.append("")

    return "\n".join(lines)


def build_citations_text(doc):
    """Build text from citation records."""
    citations = doc.get("citations", [])
    if not citations:
        return ""

    lines = []
    lines.append(f"References cited in: {doc['title']}")
    lines.append("")

    for cite in citations:
        ref_title = cite.get("reference_title", "").strip()
        ref_year  = cite.get("cited_year",       "")
        ref_id    = cite.get("reference_id",     "")
        ref_count = cite.get("citation_count",   "")

        if not ref_title:
            continue

        line = f"  [{ref_id}] {ref_title}"
        if ref_year:
            line += f" ({ref_year})"
        if ref_count:
            line += f" — cited {ref_count} times"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def process_document(doc):
    """
    Convert one paper's graph document into a list of text chunks.
    Returns list of chunk dicts.
    """
    paper_id      = doc["paper_id"]
    all_chunks    = []
    chunk_counter = 0

    # ── 1. Metadata chunk (always single chunk) ───────────────────────────────
    meta_text = build_metadata_text(doc)
    if count_tokens(meta_text) >= MIN_TOKENS:
        chunk_counter += 1
        all_chunks.append({
            "chunk_id":    f"{paper_id}_chunk_{chunk_counter:03d}",
            "paper_id":    paper_id,
            "section":     "metadata",
            "source_type": "metadata",
            "token_count": count_tokens(meta_text),
            "text":        meta_text.strip()
        })

    # ── 2. Section chunks (raw_text — primary RAG content) ───────────────────
    for section in doc.get("sections", []):
        sec_text = build_section_text(doc, section)
        if not sec_text or count_tokens(sec_text) < MIN_TOKENS:
            continue

        new_chunks, chunk_counter = split_into_chunks(
            sec_text,
            paper_id,
            section.get("name", ""),
            "section",
            chunk_counter
        )
        all_chunks.extend(new_chunks)

    # ── 3. Relationship chunks ────────────────────────────────────────────────
    rel_text = build_relationships_text(doc)
    if rel_text and count_tokens(rel_text) >= MIN_TOKENS:
        new_chunks, chunk_counter = split_into_chunks(
            rel_text,
            paper_id,
            "relationships",
            "relationships",
            chunk_counter
        )
        all_chunks.extend(new_chunks)

    # ── 4. Citation chunks ────────────────────────────────────────────────────
    cite_text = build_citations_text(doc)
    if cite_text and count_tokens(cite_text) >= MIN_TOKENS:
        new_chunks, chunk_counter = split_into_chunks(
            cite_text,
            paper_id,
            "citations",
            "citations",
            chunk_counter
        )
        all_chunks.extend(new_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    docs = load_json(INPUT_FILE)
    if not docs:
        return

    print(f"Loaded {len(docs)} paper documents from '{INPUT_FILE}'")
    print(f"Settings: target={TARGET_TOKENS} tokens | "
          f"max={MAX_TOKENS} | min={MIN_TOKENS} | overlap={OVERLAP_TOKENS}\n")

    all_chunks = []

    for doc in docs:
        paper_id = doc["paper_id"]
        chunks   = process_document(doc)
        all_chunks.extend(chunks)

        # Per-paper summary
        by_type = {}
        for c in chunks:
            t = c["source_type"]
            by_type[t] = by_type.get(t, 0) + 1

        print(f"  {paper_id} — {doc['title'][:50]}")
        print(f"    Total chunks : {len(chunks)}")
        for stype, count in by_type.items():
            print(f"      {stype:<15} : {count} chunks")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    # Summary
    print(f"\nStep 2 complete -> '{OUTPUT_FILE}'")
    print(f"\nOverall summary:")
    print(f"  Total papers : {len(docs)}")
    print(f"  Total chunks : {len(all_chunks)}")

    by_type = {}
    for c in all_chunks:
        t = c["source_type"]
        by_type[t] = by_type.get(t, 0) + 1
    for stype, count in by_type.items():
        print(f"    {stype:<15} : {count} chunks")

    avg_tokens = sum(c["token_count"] for c in all_chunks) / len(all_chunks)
    print(f"  Avg tokens/chunk : {avg_tokens:.0f}")


if __name__ == "__main__":
    main()