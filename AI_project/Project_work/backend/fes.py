"""
Store raw_text on ALL Section nodes across all 10 papers.

This replaces fix_empty_sections.py — instead of targeting specific papers,
it processes every paper and every valid section in core_data.json.

What it does per section:
  - Stores raw_text, char_count, word_count on the Section node
  - Sets text_only=True if section has 0 entity mentions
  - Re-runs entity extraction for sections with < 5 entities
  - Skips: Paper Details, References, reference continuations, text < 30 chars
"""

import json
import os
import sys
import re
import spacy
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# ── Settings ──────────────────────────────────────────────────────────────────
URI      = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"
DATABASE = "neo4j"

CORE_FILE = "core_data.json"

# Re-extract entities for sections with fewer than this many entities
SPARSE_THRESHOLD = 5

MIN_TEXT_LENGTH = 30

ALWAYS_SKIP = {"paper details", "references"}

VALID_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
    "EVENT", "NORP", "FAC", "LOC", "LANGUAGE",
    "DATE", "PERCENT", "CARDINAL", "QUANTITY", "LAW"
}

nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_reference_continuation(section_name, text):
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r"\[\d+\]", stripped):
        return True
    lines = stripped.splitlines()
    if len(lines) > 2:
        cite_lines = sum(1 for l in lines if re.search(r"\[\d+\]", l))
        if cite_lines / len(lines) > 0.4:
            return True
    if len(stripped) < 200 and stripped[0].islower():
        return True
    if len(stripped) < 500:
        has_citations = bool(re.search(r"\[\d+\]", stripped))
        starts_with_section = stripped.lower().startswith(section_name.strip().lower())
        if has_citations and starts_with_section:
            return True
    return False


def extract_entities(text):
    if not text or len(text.strip()) < 5:
        return []
    doc  = nlp(text[:1_000_000])
    seen = set()
    out  = []
    for ent in doc.ents:
        if ent.label_ not in VALID_ENTITY_TYPES:
            continue
        norm = ent.text.strip().lower()
        key  = (norm, ent.label_)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "text":       ent.text.strip(),
            "normalized": norm,
            "type":       ent.label_
        })
    return out


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

Q_ENSURE_SECTION = """
MERGE (s:Section {paper_id: $paper_id, name: $section_name})
WITH s
MATCH (p:Paper {paper_id: $paper_id})
MERGE (p)-[:HAS_SECTION]->(s)
"""

Q_STORE_RAW_TEXT = """
MATCH (s:Section {paper_id: $paper_id, name: $section_name})
SET s.raw_text   = $raw_text,
    s.char_count = $char_count,
    s.word_count = $word_count,
    s.text_only  = $text_only
"""

Q_GET_ENTITY_COUNT = """
MATCH (s:Section {paper_id: $paper_id, name: $section_name})
OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
RETURN count(e) AS count
"""

Q_ENTITY_SECTION = """
MERGE (e:Entity {normalized: $normalized, type: $type})
SET e.text = $text
WITH e
MATCH (s:Section {paper_id: $paper_id, name: $section_name})
MERGE (s)-[:MENTIONS]->(e)
"""

Q_ALL_SECTIONS_STATUS = """
MATCH (p:Paper)-[:HAS_SECTION]->(s:Section)
RETURN p.paper_id   AS paper_id,
       s.name       AS section,
       count { (s)-[:MENTIONS]->() }  AS entity_count,
       s.raw_text IS NOT NULL         AS has_raw_text,
       s.char_count                   AS char_count
ORDER BY paper_id, section
"""

# ---------------------------------------------------------------------------
# Per-section fix
# ---------------------------------------------------------------------------

def process_section(session, paper_id, section_name, section_text):
    """
    Process one section:
      1. Get current entity count from graph
      2. Store raw_text always
      3. Re-extract entities if count < SPARSE_THRESHOLD
    Returns (status, entities_added)
    """
    cleaned    = section_text.strip()
    char_count = len(cleaned)
    word_count = len(cleaned.split())

    if char_count < MIN_TEXT_LENGTH:
        return "too_short", 0

    if is_reference_continuation(section_name, cleaned):
        return "ref_continuation", 0

    # Ensure section node exists in graph
    session.run(Q_ENSURE_SECTION,
        paper_id     = paper_id,
        section_name = section_name
    )

    # Get current entity count
    result        = session.run(Q_GET_ENTITY_COUNT,
                        paper_id=paper_id, section_name=section_name)
    current_count = result.single()["count"]

    # Re-extract entities if sparse
    entities_added = 0
    if current_count < SPARSE_THRESHOLD:
        entities = extract_entities(cleaned)
        for ent in entities:
            session.run(Q_ENTITY_SECTION,
                paper_id     = paper_id,
                section_name = section_name,
                normalized   = ent["normalized"],
                type         = ent["type"],
                text         = ent["text"]
            )
            entities_added += 1

    # Get updated count after re-extraction
    result        = session.run(Q_GET_ENTITY_COUNT,
                        paper_id=paper_id, section_name=section_name)
    final_count   = result.single()["count"]
    text_only     = final_count == 0

    # Store raw_text on section node
    session.run(Q_STORE_RAW_TEXT,
        paper_id     = paper_id,
        section_name = section_name,
        raw_text     = cleaned,
        char_count   = char_count,
        word_count   = word_count,
        text_only    = text_only
    )

    return "fixed", entities_added

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    core_data = load_json(CORE_FILE)
    if not core_data:
        return

    # Connect
    try:
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to Neo4j — database: '{DATABASE}'\n")
    except ServiceUnavailable:
        print("ERROR: Neo4j not reachable. Start your DBMS in Neo4j Desktop.")
        sys.exit(1)

    # ── Print before state ────────────────────────────────────────────────────
    print("Before state — sections missing raw_text:")
    with driver.session(database=DATABASE) as session:
        rows = session.run(Q_ALL_SECTIONS_STATUS)
        before = [r.data() for r in rows]
    missing = [r for r in before if not r["has_raw_text"]]
    print(f"  {len(missing)} sections missing raw_text out of {len(before)} total\n")

    # ── Process all papers ────────────────────────────────────────────────────
    total_fixed    = 0
    total_skipped  = 0
    total_entities = 0

    for rec in sorted(core_data, key=lambda x: x.get("Paper_ID", "")):
        paper_id = rec.get("Paper_ID", "")
        sections = rec.get("sections", {})

        if not paper_id:
            continue

        print(f"  Processing {paper_id} ({len(sections)} sections) ...")
        paper_fixed = 0
        paper_ents  = 0

        for section_name, section_text in sections.items():
            if not isinstance(section_text, str):
                continue
            if section_name.strip().lower() in ALWAYS_SKIP:
                continue

            with driver.session(database=DATABASE) as session:
                status, added = process_section(
                    session, paper_id, section_name, section_text
                )

            if status == "fixed":
                paper_fixed    += 1
                paper_ents     += added
                total_fixed    += 1
                total_entities += added
            else:
                total_skipped  += 1

        print(f"    -> {paper_fixed} sections fixed | {paper_ents} entities added")

    # ── Print after state ─────────────────────────────────────────────────────
    print(f"\nAfter state:")
    with driver.session(database=DATABASE) as session:
        rows  = session.run(Q_ALL_SECTIONS_STATUS)
        after = [r.data() for r in rows]

    still_missing = [r for r in after if not r["has_raw_text"]]
    text_only     = [r for r in after
                     if r["has_raw_text"] and r["entity_count"] == 0]

    print(f"  Total sections          : {len(after)}")
    print(f"  Sections with raw_text  : {len(after) - len(still_missing)}")
    print(f"  Still missing raw_text  : {len(still_missing)}")
    print(f"  text_only sections      : {len(text_only)} "
          f"(RAG will embed raw_text directly)")

    if still_missing:
        print(f"\n  Still missing (no text in core_data.json):")
        for r in still_missing:
            print(f"    {r['paper_id']} — {r['section']} "
                  f"(chars: {r['char_count']})")

    driver.close()
    print(f"\nDisconnected from Neo4j.")
    print(f"\nFix complete:")
    print(f"  Sections fixed    : {total_fixed}")
    print(f"  Entities added    : {total_entities}")
    print(f"  Sections skipped  : {total_skipped}")
    print(f"\nRe-run step1_neo4j_fetch.py to pick up raw_text in graph_documents.json")


if __name__ == "__main__":
    main()