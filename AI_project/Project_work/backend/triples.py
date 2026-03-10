import json
import os
from collections import Counter

# Input / Output
INPUT_FILE  = "entity_relationships.json"
OUTPUT_FILE = "triples.json"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} paper records from '{file_path}'")
    return data

# ---------------------------------------------------------------------------
# Triple builder
# ---------------------------------------------------------------------------

def build_triples(data):
    """
    Flatten every relationship into a flat triple record ready for Neo4j import.

    Triple fields:
      Paper_ID         paper this triple belongs to
      paper_title      full title of the paper
      subject          entity surface text
      subject_norm     lowercased for dedup / matching in Neo4j
      subject_type     spaCy label (PERSON, ORG, PRODUCT ...)
      relation         predicate / verb  (e.g. AUTHORED_BY, CITES, PROPOSE)
      object           entity surface text
      object_norm      lowercased for dedup / matching in Neo4j
      object_type      spaCy label
      source           which file/section produced this
                         e.g. metadata:structured | core::Introduction |
                              extracted_entities::Methodology | citation:structured
      section          section name  (only for core / section-level triples)
      sentence         original sentence for provenance and context
      citation_metadata  extra fields only on CITES triples:
                           reference_id, cited_year, citation_count, reference_link
    """
    triples = []

    for paper in data:
        paper_id    = paper.get("Paper_ID", "")
        paper_title = paper.get("paper_title", paper_id)

        for rel in paper.get("relationships", []):
            subj  = rel.get("subject", {})
            obj   = rel.get("object",  {})
            relation = rel.get("relation", "").strip()

            # Skip malformed entries
            if not subj.get("text") or not obj.get("text") or not relation:
                continue

            triple = {
                "Paper_ID":     paper_id,
                "paper_title":  paper_title,
                "subject":      subj.get("text",       "").strip(),
                "subject_norm": subj.get("normalized", subj.get("text", "").lower()).strip(),
                "subject_type": subj.get("type",       "").strip(),
                "relation":     relation,
                "object":       obj.get("text",        "").strip(),
                "object_norm":  obj.get("normalized",  obj.get("text", "").lower()).strip(),
                "object_type":  obj.get("type",        "").strip(),
                "source":       rel.get("source",      ""),
                "sentence":     rel.get("sentence",    "")
            }

            # Section — only when present (core triples)
            section = rel.get("section", "").strip()
            if section:
                triple["section"] = section

            # Citation metadata — only on CITES triples
            if relation == "CITES" and rel.get("citation_metadata"):
                triple["citation_metadata"] = rel["citation_metadata"]

            triples.append(triple)

    return triples


def remove_duplicates(triples):
    """
    Deduplicate on (Paper_ID, subject_norm, relation, object_norm).
    Paper_ID included so same fact from two papers stays as two edges in Neo4j.
    """
    seen, unique = set(), []
    for t in triples:
        key = (
            t["Paper_ID"],
            t["subject_norm"],
            t["relation"],
            t["object_norm"]
        )
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(triples):
    print("\n── Triples per paper ──────────────────────────────────────────────")
    paper_counts = Counter(t["Paper_ID"] for t in triples)
    # Map Paper_ID -> title from first matching triple
    titles = {}
    for t in triples:
        if t["Paper_ID"] not in titles:
            titles[t["Paper_ID"]] = t["paper_title"]

    for pid, count in sorted(paper_counts.items()):
        display = titles.get(pid, pid)
        if len(display) > 55:
            display = display[:52] + "..."
        print(f"  {pid}  {display:<57}  {count:>5} triples")

    print("\n── Triples per relation type ──────────────────────────────────────")
    rel_counts = Counter(t["relation"] for t in triples)
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {rel:<40}  {count:>5}")

    print("\n── Triples per source ─────────────────────────────────────────────")
    def norm_source(src):
        if "::" in src:
            prefix, part = src.split("::", 1)
            return f"{prefix} -> {part}"
        return src or "unknown"

    src_counts = Counter(norm_source(t["source"]) for t in triples)
    for src, count in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<55}  {count:>5}")

    print(f"\n  TOTAL triples : {len(triples)}")
    print("───────────────────────────────────────────────────────────────────\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_json(INPUT_FILE)
    if not data:
        return

    print("\nBuilding triples ...")
    triples = build_triples(data)
    print(f"  Raw triples       : {len(triples)}")

    triples = remove_duplicates(triples)
    print(f"  After dedup       : {len(triples)}")

    print_summary(triples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=4, ensure_ascii=False)

    print(f"Triples saved -> '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()