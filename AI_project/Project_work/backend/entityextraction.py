import re
import json
import spacy
import os

# Load spaCy English NLP model
nlp = spacy.load("en_core_web_sm")

# Input files (processed in order: metadata -> abstract -> core -> citation)
METADATA_FILE = "meta_data.json"
ABSTRACT_FILE = "abstract_data.json"
CORE_FILE     = "core_data.json"
CITATION_FILE = "citation_data.json"

# Output file
OUTPUT_FILE = "extracted_entities.json"

# Valid entity types for Neo4j knowledge graph
VALID_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
    "EVENT", "NORP", "FAC", "LOC", "LANGUAGE",
    "DATE", "PERCENT", "CARDINAL", "QUANTITY", "LAW"
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from '{file_path}'")
    return data


def index_by_paper_id(records):
    return {r["Paper_ID"]: r for r in records if r.get("Paper_ID")}


def normalize_entity(text):
    """Lowercase and strip for consistent entity matching across sections."""
    return text.strip().lower()


def extract_entities(text):
    """
    Run spaCy NER on text.
    Returns deduplicated list of {text, normalized, type}.
    """
    if not text or len(text.strip()) < 5:
        return []
    doc  = nlp(text[:1_000_000])
    seen = set()
    out  = []
    for ent in doc.ents:
        if ent.label_ not in VALID_ENTITY_TYPES:
            continue
        norm = normalize_entity(ent.text)
        key  = (norm, ent.label_)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": ent.text.strip(), "normalized": norm, "type": ent.label_})
    return out


# ---------------------------------------------------------------------------
# Per-source extractors
# ---------------------------------------------------------------------------

def extract_from_metadata(item):
    """
    Actual meta_data.json keys:
      Paper_ID | Paper title | Authors (semicolon-separated string) |
      Publication year | Journal / conference name | Keywords (comma-separated) |
      Domain / research area
    """
    title    = item.get("Paper title", "")
    authors  = [a.strip() for a in item.get("Authors", "").split(";") if a.strip()]
    keywords = [k.strip() for k in item.get("Keywords", "").split(",") if k.strip()]
    year     = str(item.get("Publication year", ""))
    venue    = item.get("Journal / conference name", "")
    domain   = item.get("Domain / research area", "")

    sections = {}
    if title:
        sections["title"]    = {"text": title,              "entities": extract_entities(title)}
    if keywords:
        sections["keywords"] = {"text": " ".join(keywords), "entities": extract_entities(" ".join(keywords))}
    if authors:
        sections["authors"]  = {"text": " ".join(authors),  "entities": extract_entities(" ".join(authors))}
    if venue:
        sections["venue"]    = {"text": venue,              "entities": extract_entities(venue)}

    return {
        "metadata": {
            # Structured fields used directly as Neo4j node properties
            "paper_title": title,
            "authors":     authors,    # clean list  -> AUTHORED_BY edges
            "keywords":    keywords,   # clean list  -> HAS_KEYWORD edges
            "year":        year,
            "venue":       venue,
            "domain":      domain,
            "sections":    sections
        }
    }


def extract_from_abstract(item):
    """Key: abstract (lowercase) confirmed from your data."""
    text = item.get("abstract", "").strip()
    return {"abstract": {"text": text, "entities": extract_entities(text)}}


# Sections always skipped — noisy or already handled by cleaner sources
# "Paper Details" : mixed abstract+metadata already extracted cleanly
# "References"    : raw citation strings handled via citation_data.json
ALWAYS_SKIP = {"paper details", "references"}

def is_reference_continuation(section_name, text):
    """
    Detect if a section is a cut-off continuation of References text,
    not real content. Correctly keeps real Methods/Methodology sections.

    Detection signals:
      1. Text starts with a citation marker [N]
      2. More than 40% of lines contain citation markers
      3. Short text (<200 chars) starting with lowercase
      4. Short text (<500 chars) starting with the section name
         AND contains citation markers — catches bleed like:
         section="Methods", text="Methods for Rates and Proportions [62]..."
    """
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

    # Short section starting with its own name + has citations = heading bleed
    if len(stripped) < 500:
        has_citations = bool(re.search(r"\[\d+\]", stripped))
        starts_with_section = stripped.lower().startswith(section_name.strip().lower())
        if has_citations and starts_with_section:
            return True

    return False


def extract_from_core(item):
    """
    Extract entities section by section.
    Enables Neo4j model: (:Paper)-[:HAS_SECTION]->(:Section)-[:MENTIONS]->(:Entity)

    Always skips: "Paper Details", "References"
    Conditionally skips: any section whose text is a reference continuation
    Keeps: Introduction, Methodology, Results, Conclusion, Implementation, etc.
    """
    processed = {}
    for section_name, section_text in item.get("sections", {}).items():
        if not isinstance(section_text, str):
            continue
        # Always skip noisy sections
        if section_name.strip().lower() in ALWAYS_SKIP:
            print(f"    Skipping (always): '{section_name}'")
            continue
        # Conditionally skip reference continuations
        if is_reference_continuation(section_name, section_text):
            print(f"    Skipping (ref continuation): '{section_name}'")
            continue
        processed[section_name] = {
            "text":     section_text.strip(),
            "entities": extract_entities(section_text.strip())
        }
    return {"core_sections": processed}


def extract_from_citations(cite_rows):
    """
    Actual citation_data.json keys per row:
      Paper_ID | Reference ID | Citation Count |
      Citated year | Reference Title | Reference link

    cite_rows: list of all citation records for one paper.
    Each row becomes one structured citation node in Neo4j.
    """
    citations = []
    for row in cite_rows:
        ref_title = row.get("Reference Title", "").strip()
        citations.append({
            "reference_id":    row.get("Reference ID",    ""),
            "reference_title": ref_title,
            "cited_year":      str(row.get("Citated year", "")),
            "reference_link":  row.get("Reference link",  ""),
            "citation_count":  row.get("Citation Count",  ""),
            # NLP entities from the reference title
            "entities":        extract_entities(ref_title)
        })
    return citations


def collect_all_entities(paper_record):
    """
    Flatten + deduplicate entities from every source into one top-level list.
    Used for fast look-ups without traversing nested sections.
    """
    seen, result = set(), []

    def _add(ents):
        for e in ents:
            key = (e.get("normalized", e["text"].lower()), e["type"])
            if key not in seen:
                seen.add(key)
                result.append(e)

    for sec in paper_record.get("metadata", {}).get("sections", {}).values():
        _add(sec.get("entities", []))
    _add(paper_record.get("abstract", {}).get("entities", []))
    for sec in paper_record.get("core_sections", {}).values():
        _add(sec.get("entities", []))
    for cite in paper_record.get("citations", []):
        _add(cite.get("entities", []))

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    metadata_data = load_json(METADATA_FILE)
    abstract_data = load_json(ABSTRACT_FILE)
    core_data     = load_json(CORE_FILE)
    citation_data = load_json(CITATION_FILE)

    metadata_idx = index_by_paper_id(metadata_data)
    abstract_idx = index_by_paper_id(abstract_data)
    core_idx     = index_by_paper_id(core_data)

    # Group citation rows by Paper_ID (each row is one citation record)
    citation_idx = {}
    for rec in citation_data:
        pid = rec.get("Paper_ID")
        if pid:
            citation_idx.setdefault(pid, []).append(rec)

    all_ids = (
        set(metadata_idx) | set(abstract_idx) |
        set(core_idx)     | set(citation_idx)
    )
    print(f"\nTotal unique papers : {len(all_ids)}\n")

    results = []

    for paper_id in sorted(all_ids):
        print(f"Processing {paper_id} ...")
        paper_record = {"Paper_ID": paper_id}

        # 1. Metadata
        if paper_id in metadata_idx:
            paper_record.update(extract_from_metadata(metadata_idx[paper_id]))

        # 2. Abstract
        if paper_id in abstract_idx:
            paper_record.update(extract_from_abstract(abstract_idx[paper_id]))

        # 3. Core sections
        if paper_id in core_idx:
            paper_record.update(extract_from_core(core_idx[paper_id]))
        else:
            print(f"  WARNING: No core data for {paper_id}")

        # 4. Citations
        cite_rows = citation_idx.get(paper_id, [])
        paper_record["citations"] = extract_from_citations(cite_rows)

        # 5. Aggregated entity list (all sources, deduped)
        paper_record["all_entities"] = collect_all_entities(paper_record)

        results.append(paper_record)
        print(f"  -> {len(paper_record['all_entities'])} unique entities | "
              f"{len(paper_record['citations'])} citations")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    total_ents  = sum(len(r["all_entities"]) for r in results)
    total_cites = sum(len(r["citations"])    for r in results)
    print(f"\nEntity extraction complete -> '{OUTPUT_FILE}'")
    print(f"  Papers processed : {len(results)}")
    print(f"  Total entities   : {total_ents}")
    print(f"  Total citations  : {total_cites}")


if __name__ == "__main__":
    main()