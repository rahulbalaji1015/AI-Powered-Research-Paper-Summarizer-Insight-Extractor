import json
import spacy
import re
import os

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Input files
ENTITIES_FILE = "extracted_entities.json"   # PRIMARY source
METADATA_FILE = "meta_data.json"
ABSTRACT_FILE = "abstract_data.json"
CORE_FILE     = "core_data.json"
CITATION_FILE = "citation_data.json"

# Output file
OUTPUT_FILE = "entity_relationships.json"

# Valid entity types
VALID_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
    "EVENT", "NORP", "FAC", "LOC", "LANGUAGE",
    "DATE", "PERCENT", "CARDINAL", "QUANTITY", "LAW"
}

# Verbs too generic to be meaningful edges in a knowledge graph
NOISE_VERBS = {
    "BE", "HAVE", "DO", "GET", "GO", "MAKE", "COME",
    "TAKE", "GIVE", "KNOW", "THINK", "SEE", "LOOK",
    "WANT", "SEEM", "BECOME", "SHOW", "FEEL", "TRY",
    "TELL", "PUT", "KEEP", "LET", "BEGIN", "APPEAR",
    "TURN", "MOVE", "PLAY", "RUN", "SET", "FIND"
}

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}
OBJECT_DEPS  = {"dobj", "pobj", "iobj", "attr", "oprd", "obj", "dative"}

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


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"References\s*\n.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def normalize(text):
    return text.strip().lower()

# ---------------------------------------------------------------------------
# Core relationship extractor
# ---------------------------------------------------------------------------

def extract_relationships(text, source, section=None):
    """
    Extract relationships from text sentence by sentence.

    Verb resolution order:
      1. Dependency-path verb — verb that syntactically links subject -> object
      2. First non-noise verb in the sentence
      3. Skip the pair entirely (no CO_OCCURS fallback — keeps output clean)

    Each relationship:
      subject  { text, normalized, type }
      relation   verb lemma UPPER CASE
      object   { text, normalized, type }
      source     file / section that produced this
      section    section name (core only)
      sentence   original sentence for provenance
    """
    if not text or len(text.strip()) < 30:
        return []

    doc   = nlp(text[:1_000_000])
    rels  = []

    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in VALID_ENTITY_TYPES]
        if len(ents) < 2:
            continue

        sent_text = sent.text.strip()

        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                subj_ent = ents[i]
                obj_ent  = ents[j]

                subj_ids = {t.i for t in subj_ent}
                obj_ids  = {t.i for t in obj_ent}
                verb     = None

                # Step 1: dependency-path verb
                for token in sent:
                    if token.pos_ != "VERB":
                        continue
                    children = list(token.children)
                    has_subj = any(c.i in subj_ids and c.dep_ in SUBJECT_DEPS for c in children)
                    has_obj  = any(c.i in obj_ids  and c.dep_ in OBJECT_DEPS  for c in children)
                    if has_subj and has_obj:
                        verb = token.lemma_.upper()
                        break

                # Step 2: first non-noise verb
                if not verb:
                    for token in sent:
                        if token.pos_ == "VERB" and token.lemma_.upper() not in NOISE_VERBS:
                            verb = token.lemma_.upper()
                            break

                # Skip if still no meaningful verb
                if not verb or verb in NOISE_VERBS:
                    continue

                rel = {
                    "subject":  {
                        "text":       subj_ent.text.strip(),
                        "normalized": normalize(subj_ent.text),
                        "type":       subj_ent.label_
                    },
                    "relation": verb,
                    "object":   {
                        "text":       obj_ent.text.strip(),
                        "normalized": normalize(obj_ent.text),
                        "type":       obj_ent.label_
                    },
                    "source":   source,
                    "sentence": sent_text
                }
                if section:
                    rel["section"] = section

                rels.append(rel)

    return rels


def remove_duplicates(rels):
    """Deduplicate on (normalized subject, relation, normalized object)."""
    seen, unique = set(), []
    for r in rels:
        key = (r["subject"]["normalized"], r["relation"], r["object"]["normalized"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# ---------------------------------------------------------------------------
# Structured relationship builders (no NLP needed — ground-truth edges)
# ---------------------------------------------------------------------------

def structured_from_metadata(meta, paper_id):
    """
    Build direct Neo4j-ready edges from metadata fields:
      AUTHORED_BY  — paper -> author
      HAS_KEYWORD  — paper -> keyword
      PUBLISHED_IN — paper -> venue
      IN_DOMAIN    — paper -> domain
    """
    title    = meta.get("paper_title", paper_id)
    authors  = meta.get("authors",  [])
    keywords = meta.get("keywords", [])
    venue    = meta.get("venue",    "")
    domain   = meta.get("domain",   "")
    year     = meta.get("year",     "")
    rels     = []

    for author in authors:
        if author:
            rels.append({
                "subject":  {"text": title,  "normalized": normalize(title),  "type": "WORK_OF_ART"},
                "relation": "AUTHORED_BY",
                "object":   {"text": author, "normalized": normalize(author), "type": "PERSON"},
                "source":   "metadata:structured",
                "sentence": f"{title} authored by {author}"
            })

    for kw in keywords:
        if kw:
            rels.append({
                "subject":  {"text": title, "normalized": normalize(title), "type": "WORK_OF_ART"},
                "relation": "HAS_KEYWORD",
                "object":   {"text": kw,    "normalized": normalize(kw),    "type": "PRODUCT"},
                "source":   "metadata:structured",
                "sentence": f"{title} has keyword: {kw}"
            })

    if venue:
        rels.append({
            "subject":  {"text": title, "normalized": normalize(title), "type": "WORK_OF_ART"},
            "relation": "PUBLISHED_IN",
            "object":   {"text": venue, "normalized": normalize(venue), "type": "EVENT"},
            "source":   "metadata:structured",
            "sentence": f"{title} published in {venue}"
        })

    if domain:
        rels.append({
            "subject":  {"text": title,  "normalized": normalize(title),  "type": "WORK_OF_ART"},
            "relation": "IN_DOMAIN",
            "object":   {"text": domain, "normalized": normalize(domain), "type": "NORP"},
            "source":   "metadata:structured",
            "sentence": f"{title} belongs to domain: {domain}"
        })

    if year:
        rels.append({
            "subject":  {"text": title, "normalized": normalize(title), "type": "WORK_OF_ART"},
            "relation": "PUBLISHED_YEAR",
            "object":   {"text": year,  "normalized": normalize(year),  "type": "DATE"},
            "source":   "metadata:structured",
            "sentence": f"{title} published in year {year}"
        })

    return rels


def structured_from_citations(citations, paper_title):
    """
    Build CITES edges from structured citation records.
    Each citation becomes: (:Paper)-[:CITES]->(:Reference)
    """
    rels = []
    for cite in citations:
        ref_title = cite.get("reference_title", "").strip()
        if not ref_title:
            continue
        rels.append({
            "subject":  {"text": paper_title, "normalized": normalize(paper_title), "type": "WORK_OF_ART"},
            "relation": "CITES",
            "object":   {"text": ref_title,   "normalized": normalize(ref_title),   "type": "WORK_OF_ART"},
            "source":   "citation:structured",
            "sentence": f"{paper_title} cites: {ref_title}",
            "citation_metadata": {
                "reference_id":   cite.get("reference_id",    ""),
                "cited_year":     cite.get("cited_year",       ""),
                "citation_count": cite.get("citation_count",  ""),
                "reference_link": cite.get("reference_link",  "")
            }
        })
    return rels

# ---------------------------------------------------------------------------
# NLP relationship builders (from text content)
# ---------------------------------------------------------------------------

def nlp_from_metadata(meta):
    title    = meta.get("paper_title", "")
    authors  = " ".join(meta.get("authors",  []))
    keywords = " ".join(meta.get("keywords", []))
    combined = clean_text(f"{title}. {authors}. {keywords}")
    return extract_relationships(combined, source="metadata:nlp")


def nlp_from_abstract(abstract_rec):
    text = clean_text(abstract_rec.get("abstract", ""))
    return extract_relationships(text, source="abstract")


# Sections always skipped — noisy or already handled by cleaner sources
ALWAYS_SKIP = {"paper details", "references"}

def is_reference_continuation(section_name, text):
    """
    Detect if a section is a cut-off continuation of References text,
    not real content. Correctly keeps real Methods/Methodology sections.

    Detection signals:
      1. Text starts with a citation marker [N]
      2. More than 40% of lines contain citation markers
      3. Short text (<200 chars) starting with lowercase
      4. Short text (<500 chars) that starts with the section name
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

    # Short section that starts with its own name + has citations = heading bleed
    if len(stripped) < 500:
        has_citations = bool(re.search(r"\[\d+\]", stripped))
        starts_with_section = stripped.lower().startswith(section_name.strip().lower())
        if has_citations and starts_with_section:
            return True

    return False


def nlp_from_core_sections(core_rec):
    """
    NLP pass over each core section independently.
    Always skips: Paper Details, References.
    Conditionally skips: any section detected as reference continuation.
    Keeps: Introduction, Methodology, Methods (real), Results, Conclusion, etc.
    """
    all_rels = []
    for section_name, section_text in core_rec.get("sections", {}).items():
        if not isinstance(section_text, str):
            continue
        if section_name.strip().lower() in ALWAYS_SKIP:
            continue
        if is_reference_continuation(section_name, section_text):
            continue
        cleaned = clean_text(section_text)
        rels    = extract_relationships(cleaned, source=f"core::{section_name}", section=section_name)
        all_rels.extend(rels)
    return all_rels


def nlp_from_extracted_entities(entity_rec):
    """
    PRIMARY NLP pass — runs on stored section texts from extracted_entities.json
    to guarantee consistency with the entity extraction step.
    """
    all_rels = []

    # Abstract
    ab_text = clean_text(entity_rec.get("abstract", {}).get("text", ""))
    if ab_text:
        all_rels += extract_relationships(ab_text, source="extracted_entities::abstract")

    # Core sections — entity file already filtered noisy sections at extraction time
    for section_name, sec_data in entity_rec.get("core_sections", {}).items():
        if section_name.strip().lower() in ALWAYS_SKIP:
            continue
        sec_text = clean_text(sec_data.get("text", ""))
        if sec_text:
            all_rels += extract_relationships(
                sec_text,
                source=f"extracted_entities::{section_name}",
                section=section_name
            )

    # Citation reference titles
    for cite in entity_rec.get("citations", []):
        ref_text = clean_text(cite.get("reference_title", ""))
        if ref_text:
            all_rels += extract_relationships(ref_text, source="extracted_entities::citations")

    return all_rels

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # Load all sources
    entities_data = load_json(ENTITIES_FILE)
    metadata_data = load_json(METADATA_FILE)
    abstract_data = load_json(ABSTRACT_FILE)
    core_data     = load_json(CORE_FILE)
    citation_data = load_json(CITATION_FILE)

    entities_idx = index_by_paper_id(entities_data)
    metadata_idx = index_by_paper_id(metadata_data)
    abstract_idx = index_by_paper_id(abstract_data)
    core_idx     = index_by_paper_id(core_data)

    # Group citation rows by Paper_ID
    citation_idx = {}
    for rec in citation_data:
        pid = rec.get("Paper_ID")
        if pid:
            citation_idx.setdefault(pid, []).append(rec)

    all_ids = (
        set(entities_idx) | set(metadata_idx) |
        set(abstract_idx) | set(core_idx)     |
        set(citation_idx)
    )
    print(f"\nTotal unique papers : {len(all_ids)}\n")

    results = []

    for paper_id in sorted(all_ids):
        print(f"Processing {paper_id} ...")

        # ── Title resolution ─────────────────────────────────────────────
        # Use entity extraction output (already has clean metadata)
        entity_rec  = entities_idx.get(paper_id, {})
        meta_block  = entity_rec.get("metadata", {})

        title = meta_block.get("paper_title", "")
        if not title:
            title = (metadata_idx.get(paper_id) or {}).get("Paper title", "")
        if not title:
            print(f"  WARNING: No title found for {paper_id}, using Paper_ID")
            title = paper_id

        all_rels = []

        # ── Structured edges (ground-truth, no NLP) ──────────────────────
        if meta_block:
            all_rels += structured_from_metadata(meta_block, paper_id)

        cite_rows = citation_idx.get(paper_id, [])
        structured_cites = []
        for row in cite_rows:
            ref_title = row.get("Reference Title", "").strip()
            if ref_title:
                structured_cites.append({
                    "reference_id":    row.get("Reference ID",    ""),
                    "reference_title": ref_title,
                    "cited_year":      str(row.get("Citated year", "")),
                    "reference_link":  row.get("Reference link",  ""),
                    "citation_count":  row.get("Citation Count",  "")
                })
        all_rels += structured_from_citations(structured_cites, title)

        # ── NLP edges (order: metadata -> abstract -> core -> entities) ───
        if meta_block:
            all_rels += nlp_from_metadata(meta_block)

        if paper_id in abstract_idx:
            all_rels += nlp_from_abstract(abstract_idx[paper_id])

        if paper_id in core_idx:
            all_rels += nlp_from_core_sections(core_idx[paper_id])
        else:
            print(f"  WARNING: No core data found for {paper_id}")

        if paper_id in entities_idx:
            all_rels += nlp_from_extracted_entities(entities_idx[paper_id])

        # Deduplicate
        all_rels = remove_duplicates(all_rels)

        results.append({
            "Paper_ID":            paper_id,
            "paper_title":         title,
            "total_relationships": len(all_rels),
            "relationships":       all_rels
        })
        print(f"  -> {len(all_rels)} unique relationships")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    total = sum(r["total_relationships"] for r in results)
    print(f"\nRelationship extraction complete -> '{OUTPUT_FILE}'")
    print(f"  Papers processed    : {len(results)}")
    print(f"  Total relationships : {total}")


if __name__ == "__main__":
    main()