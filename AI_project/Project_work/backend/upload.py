"""
full_process_upload.py

Runs the complete RAG pipeline for a newly uploaded paper:
  1. Extract text from PDF/DOCX/TXT
  2. Detect sections from raw text
  3. Run spaCy NER (entity extraction)
  4. Extract NLP relationship triples
  5. Import to Neo4j knowledge graph
  6. Build 4 chunk types (metadata + sections + relationships + citations)
  7. Generate embeddings
  8. Add to FAISS index

Import this in app.py:
  from full_process_upload import full_process_paper
"""

import os
import re
import json
import time
import tempfile
import numpy as np
import faiss
import spacy
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ── Settings ──────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"   # ← update
NEO4J_DATABASE = "neo4j"           # ← update

FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE    = "faiss_metadata.json"
CHUNKS_FILE      = "chunk_embeddings.json"

EMBED_MODEL      = "all-MiniLM-L6-v2"
TARGET_TOKENS    = 400
OVERLAP_TOKENS   = 50
MIN_TOKENS       = 50

ALWAYS_SKIP      = {"paper details", "references"}

VALID_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
    "EVENT", "NORP", "FAC", "LOC", "LANGUAGE",
    "DATE", "PERCENT", "CARDINAL", "QUANTITY", "LAW"
}

NOISE_VERBS = {
    "BE", "HAVE", "DO", "GET", "GO", "MAKE", "COME", "TAKE",
    "GIVE", "KNOW", "THINK", "SEE", "LOOK", "WANT", "SEEM",
    "BECOME", "SHOW", "FEEL", "TRY", "TELL", "PUT", "KEEP",
    "LET", "BEGIN", "APPEAR", "TURN", "MOVE", "PLAY", "RUN",
    "SET", "FIND"
}

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


def count_tokens(text):
    return len(text.split())


# ---------------------------------------------------------------------------
# Step 1 — Text extraction (same as app.py)
# ---------------------------------------------------------------------------

def extract_text_from_file(uploaded_file):
    """Extract plain text from uploaded PDF, DOCX, or TXT."""
    name = uploaded_file.name.lower()
    text = ""

    if name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif name.endswith(".pdf"):
        import pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        os.unlink(tmp_path)

    elif name.endswith(".docx"):
        from docx import Document as DocxDocument
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        doc = DocxDocument(tmp_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        os.unlink(tmp_path)

    return text.strip()


# ---------------------------------------------------------------------------
# Step 2 — Section detection from raw text
# ---------------------------------------------------------------------------

# Common section heading patterns in research papers
SECTION_PATTERNS = [
    r"^(abstract|introduction|background|related work|literature review|"
    r"methodology|methods|method|approach|system|architecture|"
    r"implementation|experiments|experimental setup|results|"
    r"evaluation|discussion|conclusion|conclusions|"
    r"future work|acknowledgements?|references)[\s\.\:]*$"
]


def detect_sections(text):
    """
    Split raw text into sections by detecting common section headings.
    Returns dict: {section_name: section_text}
    """
    lines    = text.split("\n")
    sections = {}
    current_section = "Introduction"   # default if no heading found
    current_lines   = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        # Check if line looks like a section heading
        is_heading = False
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, stripped.lower()):
                is_heading = True
                break

        # Also detect numbered sections like "1. Introduction" "2. Methods"
        if re.match(r"^\d+\.?\s+[A-Z][a-zA-Z\s]+$", stripped):
            is_heading = True

        if is_heading and len(stripped) < 60:
            # Save previous section
            section_text = "\n".join(current_lines).strip()
            if section_text and len(section_text) > 50:
                sections[current_section] = section_text

            # Start new section
            current_section = stripped.title()
            current_lines   = []
        else:
            current_lines.append(line)

    # Save last section
    section_text = "\n".join(current_lines).strip()
    if section_text and len(section_text) > 50:
        sections[current_section] = section_text

    # If no sections detected, treat entire text as one section
    if not sections:
        sections["Content"] = text

    return sections


# ---------------------------------------------------------------------------
# Step 3 — Entity extraction (same as entity_extraction.py)
# ---------------------------------------------------------------------------

def extract_entities(text, nlp):
    """Extract named entities using spaCy."""
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
# Step 4 — Relationship extraction (same as relationship_extraction.py)
# ---------------------------------------------------------------------------

def extract_relationships(text, section_name, nlp):
    """Extract subject-verb-object triples from text."""
    if not text or len(text.strip()) < 30:
        return []

    doc   = nlp(text[:500_000])
    rels  = []
    seen  = set()

    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in VALID_ENTITY_TYPES]
        if len(ents) < 2:
            continue

        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                subj_ent = ents[i]
                obj_ent  = ents[j]

                # Find verb connecting them
                verb = None
                for token in sent:
                    if token.pos_ == "VERB":
                        lemma = token.lemma_.upper()
                        if lemma not in NOISE_VERBS:
                            verb = lemma
                            break

                if not verb:
                    continue

                key = (subj_ent.text.lower(), verb, obj_ent.text.lower())
                if key in seen:
                    continue
                seen.add(key)

                rels.append({
                    "subject":  {"text": subj_ent.text, "normalized": subj_ent.text.lower(), "type": subj_ent.label_},
                    "relation": verb,
                    "object":   {"text": obj_ent.text,  "normalized": obj_ent.text.lower(),  "type": obj_ent.label_},
                    "section":  section_name,
                    "sentence": sent.text.strip()
                })

    return rels


# ---------------------------------------------------------------------------
# Step 5 — Neo4j import
# ---------------------------------------------------------------------------

def import_to_neo4j(paper_id, title, sections, all_entities,
                     relationships, progress_callback=None):
    """Import paper data into Neo4j knowledge graph."""
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
    except Exception as e:
        return False, f"Neo4j connection failed: {e}"

    try:
        with driver.session(database=NEO4J_DATABASE) as session:

            # Paper node
            session.run("""
                MERGE (p:Paper {paper_id: $paper_id})
                SET p.title  = $title,
                    p.domain = 'uploaded'
            """, paper_id=paper_id, title=title)

            if progress_callback:
                progress_callback("Paper node created")

            # Section nodes + raw_text
            for sec_name, sec_text in sections.items():
                session.run("""
                    MERGE (s:Section {paper_id: $paper_id, name: $name})
                    SET s.raw_text   = $raw_text,
                        s.char_count = $char_count,
                        s.word_count = $word_count
                    WITH s
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:HAS_SECTION]->(s)
                """,
                    paper_id   = paper_id,
                    name       = sec_name,
                    raw_text   = sec_text,
                    char_count = len(sec_text),
                    word_count = len(sec_text.split())
                )

            if progress_callback:
                progress_callback(f"{len(sections)} sections imported")

            # Entity nodes + MENTIONS edges
            for ent in all_entities:
                session.run("""
                    MERGE (e:Entity {normalized: $normalized, type: $type})
                    SET e.text = $text
                    WITH e
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:MENTIONS]->(e)
                """,
                    paper_id   = paper_id,
                    normalized = ent["normalized"],
                    type       = ent["type"],
                    text       = ent["text"]
                )

            if progress_callback:
                progress_callback(f"{len(all_entities)} entities imported")

            # Relationship triples as RELATED edges
            for rel in relationships:
                session.run("""
                    MERGE (subj:Entity {normalized: $subj_norm, type: $subj_type})
                    SET subj.text = $subj_text
                    MERGE (obj:Entity  {normalized: $obj_norm,  type: $obj_type})
                    SET obj.text  = $obj_text
                    MERGE (subj)-[r:RELATED {
                        paper_id: $paper_id,
                        relation: $relation
                    }]->(obj)
                    SET r.source   = $section,
                        r.sentence = $sentence
                """,
                    paper_id   = paper_id,
                    subj_norm  = rel["subject"]["normalized"],
                    subj_type  = rel["subject"]["type"],
                    subj_text  = rel["subject"]["text"],
                    obj_norm   = rel["object"]["normalized"],
                    obj_type   = rel["object"]["type"],
                    obj_text   = rel["object"]["text"],
                    relation   = rel["relation"],
                    section    = rel["section"],
                    sentence   = rel["sentence"]
                )

            if progress_callback:
                progress_callback(f"{len(relationships)} relationships imported")

        driver.close()
        return True, "Neo4j import successful"

    except Exception as e:
        driver.close()
        return False, f"Neo4j import error: {e}"


# ---------------------------------------------------------------------------
# Step 6 — Build 4 chunk types (same as step2_text_conversion.py)
# ---------------------------------------------------------------------------

def split_into_chunks(text, paper_id, section_name, source_type, counter):
    """Split text into overlapping chunks."""
    words  = text.split()
    chunks = []

    if len(words) < MIN_TOKENS:
        return chunks, counter

    if len(words) <= TARGET_TOKENS:
        counter += 1
        chunks.append({
            "chunk_id":    f"{paper_id}_chunk_{counter:03d}",
            "paper_id":    paper_id,
            "section":     section_name,
            "source_type": source_type,
            "token_count": len(words),
            "text":        text.strip()
        })
        return chunks, counter

    start = 0
    while start < len(words):
        end   = min(start + TARGET_TOKENS, len(words))
        chunk = words[start:end]
        if len(chunk) >= MIN_TOKENS:
            counter += 1
            chunks.append({
                "chunk_id":    f"{paper_id}_chunk_{counter:03d}",
                "paper_id":    paper_id,
                "section":     section_name,
                "source_type": source_type,
                "token_count": len(chunk),
                "text":        " ".join(chunk)
            })
        start += TARGET_TOKENS - OVERLAP_TOKENS

    return chunks, counter


def build_chunks(paper_id, title, sections,
                 all_entities, relationships):
    """Build all 4 chunk types for the paper."""
    all_chunks = []
    counter    = 0

    # 1. Metadata chunk
    meta_text = f"Paper ID: {paper_id}\nTitle: {title}\nDomain: uploaded"
    counter  += 1
    all_chunks.append({
        "chunk_id":    f"{paper_id}_chunk_{counter:03d}",
        "paper_id":    paper_id,
        "section":     "metadata",
        "source_type": "metadata",
        "token_count": count_tokens(meta_text),
        "text":        meta_text
    })

    # 2. Section chunks
    ent_by_section = {}
    for e in all_entities:
        sec = e.get("section", "")
        ent_by_section.setdefault(sec, []).append(e)

    for sec_name, sec_text in sections.items():
        if sec_name.lower() in ALWAYS_SKIP:
            continue
        if not sec_text or len(sec_text.strip()) < 50:
            continue

        # Build section text with entity context
        lines = [f"Section: {sec_name}", f"Paper: {title}", "", sec_text]
        sec_ents = ent_by_section.get(sec_name, [])
        if sec_ents:
            by_type = {}
            for e in sec_ents:
                by_type.setdefault(e["type"], []).append(e["text"])
            lines.append("\nKey entities:")
            for etype, names in by_type.items():
                unique = list(dict.fromkeys(names))[:8]
                lines.append(f"  {etype}: {', '.join(unique)}")

        new_chunks, counter = split_into_chunks(
            "\n".join(lines), paper_id, sec_name, "section", counter
        )
        all_chunks.extend(new_chunks)

    # 3. Relationship chunks
    if relationships:
        rel_lines = [f"Relationships in: {title}", ""]
        by_section = {}
        for r in relationships:
            by_section.setdefault(r["section"], []).append(r)

        for sec_name, sec_rels in by_section.items():
            rel_lines.append(f"[{sec_name}]")
            for r in sec_rels[:25]:
                if r.get("sentence"):
                    rel_lines.append(f"  {r['sentence']}")
                else:
                    rel_lines.append(
                        f"  {r['subject']['text']} "
                        f"{r['relation']} "
                        f"{r['object']['text']}"
                    )
            rel_lines.append("")

        new_chunks, counter = split_into_chunks(
            "\n".join(rel_lines), paper_id,
            "relationships", "relationships", counter
        )
        all_chunks.extend(new_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Step 7+8 — Embed and add to FAISS
# ---------------------------------------------------------------------------

def embed_and_index(chunks, embed_model, progress_callback=None):
    """Generate embeddings and add chunks to FAISS index."""
    if not chunks:
        return 0

    texts      = [c["text"] for c in chunks]
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings = True,
        convert_to_numpy     = True,
        show_progress_bar    = False
    ).astype(np.float32)

    if progress_callback:
        progress_callback(f"Generated {len(chunks)} embeddings")

    # Load existing FAISS index
    index      = faiss.read_index(FAISS_INDEX_FILE)
    metadata   = load_json(METADATA_FILE) or []
    all_chunks = load_json(CHUNKS_FILE)   or []
    next_id    = index.ntotal

    # Add to FAISS
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Update metadata
    for i, chunk in enumerate(chunks):
        metadata.append({
            "faiss_id":    next_id + i,
            "chunk_id":    chunk["chunk_id"],
            "paper_id":    chunk["paper_id"],
            "section":     chunk["section"],
            "source_type": chunk["source_type"],
            "token_count": chunk["token_count"],
            "text":        chunk["text"]
        })
    save_json(metadata, METADATA_FILE)

    # Update chunk_embeddings
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            **chunk,
            "embedding_dim": int(embeddings.shape[1]),
            "embedding":     embeddings[i].tolist()
        })
    save_json(all_chunks, CHUNKS_FILE)

    if progress_callback:
        progress_callback(f"Added {len(chunks)} vectors to FAISS index")

    return len(chunks)


# ---------------------------------------------------------------------------
# Main function — called from app.py
# ---------------------------------------------------------------------------

def full_process_paper(uploaded_file, paper_id, title,
                        embed_model, progress_callback=None):
    """
    Full pipeline for an uploaded paper.

    Args:
        uploaded_file   : Streamlit UploadedFile object
        paper_id        : e.g. "P11"
        title           : paper title (user-provided or filename)
        embed_model     : loaded SentenceTransformer
        progress_callback: function(message) for UI updates

    Returns:
        success (bool), summary (dict), error_message (str)
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    try:
        # Load spaCy
        log("Loading spaCy model ...")
        nlp = spacy.load("en_core_web_sm")

        # Step 1 — Extract text
        log("Extracting text from file ...")
        text = extract_text_from_file(uploaded_file)
        if not text:
            return False, {}, "Could not extract text from file."
        log(f"Extracted {len(text):,} characters")

        # Step 2 — Detect sections
        log("Detecting sections ...")
        sections = detect_sections(text)
        log(f"Found {len(sections)} sections: {list(sections.keys())}")

        # Step 3 — Entity extraction
        log("Extracting entities with spaCy ...")
        all_entities = []
        for sec_name, sec_text in sections.items():
            ents = extract_entities(sec_text, nlp)
            for e in ents:
                e["section"] = sec_name
            all_entities.extend(ents)

        # Deduplicate
        seen       = set()
        dedup_ents = []
        for e in all_entities:
            key = (e["normalized"], e["type"])
            if key not in seen:
                seen.add(key)
                dedup_ents.append(e)
        log(f"Extracted {len(dedup_ents)} unique entities")

        # Step 4 — Relationship extraction
        log("Extracting relationships ...")
        all_relationships = []
        for sec_name, sec_text in sections.items():
            rels = extract_relationships(sec_text, sec_name, nlp)
            all_relationships.extend(rels)
        log(f"Extracted {len(all_relationships)} relationships")

        # Step 5 — Neo4j import
        log("Importing to Neo4j ...")
        neo4j_ok, neo4j_msg = import_to_neo4j(
            paper_id, title, sections,
            dedup_ents, all_relationships, log
        )
        if not neo4j_ok:
            log(f"Neo4j warning: {neo4j_msg} — continuing without graph import")

        # Step 6 — Build chunks
        log("Building text chunks ...")
        chunks = build_chunks(
            paper_id, title, sections,
            dedup_ents, all_relationships
        )
        log(f"Built {len(chunks)} chunks")

        # Step 7+8 — Embed and add to FAISS
        log("Generating embeddings and indexing ...")
        n_indexed = embed_and_index(chunks, embed_model, log)

        summary = {
            "paper_id":      paper_id,
            "title":         title,
            "sections":      len(sections),
            "entities":      len(dedup_ents),
            "relationships": len(all_relationships),
            "chunks":        len(chunks),
            "indexed":       n_indexed,
            "neo4j":         neo4j_ok
        }

        log(f"Done! {n_indexed} chunks ready for querying.")
        return True, summary, ""

    except Exception as e:
        return False, {}, str(e)