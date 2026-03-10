import json
import os
import sys
from neo4j import GraphDatabase
from neo4j import ServiceUnavailable, ClientError

# ── Connection settings ───────────────────────────────────────────────────────
URI      = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"        # ← change to your Neo4j password
DATABASE = "neo4j"                # ← change if your DB has a different name

# Output file — used by Step 2
OUTPUT_FILE = "graph_documents.json"

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_driver():
    """Connect to Neo4j and verify the connection is live."""
    try:
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {URI} — database: '{DATABASE}'")
        return driver
    except ServiceUnavailable:
        print("\nERROR: Cannot reach Neo4j.")
        print("  Make sure your DBMS is started in Neo4j Desktop (green dot).")
        sys.exit(1)
    except ClientError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Fetch functions — one per node/relationship type
# ---------------------------------------------------------------------------

def fetch_papers(session) -> list:
    """Fetch all Paper nodes with their properties."""
    result = session.run("""
        MATCH (p:Paper)
        RETURN p.paper_id AS paper_id,
               p.title    AS title,
               p.year     AS year,
               p.venue    AS venue,
               p.domain   AS domain
        ORDER BY p.paper_id
    """)
    papers = [record.data() for record in result]
    print(f"  Papers fetched          : {len(papers)}")
    return papers


def fetch_authors(session) -> list:
    """Fetch all AUTHORED_BY edges — Paper → Author."""
    result = session.run("""
        MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        RETURN p.paper_id  AS paper_id,
               a.name      AS author
        ORDER BY p.paper_id
    """)
    rows = [record.data() for record in result]
    print(f"  Author edges fetched    : {len(rows)}")
    return rows


def fetch_keywords(session) -> list:
    """Fetch all HAS_KEYWORD edges — Paper → Keyword."""
    result = session.run("""
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN p.paper_id  AS paper_id,
               k.text      AS keyword
        ORDER BY p.paper_id
    """)
    rows = [record.data() for record in result]
    print(f"  Keyword edges fetched   : {len(rows)}")
    return rows


def fetch_sections(session) -> list:
    """
    Fetch all Section nodes linked to Papers.
    Includes raw_text, char_count, word_count if stored by fix_empty_sections.py
    """
    result = session.run("""
        MATCH (p:Paper)-[:HAS_SECTION]->(s:Section)
        RETURN p.paper_id   AS paper_id,
               s.name       AS section_name,
               s.raw_text   AS raw_text,
               s.char_count AS char_count,
               s.word_count AS word_count
        ORDER BY p.paper_id, s.name
    """)
    rows = [record.data() for record in result]
    print(f"  Section edges fetched   : {len(rows)}")
    return rows


def fetch_entities(session) -> list:
    """
    Fetch all Entity nodes mentioned in sections or papers.
    Includes the section name when available.
    """
    result = session.run("""
        MATCH (p:Paper)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(e:Entity)
        RETURN p.paper_id  AS paper_id,
               s.name      AS section_name,
               e.text      AS entity_text,
               e.type      AS entity_type,
               e.normalized AS normalized
        UNION
        MATCH (p:Paper)-[:MENTIONS]->(e:Entity)
        RETURN p.paper_id  AS paper_id,
               'abstract'  AS section_name,
               e.text      AS entity_text,
               e.type      AS entity_type,
               e.normalized AS normalized
        ORDER BY paper_id, section_name
    """)
    rows = [record.data() for record in result]
    print(f"  Entity mentions fetched : {len(rows)}")
    return rows


def fetch_relationships(session) -> list:
    """
    Fetch all RELATED triples — Entity → relation → Entity.
    These are the NLP-extracted subject-relation-object triples.
    """
    result = session.run("""
        MATCH (subj:Entity)-[r:RELATED]->(obj:Entity)
        RETURN r.paper_id  AS paper_id,
               subj.text   AS subject,
               subj.type   AS subject_type,
               r.relation  AS relation,
               obj.text    AS object,
               obj.type    AS object_type,
               r.source    AS source,
               r.sentence  AS sentence
        ORDER BY r.paper_id
    """)
    rows = [record.data() for record in result]
    print(f"  Relationship triples    : {len(rows)}")
    return rows


def fetch_citations(session) -> list:
    """Fetch all CITES edges — Paper → Reference."""
    result = session.run("""
        MATCH (p:Paper)-[:CITES]->(ref:Reference)
        RETURN p.paper_id       AS paper_id,
               ref.title        AS reference_title,
               ref.reference_id AS reference_id,
               ref.cited_year   AS cited_year,
               ref.citation_count AS citation_count,
               ref.reference_link AS reference_link
        ORDER BY p.paper_id
    """)
    rows = [record.data() for record in result]
    print(f"  Citation edges fetched  : {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Assemble into per-paper documents
# ---------------------------------------------------------------------------

def assemble_documents(papers, authors, keywords,
                        sections, entities, relationships, citations) -> list:
    """
    Combine all fetched data into one document per paper.
    Each document contains all graph data for that paper
    in a structured format ready for Step 2 (text conversion).
    """
    # Build lookup dicts keyed by paper_id
    def group(rows, key="paper_id"):
        result = {}
        for r in rows:
            pid = r[key]
            result.setdefault(pid, []).append(r)
        return result

    author_map   = group(authors)
    keyword_map  = group(keywords)
    section_map  = group(sections)
    entity_map   = group(entities)
    rel_map      = group(relationships)
    citation_map = group(citations)

    documents = []
    for paper in papers:
        pid = paper["paper_id"]
        documents.append({
            "paper_id":      pid,
            "title":         paper.get("title",  ""),
            "year":          paper.get("year",   ""),
            "venue":         paper.get("venue",  ""),
            "domain":        paper.get("domain", ""),
            # Lists
            "authors":       [r["author"]   for r in author_map.get(pid, [])],
            "keywords":      [r["keyword"]  for r in keyword_map.get(pid, [])],
            # Section names + raw text (for RAG embedding)
            "sections": [
                {
                    "name":       r["section_name"],
                    "raw_text":   r.get("raw_text",   ""),
                    "char_count": r.get("char_count",  0),
                    "word_count": r.get("word_count",  0)
                }
                for r in section_map.get(pid, [])
            ],
            # Detailed records
            "entities":      entity_map.get(pid, []),
            "relationships": rel_map.get(pid, []),
            "citations":     citation_map.get(pid, [])
        })

    return documents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    driver = get_driver()

    print("\nFetching graph data ...")
    with driver.session(database=DATABASE) as session:
        papers        = fetch_papers(session)
        authors       = fetch_authors(session)
        keywords      = fetch_keywords(session)
        sections      = fetch_sections(session)
        entities      = fetch_entities(session)
        relationships = fetch_relationships(session)
        citations     = fetch_citations(session)

    driver.close()
    print("Disconnected from Neo4j.")

    print("\nAssembling per-paper documents ...")
    documents = assemble_documents(
        papers, authors, keywords,
        sections, entities, relationships, citations
    )
    print(f"  Documents assembled     : {len(documents)}")

    # Save for Step 2
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)

    print(f"\nStep 1 complete -> '{OUTPUT_FILE}'")
    print(f"  Ready for Step 2 (text conversion + chunking)")

    # Quick summary
    total_rels   = sum(len(d["relationships"]) for d in documents)
    total_ents   = sum(len(d["entities"])      for d in documents)
    total_cites  = sum(len(d["citations"])     for d in documents)
    print(f"\n  Summary:")
    print(f"    Papers       : {len(documents)}")
    print(f"    Entities     : {total_ents}")
    print(f"    Relationships: {total_rels}")
    print(f"    Citations    : {total_cites}")


if __name__ == "__main__":
    main()