# AI-Powered-Research-Paper-Summarizer-Insight-Extractor

A full-stack AI-powered Research Assistant that combines Knowledge Graph technology with Retrieval-Augmented Generation (RAG) to enable intelligent querying over a collection of research papers. Built with Neo4j, FAISS, Sentence Transformers, Groq LLM, and Streamlit.

---

## 📌 Project Overview

This project enables users to upload research papers and ask natural language questions about them. The system extracts structured knowledge from papers, stores it in a graph database, builds a semantic vector index, and uses a large language model to generate accurate, source-cited answers through a conversational chat interface.

---

## 🗂️ Project Structure
```
project/
├── config.py                   ← Single source of truth for all paths and settings
├── backend/
│   ├── entity_extraction.py
│   ├── relationship_extraction.py
│   ├── triple_extraction.py
│   ├── neo4j_import.py
│   ├── fix_all_sections.py
│   ├── fix_empty_sections.py
│   ├── step1_neo4j_fetch.py
│   ├── step2_text_conversion.py
│   ├── step3_embeddings.py
│   ├── step4_faiss_store.py
│   ├── step5_query_retrieval.py
│   ├── step6_llm_answer.py
│   ├── step7_evaluation.py
│   ├── step8_rag_pipeline.py
│   ├── full_process_upload.py
│   ├── chat_db.py
│   └── pdf_export.py
├── frontend/
│   └── app.py
└── data/                       ← Auto-created on first run
    ├── faiss_index.bin
    ├── faiss_metadata.json
    ├── chunk_embeddings.json
    ├── embeddings_matrix.npy
    ├── graph_documents.json
    ├── text_chunks.json
    ├── chat_memory.json
    └── rag_chat.db
```

---

## 🧩 Modules

### Module 1 — Data Collection, Entity Extraction, Relationship Extraction & Triples

The first module handles all data ingestion and knowledge extraction from research papers.

**Data Collection(`coredata.py`, `abstract.py` & `exceltojson.py`)**
- Collected 10–20 research papers in PDF format relevant to the project domain
- Each paper was parsed using `pdfplumber` and `python-docx` to extract raw text
- Paper content was split into 4 structured JSON types:
  - `core_data.json` — main body content of each paper
  - `abstract_data.json` — abstract section of each paper
  - `metadata_data.json` — title, authors, year, and keywords
  - `citation_data.json` — references and citation lists

**Entity Extraction(`entityextraction.py`)**
- Used spaCy (`en_core_web_sm`) to perform Named Entity Recognition (NER) on all 4 JSON files
- Extracted entity types include persons, organisations, concepts, and technologies
- All extracted entities are stored in `entities.json`
- Handles edge cases such as empty sections, missing text, and duplicate entities

**Relationship Extraction(`relationshipextraction.py`)**
- Identified relationships between extracted entities using pattern-based and dependency-parse techniques
- Each relationship is structured as a subject–predicate–object triple
- Output stored in `relationships.json`

**Triple Extraction(`triples.py`)**
- Generated formal knowledge triples from entities and relationships
- Triples form the basis for the knowledge graph structure
- Output stored in `triples.json`

---

### Module 2 — Neo4j and Knowledge Graph

The second module builds a structured knowledge graph from the extracted data and stores it in Neo4j.

**Graph Import**
- All entities, relationships, and triples from the JSON files are imported into Neo4j using Cypher queries via `neo4j_import.py`
- Nodes represent entities (papers, authors, concepts, methods)
- Directed edges represent relationships and triples between nodes
- Uses `MERGE` instead of `CREATE` to ensure idempotent imports with no duplicate nodes

**Section Node Patching**
- `fix_all_sections.py` backfills the `raw_text` property on all Section nodes that were created without text content
- `fix_empty_sections.py` patches any remaining Section nodes with empty or null `raw_text` values
- Ensures all nodes in the graph have complete, queryable content

**Graph Schema**
- Node types: `Paper`, `Section`, `Entity`, `Author`, `Concept`
- Relationship types: `HAS_SECTION`, `MENTIONS`, `RELATES_TO`, `CITES`, `AUTHORED_BY`
- Graph supports 7 distinct Cypher query patterns used in the RAG pipeline

---

### Module 3 — Implementation of RAG Pipeline

The third module implements the full 8-step Retrieval-Augmented Generation pipeline.

**Step 1 — Neo4j Fetch (`neo4j_fetch.py`)**
- Queries Neo4j with 7 Cypher queries to retrieve graph documents
- Fetches section text, relationships, citations, and metadata nodes
- Output saved to `graph_documents.json`

**Step 2 — Text Chunk Conversion (`text_conversion.py`)**
- Converts graph documents into 4 types of text chunks:
  - `section` chunks — main paper content
  - `relationship` chunks — entity relationships
  - `citation` chunks — references
  - `metadata` chunks — paper info
- Produces 165 chunks total, saved to `text_chunks.json`

**Step 3 — Embedding Generation (`embeddings.py`)**
- Generates 384-dimensional sentence embeddings using `all-MiniLM-L6-v2`
- All embeddings are L2-normalised for cosine similarity compatibility
- Saved to `embeddings_matrix.npy` and `chunk_embeddings.json`

**Step 4 — FAISS Index Build (`faiss_store.py`)**
- Builds a `IndexFlatIP` (inner product) FAISS vector index from all embeddings
- Equivalent to cosine similarity for normalised vectors
- Index saved to `faiss_index.bin`; position-to-chunk mapping saved to `faiss_metadata.json`

**Step 5 — Query Retrieval (`query_retrieval.py`)**
- Encodes user query using the same sentence transformer model
- Searches FAISS index for top-K most similar chunks (default K=7)
- Supports paper-specific filtering, score thresholds, and deduplication

**Step 6 — LLM Answer Generation (`llm_ans.py`)**
- Sends retrieved chunks as context to Groq API (`llama-3.1-8b-instant`)
- Multiple prompt strategies: RAG answer, AI analysis, additional knowledge, about paper
- All answers cite Paper IDs from the retrieved sources

**Step 7 — Evaluation Metrics (`evaluation.py`)**
Computes 5 metrics on every response:
| Metric | Description |
|---|---|
| Confidence | Weighted average of FAISS retrieval scores |
| Precision@K | Fraction of top-K chunks above relevance threshold |
| Recall@K | Fraction of all relevant chunks that were retrieved |
| Semantic Similarity | Cosine similarity between query and answer embeddings |
| Faithfulness | Sentence-level alignment between answer and source chunks |

**Step 8 — End-to-End Pipeline (`rag_pipeline.py`)**
- Wires all 7 steps into a single terminal-runnable pipeline
- Includes progress logging and error handling at each step

---

### Module 4 — Frontend: Chat Application

The fourth module is a fully-featured Streamlit chat application for interacting with the RAG system.

**Chat Interface**
- Clean conversational UI with persistent chat history
- Paper selector dropdown to focus queries on a specific paper or all papers
- Source chips displayed below each answer showing Paper ID, section, and retrieval score
- Quick suggestion buttons for first-time users
- Stop Generation button to halt mid-stream LLM calls

**Output Modes**
Six output modes available via sidebar:
| Mode | Description |
|---|---|
| 🔀 All Combined | Four separate answer boxes — RAG, AI Analysis, Additional Knowledge, About Paper |
| 📝 Single Combined | One unified flowing answer with four ### sections |
| 📄 RAG Only | Answer strictly from retrieved chunks |
| 🤖 AI Analysis | Deep analytical insight on the selected paper |
| 💡 Additional Knowledge | Broader field connections and real-world implications |
| 📋 About Paper | Structured paper overview relevant to the question |

**Memory System**
- Every Q&A pair is embedded and saved to `chat_memory.json`
- On new queries, cosine similarity is computed against stored memory
- If similarity ≥ 0.80, the matching past answer is injected into the prompt as context
- Memory is capped at 50 entries (oldest removed first)
- `/memory` command displays all stored Q&A pairs

**Irrelevance Detection**
Two-stage filter to block off-topic questions:
- Stage 1: 20+ regex patterns instantly reject general knowledge questions (weather, sports, politics, recipes etc.)
- Stage 2: FAISS score below threshold (0.22) triggers a warning but still attempts to answer

**Slash Commands**
| Command | Description |
|---|---|
| `/overview` | Full structured overview of selected paper |
| `/terms` | 10 key technical terms with definitions |
| `/insights` | Contributions, limitations, and future directions |
| `/summary` | Full 4–5 paragraph academic summary |
| `/compare` | Side-by-side comparison of up to 4 papers |
| `/memory` | Show all stored Q&A memory pairs |
| `/help` | List all available commands |

**Session Management (SQLite)**
- Every chat session is persisted in a local SQLite database (`rag_chat.db`)
- Sessions and messages are stored with full metadata
- Load any past session from the sidebar dropdown
- Rename and delete sessions
- Message count and last updated time shown per session

**PDF Export**
- Export any session or selected messages as a formatted PDF
- Built with ReportLab `BaseDocTemplate`
- Every page has a branded header and page-numbered footer
- Colour-coded section bars per answer type
- 5-column evaluation metrics table per message
- Download button available both in sidebar and inline after each answer

**Paper Management**
- Upload new papers (PDF, DOCX, TXT) directly from the sidebar
- Full pipeline runs automatically on upload: parse → extract → embed → index
- Delete existing papers from the knowledge base with automatic FAISS index rebuild

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Graph Database | Neo4j Desktop |
| Vector Store | FAISS (IndexFlatIP) |
| Embedding Model | all-MiniLM-L6-v2 (Sentence Transformers) |
| LLM | Groq API — llama-3.1-8b-instant |
| NLP | spaCy en_core_web_sm |
| PDF Parsing | pdfplumber, python-docx |
| Chat Persistence | SQLite (via chat_db.py) |
| PDF Generation | ReportLab |
| Frontend | Streamlit |

---

## ⚙️ Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-username/research-rag-assistant.git
cd research-rag-assistant

# 2. Install dependencies
pip install streamlit sentence-transformers faiss-cpu groq numpy \
            reportlab pdfplumber python-docx neo4j spacy
pip install --upgrade joblib scikit-learn sentence-transformers
python -m spacy download en_core_web_sm
```

---

## 🔑 Configuration

Open `config.py` and set your credentials:
```python
GROQ_API_KEY   = "your_groq_api_key_here"
NEO4J_PASSWORD = "your_neo4j_password_here"
```

All file paths, model names, and RAG settings are also controlled from `config.py`.

---

## 🚀 Running the Project

### Step 1 — Prepare Data (run once from project root)
```bash
# Extract entities, relationships, and triples
python backend/entity_extraction.py
python backend/relationship_extraction.py
python backend/triple_extraction.py

# Import into Neo4j (ensure Neo4j Desktop is running)
python backend/neo4j_import.py
python backend/fix_all_sections.py
```

### Step 2 — Build RAG Index (run once from project root)
```bash
python backend/step1_neo4j_fetch.py
python backend/step2_text_conversion.py
python backend/step3_embeddings.py
python backend/step4_faiss_store.py
```

### Step 3 — Launch the App
```bash
streamlit run frontend/app.py
```

---

## 📊 System Stats

| Metric | Value |
|---|---|
| Research papers | 10 |
| Total text chunks | 165 |
| Embedding dimensions | 384 |
| Chunk types | 4 (section, relationship, citation, metadata) |
| Cypher query patterns | 7 |
| Output modes | 6 |
| Evaluation metrics | 5 |
| Slash commands | 7 |

---

## 📄 License

This project was developed as part of the Infosys Springboard Internship Program.

---

## 🙌 Acknowledgements

- [Groq](https://groq.com) for ultra-fast LLM inference
- [Neo4j](https://neo4j.com) for the graph database
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI for vector search
- [Sentence Transformers](https://www.sbert.net) for semantic embeddings
- [Streamlit](https://streamlit.io) for the chat frontend
