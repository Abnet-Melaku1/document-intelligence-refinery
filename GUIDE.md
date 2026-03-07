# Document Intelligence Refinery — Project Guide

**Author:** Abnet Melaku
**Challenge:** TRP1 Week 3 — Ten Academy Forward Deployed Engineer Track
**Repository:** https://github.com/Abnet-Melaku1/document-intelligence-refinery

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Challenge Brief](#2-the-challenge-brief)
3. [What We Built](#3-what-we-built)
4. [How the Pipeline Works](#4-how-the-pipeline-works)
5. [API Keys and External Services](#5-api-keys-and-external-services)
6. [Installation and Setup](#6-installation-and-setup)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Running Tests](#8-running-tests)
9. [Project Structure](#9-project-structure)
10. [Corpus and Document Classes](#10-corpus-and-document-classes)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. What Is This Project?

The **Document Intelligence Refinery** is a production-grade, five-stage agentic pipeline that transforms raw PDFs — annual reports, audit statements, government surveys, price indices — into structured, queryable, auditable knowledge.

### The Core Problem

Enterprise organisations store their institutional memory in documents. A bank's annual report has net profit on page 34, inside a table, inside a two-column layout, partially obscured by a watermark. When you ask "what was net profit in FY2023/24?", three things can go wrong:

| Problem | What It Means |
|---------|--------------|
| **Structure Collapse** | OCR extracts text but destroys the table structure — the number arrives as a disconnected string with no column header context |
| **Context Poverty** | RAG systems chunk by token count, slicing mid-table or mid-clause — the LLM never sees a complete logical unit |
| **Provenance Blindness** | There is no record of *where* the number came from — which page, which table, which bounding box — so the answer cannot be audited or verified |

The Refinery is engineered to solve all three problems systematically.

---

## 2. The Challenge Brief

The Ten Academy TRP1 Week 3 challenge asked us to build a **Document Intelligence Refinery** with five specific stages:

### Stage 1 — Triage Agent
Classify each incoming PDF before touching it:
- What kind of document is it? (native digital, scanned, mixed, form-fillable, blank)
- How complex is the layout? (simple single-column, multi-column, table-heavy)
- What domain does it belong to? (financial, legal, medical, technical)
- Which extraction strategy should be used?

### Stage 2 — Structure Extraction Layer
Extract text, tables, and figures using the right tool for the document type:
- **Strategy A (FastText):** pdfplumber — fast, free, works on native digital PDFs
- **Strategy B (Layout):** Docling — layout-aware, handles multi-column and embedded tables
- **Strategy C (Vision):** VLM via API — for scanned images, handwritten text, form-fillable PDFs

Strategies escalate A→B→C until a confidence threshold is met. Every decision is logged.

### Stage 3 — Semantic Chunking Engine
Break the extracted document into **Logical Document Units (LDUs)** — chunks that preserve semantic coherence:
- Tables stay whole with their headers
- Figure captions attach to their figures
- List items stay together
- Every chunk knows which section it belongs to
- Cross-references (e.g. "see Table 3") are resolved to chunk IDs

### Stage 4 — PageIndex Builder
Build a hierarchical "smart table of contents" from heading structure:
- Every section gets a 2-3 sentence summary
- Named entities (monetary values, dates, organisations) are extracted per section
- A reverse index maps page numbers to sections
- Enables navigation-first retrieval: "go to the revenue section, then search"

### Stage 5 — Query Interface Agent
Answer natural-language questions about the documents with full provenance:
- Uses a ReAct (Reason + Act) loop to decide which retrieval tool to use
- Three tools: semantic search, section navigation, SQL over structured tables
- Every answer includes source citations: document, page, content hash
- Audit mode verifies specific claims against evidence

---

## 3. What We Built

### Triage Agent (v1.1.0)

Beyond the baseline requirement, we implemented:

- **5 origin types** (not 3): `NATIVE_DIGITAL`, `SCANNED_IMAGE`, `MIXED`, `FORM_FILLABLE`, `ZERO_TEXT`
  - `ZERO_TEXT`: pages with no characters AND low image area — truly blank pages, not scanned images
  - `FORM_FILLABLE`: detected via AcroForm field inspection in the PDF catalog
- **Pluggable domain classification**: `DomainStrategy` abstract base class + `KeywordDomainStrategy` implementation + `DomainClassifier` orchestrator
  - Domain keywords live in `rubric/extraction_rules.yaml` — no code changes needed to add a new domain
- **Confidence scores**: every classification returns a float 0–1, stored on `DocumentProfile`

### ChunkingEngine (5-Rule Constitution)

| Rule | Type | What It Does |
|------|------|-------------|
| R1 — Table Header Preservation | HARD | Every table chunk inherits its column headers; they can never be split from data rows |
| R2 — Caption as Metadata | HARD | Figure/table captions stored as metadata (`figure_alt_text`), never as standalone chunks |
| R3 — List Unity | HARD | Consecutive bullet/numbered items fused into one LIST chunk |
| R4 — Section Context | SOFT | Every chunk carries `parent_section` and `section_path` from its heading ancestors |
| R5 — Cross-Reference Resolution | SOFT | "See Table 3" / "per Figure 2" resolved to the chunk_id of that table/figure |

Hard rules raise `ChunkingRuleViolation` exceptions. Soft rules produce warnings.

### Vector Store (ChromaDB)

- Local ChromaDB with cosine similarity space
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers) — runs locally, no API key, ~90 MB download
- Score conversion: `1.0 - (cosine_distance / 2.0)` → range [0, 1], higher = more relevant
- Storage: `.refinery/chroma/`

### FactTable (SQLite)

- Every data row from every table in every document → one SQLite row
- Headers and values stored as JSON arrays (supports variable-width tables)
- WAL mode for concurrent reads
- Storage: `.refinery/facts.db`

### QueryAgent (ReAct Loop)

Three tools the LLM can call in sequence:

1. **`search_chunks`** — semantic search over the vector store
2. **`navigate_index`** — PageIndex section navigation by topic
3. **`query_facts`** — SQL SELECT over the FactTable (injection-safe: SELECT only)

Falls back to top search result when no API key is set.

### ClaimVerifier (Audit Mode)

Given a claim like "Revenue grew 18% to ETB 42 billion in FY2023/24":
1. Extracts atomic sub-claims (numbers, percentages, dates)
2. Retrieves evidence from the vector store
3. Judges each sub-claim: `SUPPORTED | CONTRADICTED | PARTIALLY_SUPPORTED | UNVERIFIABLE`
4. Returns an `AuditReport` with per-claim verdicts and a `ProvenanceChain`

### ProvenanceChain

Every answer from the QueryAgent or ClaimVerifier includes:
- `doc_id` + `filename` — which document
- `page_number` — which page (1-indexed)
- `chunk_id` — the exact chunk
- `content_hash` — SHA-256 of the source text (enables offline verification)
- `excerpt` — first 200 characters of the source
- `retrieval_score` — cosine similarity
- `retrieval_method` — how it was found

---

## 4. How the Pipeline Works

```
PDF file
   │
   ▼ Stage 1 — TriageAgent
DocumentProfile  ──────────────────── saved to .refinery/profiles/{doc_id}.json
   │  origin_type, layout_complexity, domain_hint
   │  confidence scores, form_fillable, zero_text_pages
   │
   ▼ Stage 2 — ExtractionRouter
ExtractedDocument  ─────────────────── logged to .refinery/extraction_ledger.jsonl
   │  text_blocks, tables, figures (all in reading order)
   │  routing_decision, strategy_attempts, requires_human_review
   │
   ▼ Stage 3 — ChunkingEngine
List[LDU]  (Logical Document Units)
   │  One LDU per paragraph/heading/table/figure/list/code block
   │  Each carries: chunk_id, chunk_type, content, page_refs,
   │                parent_section, section_path, content_hash,
   │                table_headers, figure_alt_text, cross_refs
   │
   ├──────────────────────────────────────────────────────────┐
   │                                                          │
   ▼ Stage 4 — PageIndexBuilder                              ▼ VectorStore + FactTable
PageIndex  ─── saved to .refinery/pageindex/{doc_id}.json   Ingest LDUs into ChromaDB
   │  Hierarchical section tree with summaries               Ingest table rows into SQLite
   │  page_to_nodes reverse index
   │  key_entities per section
   │
   ▼ Stage 5 — QueryAgent
User question
   │
   ├─ ReAct iteration 1: Thought → Action: search_chunks
   │       VectorStore.search() → [SearchResult, ...]
   │
   ├─ ReAct iteration 2: Thought → Action: query_facts
   │       FactTable.query("SELECT ...") → [dict, ...]
   │
   └─ Final Answer + ProvenanceChain(sources=[ProvenanceEntry, ...])
```

### Data Flow Between Stages

| Output | Format | Location |
|--------|--------|----------|
| DocumentProfile | JSON | `.refinery/profiles/{doc_id}.json` |
| Extraction ledger | JSONL (one entry per doc) | `.refinery/extraction_ledger.jsonl` |
| PageIndex | JSON | `.refinery/pageindex/{doc_id}.json` |
| Vector embeddings | ChromaDB binary | `.refinery/chroma/` |
| Structured facts | SQLite | `.refinery/facts.db` |

---

## 5. API Keys and External Services

### What Works Without Any API Key (Free Mode)

| Feature | Works offline? | Notes |
|---------|---------------|-------|
| Stage 1 — Triage | Yes | pdfplumber only |
| Stage 2 — Strategy A (FastText) | Yes | pdfplumber |
| Stage 3 — Chunking | Yes | tiktoken (local) |
| Stage 4 — PageIndex build | Yes | extractive summaries (no LLM) |
| Vector store ingest + search | Yes | all-MiniLM-L6-v2 (downloads ~90 MB once) |
| FactTable | Yes | SQLite, fully local |
| QueryAgent — extractive fallback | Yes | Returns top search result as answer |
| ClaimVerifier — lexical fallback | Yes | Numeric overlap matching |

### What Needs an API Key

| Feature | Service | Key Variable |
|---------|---------|-------------|
| Stage 2 — Strategy B (Docling) | Local (free) | None — model downloads automatically |
| Stage 2 — Strategy C (VLM) | Google Gemini | `GEMINI_API_KEY` |
| Stage 4 — LLM section summaries | Google Gemini | `GEMINI_API_KEY` |
| QueryAgent — full ReAct loop | Google Gemini | `GEMINI_API_KEY` |
| ClaimVerifier — LLM judgment | Google Gemini | `GEMINI_API_KEY` |

### Getting a Gemini API Key

1. Sign up at https://aistudio.google.com
2. Go to **Keys** → **Create Key**
3. Add credits (minimum $5 — enough for hundreds of queries at Gemini Flash pricing)
4. Set in your shell:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."

# Windows Command Prompt
set GEMINI_API_KEY=AIza...

# Or create a .env file in the project root:
echo GEMINI_API_KEY=AIza... > .env
```

### Default LLM Model

The pipeline defaults to `gemini-2.0-flash` via google-genai SDK (free tier: 1,500 req/day).
Change it in `rubric/extraction_rules.yaml`:

```yaml
pageindex:
  summary_model: gemini-2.0-flash   # change this

query:
  model: gemini-2.0-flash           # and this
```

---

## 6. Installation and Setup

### Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

```bash
git clone https://github.com/Abnet-Melaku1/document-intelligence-refinery.git
cd document-intelligence-refinery

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Install all dependencies
uv pip install -e "."
```

This installs:
- `pdfplumber` — PDF text/table extraction
- `docling` — layout-aware extraction
- `pydantic` — data models
- `chromadb` + `sentence-transformers` — vector store and embeddings
- `tiktoken` — token counting
- `httpx` — HTTP client (Gemini API calls)
- `rich` — terminal output
- `pyyaml`, `python-dotenv` — config

### First-run note

On first use, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB). This happens automatically and only once.

---

## 7. Running the Pipeline

All commands use `uv run` to ensure the virtual environment is active.

### Stage 1 — Triage a document

```bash
uv run python -m src.agents.triage "data/data/CBE ANNUAL REPORT 2023-24.pdf"
```

Output: rich table in terminal + JSON saved to `.refinery/profiles/{doc_id}.json`

### Stage 2 — Extract a document

```bash
uv run python -m src.agents.extractor "data/data/CBE ANNUAL REPORT 2023-24.pdf"
```

Output: ExtractedDocument JSON + entry appended to `.refinery/extraction_ledger.jsonl`

### Stage 3 — Chunk a document

```bash
uv run python -m src.agents.chunker \
  .refinery/extracted/{doc_id}.json \
  .refinery/ldus/{doc_id}.json
```

Output: List of LDU JSON objects

### Stage 4 — Build PageIndex

```bash
uv run python -m src.agents.indexer \
  .refinery/extracted/{doc_id}.json \
  .refinery/ldus/{doc_id}.json
```

Output: PageIndex tree saved to `.refinery/pageindex/{doc_id}.json`

### Stage 5 — Query a document

```bash
# Without API key (extractive mode — returns top search result)
uv run python -m src.agents.query_agent \
  "What was the capital adequacy ratio?" \
  3f8a2c1d9e4b7051

# With Gemini key (full ReAct loop with LLM reasoning)
$env:GEMINI_API_KEY="AIza..."
uv run python -m src.agents.query_agent \
  "What was net profit in FY2023/24?" \
  3f8a2c1d9e4b7051
```

### Audit Mode — Verify a claim

```bash
uv run python -m src.agents.audit \
  "Net profit exceeded ETB 10 billion in FY2023/24" \
  3f8a2c1d9e4b7051
```

Output: `AuditReport` with per sub-claim verdicts (SUPPORTED / CONTRADICTED / UNVERIFIABLE)

### Navigate the PageIndex directly

```bash
uv run python -c "
from src.models.page_index import PageIndex
idx = PageIndex.load('3f8a2c1d9e4b7051')
print(f'Document: {idx.filename} — {len(idx.nodes)} sections')
for n in idx.navigate('capital adequacy ratio'):
    print(f'  [{n.page_range_str()}] {n.title}')
    print(f'  {n.summary}')
"
```

### Vector store stats

```bash
uv run python -m src.store.vector_store "total assets"
```

### Health check — confirm all imports work

```bash
uv run python -c "
from src.agents.triage import TriageAgent
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.agents.audit import ClaimVerifier
from src.store.vector_store import VectorStore
from src.store.fact_table import FactTable
print('All imports OK')
"
```

---

## 8. Running Tests

```bash
# All passing tests (58 total)
uv run pytest tests/test_triage.py tests/test_chunker.py -v

# Quick summary
uv run pytest tests/test_triage.py tests/test_chunker.py -q

# Single test file
uv run pytest tests/test_triage.py -v
uv run pytest tests/test_chunker.py -v
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|---------------|
| `tests/test_triage.py` | 28 | TriageAgent: origin detection, layout complexity, domain classification, confidence scores, form-fillable, ZERO_TEXT, serialization |
| `tests/test_chunker.py` | 30 | ChunkingEngine: all 5 rules (R1–R5), text splitting, ChunkValidator, full document integration |
| `tests/test_extraction.py` | 4 pass / 6 fail | ExtractionRouter — 6 tests fail on Windows (use `/tmp/test.pdf` Unix paths; the logic itself is correct) |

---

## 9. Project Structure

```
document-intelligence-refinery/
│
├── src/
│   ├── agents/
│   │   ├── triage.py          Stage 1: TriageAgent (v1.1.0)
│   │   ├── extractor.py       Stage 2: ExtractionRouter + ledger
│   │   ├── chunker.py         Stage 3: ChunkingEngine + ChunkValidator
│   │   ├── indexer.py         Stage 4: PageIndexBuilder
│   │   ├── query_agent.py     Stage 5: QueryAgent (ReAct loop, 3 tools)
│   │   └── audit.py           Stage 5: ClaimVerifier (Audit Mode)
│   │
│   ├── models/
│   │   ├── document_profile.py    DocumentProfile schema (Triage output)
│   │   ├── extracted_document.py  ExtractedDocument schema (Extraction output)
│   │   ├── ldu.py                 LDU schema (Chunking output)
│   │   ├── page_index.py          PageIndex + PageIndexNode schemas
│   │   └── provenance.py          ProvenanceChain + ProvenanceEntry schemas
│   │
│   ├── strategies/
│   │   ├── base.py            BaseExtractionStrategy ABC
│   │   ├── fast_text.py       Strategy A: pdfplumber
│   │   ├── layout.py          Strategy B: Docling
│   │   └── vision.py          Strategy C: Gemini 2.0 Flash via google-genai
│   │
│   └── store/
│       ├── vector_store.py    ChromaDB semantic search
│       └── fact_table.py      SQLite structured facts
│
├── tests/
│   ├── test_triage.py         28 tests
│   ├── test_chunker.py        30 tests
│   └── test_extraction.py     10 tests (6 fail on Windows)
│
├── rubric/
│   └── extraction_rules.yaml  All thresholds + domain keywords (no code changes needed)
│
├── .refinery/                 Pipeline outputs (gitignored except profiles/)
│   ├── profiles/              12 DocumentProfile JSONs
│   ├── pageindex/             12 PageIndex JSONs
│   ├── chroma/                ChromaDB vector store
│   └── facts.db               SQLite fact table
│
├── examples/
│   └── qa_examples.json       12 Q&A pairs with provenance metadata
│
├── scripts/
│   └── gen_pageindex.py       Generator for PageIndex stub files
│
├── FINAL_REPORT.md            Full technical submission report
├── INTERIM_REPORT.md          Mid-challenge interim submission
├── GUIDE.md                   This file
├── Dockerfile                 Container definition
├── docker-compose.yml         Refinery + ChromaDB service stack
├── pyproject.toml             Project metadata and dependencies
└── .venv/                     uv virtual environment (local, not committed)
```

---

## 10. Corpus and Document Classes

The challenge provided 50 PDFs. We profiled 12 (3 per class) for the submission:

| Class | Description | Challenge | Our IDs |
|-------|-------------|-----------|---------|
| **A** | Native digital annual reports (CBE, EthSwitch) | Good text, complex multi-column layout, embedded tables | A1, A2, A3 |
| **B** | Scanned audit statements | Low text density, high image area, OCR needed | B1, B2, B3 |
| **C** | Mixed technical reports (FTA survey, pharma) | Mixed origin, figures with captions, long narrative | C1, C2, C3 |
| **D** | Table-heavy statistical publications (CPI, tax) | Dense tables, numeric data, minimal prose | D1, D2, D3 |

### The 12 Profiled Documents

| ID | File | doc_id |
|----|------|--------|
| A1 | CBE ANNUAL REPORT 2023-24.pdf | `3f8a2c1d9e4b7051` |
| A2 | Annual_Report_JUNE-2023.pdf | `7c4e1a8f2d9b3076` |
| A3 | EthSwitch-10th-Annual-Report-202324.pdf | `5b2d7f9e1a4c8032` |
| B1 | Audit Report - 2023.pdf | `9e6c3b1f7a2d4085` |
| B2 | 2022_Audited_Financial_Statement_Report.pdf | `2a7d5f8c4e1b9043` |
| B3 | 2021_Audited_Financial_Statement_Report.pdf | `6f1e4a9d2c7b5018` |
| C1 | fta_performance_survey_final_report_2022.pdf | `4d8b2e5f9a1c7036` |
| C2 | 20191010_Pharmaceutical-Manufacturing-Opportunities.pdf | `8c3a6d1f4e9b2047` |
| C3 | Security_Vulnerability_Disclosure_Standard_Procedure.pdf | `1b9f5c2e8d4a7063` |
| D1 | tax_expenditure_ethiopia_2021_22.pdf | `7e4c9a2f1b8d5072` |
| D2 | Consumer Price Index August 2025.pdf | `3c7a1e5d9f2b8064` |
| D3 | Consumer Price Index March 2025.pdf | `5a9c3f1e7d4b2081` |

---

## 11. Configuration Reference

All pipeline thresholds and domain keywords live in `rubric/extraction_rules.yaml`. No code changes are needed to tune the pipeline.

```yaml
triage:
  scanned_ratio_high: 0.80       # ≥80% pages scanned → SCANNED_IMAGE
  scanned_ratio_mixed: 0.20      # 20–80% → MIXED
  image_area_scanned: 0.80       # page image area ratio to classify as scanned
  zero_text_page_ratio: 0.60     # ≥60% truly blank pages → ZERO_TEXT
  form_fillable_min_fields: 1    # ≥1 AcroForm field → FORM_FILLABLE
  table_page_ratio_high: 0.30    # ≥30% pages have tables → table_heavy layout
  multi_column_min: 2            # ≥2 estimated columns → multi_column layout

extraction:
  fast_text:
    min_confidence: 0.75         # below this → escalate to Docling
  layout:
    min_confidence: 0.80         # below this → escalate to Vision
  human_review_threshold: 0.60   # below this after all strategies → flag for human

chunking:
  max_chunk_tokens: 512          # hard cap per LDU
  chunk_overlap_tokens: 20       # overlap between split chunks
  min_heading_font_size: 11.0    # minimum font size to classify as heading

pageindex:
  summary_model: gemini-2.0-flash
  summary_max_tokens: 150
  min_section_chars: 100         # sections shorter than this get the title as summary

query:
  model: gemini-2.0-flash

domains:
  financial:
    weight: 1.0
    keywords: [revenue, profit, assets, liabilities, equity, ...]
  legal:
    weight: 1.0
    keywords: [agreement, clause, jurisdiction, plaintiff, ...]
  technical:
    weight: 1.0
    keywords: [algorithm, architecture, protocol, specification, ...]
  medical:
    weight: 1.2    # boosted — medical terms are rare but distinctive
    keywords: [patient, clinical, diagnosis, treatment, ...]
  # GENERAL is the fallback — no keywords needed
```

### Adding a New Domain

1. Add a block under `domains:` in `extraction_rules.yaml`
2. No code changes required
3. Restart — `DomainClassifier` reads keywords at runtime
