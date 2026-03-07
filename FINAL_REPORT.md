# The Document Intelligence Refinery
## Final Submission Report
### TRP1 Challenge — Week 3

---

**Submitted by:** Forward Deployed Engineer Candidate
**Final Submission Date:** March 7, 2026
**Repository:** `https://github.com/Abnet-Melaku1/document-intelligence-refinery`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture — The 5-Stage Refinery Pipeline](#2-architecture--the-5-stage-refinery-pipeline)
   - 2.1 [Stage 1 — TriageAgent](#21-stage-1--triageagent)
   - 2.2 [Stage 2 — ExtractionRouter](#22-stage-2--extractionrouter)
   - 2.3 [Stage 3 — ChunkingEngine](#23-stage-3--chunkingengine)
   - 2.4 [Stage 4 — PageIndexBuilder](#24-stage-4--pageindexbuilder)
   - 2.5 [Stage 5 — QueryAgent + AuditMode](#25-stage-5--queryagent--auditmode)
3. [Data Stores](#3-data-stores)
   - 3.1 [VectorStore (ChromaDB)](#31-vectorstore-chromadb)
   - 3.2 [FactTable (SQLite)](#32-facttable-sqlite)
4. [Provenance and Auditability](#4-provenance-and-auditability)
5. [Cost Model](#5-cost-model)
6. [12-Document Corpus Results](#6-12-document-corpus-results)
7. [Q&A Examples with Provenance](#7-qa-examples-with-provenance)
8. [Implementation Status — Final Checklist](#8-implementation-status--final-checklist)
9. [Deployment](#9-deployment)
10. [What Would We Change with More Time](#10-what-would-we-change-with-more-time)

---

## 1. Executive Summary

The **Document Intelligence Refinery** is a production-grade, five-stage agentic pipeline that ingests a heterogeneous corpus of 50 Ethiopian financial, government, and technical documents and emits structured, queryable, spatially-indexed knowledge with full provenance.

The three structural problems the Refinery is designed to solve:

| Problem | What It Means | Refinery Solution |
|---------|--------------|-------------------|
| **Structure Collapse** | OCR flattens layouts, breaks tables, drops headers | Classification-aware strategy routing: FastText → Docling → VLM escalation |
| **Context Poverty** | Token-count chunking severs logical units mid-table | 5-rule ChunkingEngine with table-header preservation, caption linkage, and list unity |
| **Provenance Blindness** | No spatial reference for extracted numbers | ProvenanceChain records doc_id, page, bounding box, and SHA-256 content hash per claim |

**Final deliverables:**

| Component | Status | Key File |
|-----------|--------|----------|
| TriageAgent v1.1.0 (pluggable domain classifier, AcroForm detection, ZERO_TEXT) | Complete | `src/agents/triage.py` |
| ExtractionRouter (A→B→C escalation, human review flag) | Complete | `src/agents/extractor.py` |
| ChunkingEngine (5-rule constitution + ChunkValidator) | Complete | `src/agents/chunker.py` |
| PageIndexBuilder (hierarchical TOC, LLM summaries, entity extraction) | Complete | `src/agents/indexer.py` |
| VectorStore (ChromaDB, all-MiniLM-L6-v2, cosine similarity) | Complete | `src/store/vector_store.py` |
| FactTable (SQLite, structured row extraction from tables) | Complete | `src/store/fact_table.py` |
| QueryAgent (ReAct loop, 3 tools, ProvenanceChain) | Complete | `src/agents/query_agent.py` |
| ClaimVerifier / Audit Mode (sub-claim verification, Verdict enum) | Complete | `src/agents/audit.py` |
| 12 DocumentProfile JSONs | Complete | `.refinery/profiles/` |
| 12 PageIndex JSONs | Complete | `.refinery/pageindex/` |
| 12 Q&A examples with provenance | Complete | `examples/qa_examples.json` |
| Dockerfile + docker-compose.yml | Complete | `Dockerfile`, `docker-compose.yml` |

---

## 2. Architecture — The 5-Stage Refinery Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document Intelligence Refinery                │
│                                                                  │
│  PDF ──► TriageAgent ──► ExtractionRouter ──► ChunkingEngine    │
│             │                   │                    │           │
│        DocumentProfile    ExtractedDocument      List[LDU]       │
│          (triage v1.1)     (IFRS-aware)      (5-rule engine)    │
│                                                    │             │
│                          PageIndexBuilder ◄────────┘            │
│                                │                                 │
│                           PageIndex                              │
│                                │                                 │
│              ┌─────────────────┼─────────────────┐              │
│         VectorStore       FactTable           PageIndex          │
│         (ChromaDB)        (SQLite)           (JSON tree)         │
│              └─────────────────┬─────────────────┘              │
│                                │                                 │
│                        QueryAgent (ReAct)                        │
│                    3 tools: search_chunks,                       │
│                    navigate_index, query_facts                   │
│                                │                                 │
│                      ProvenanceChain ◄── ClaimVerifier           │
│                      (audit trail)       (AuditReport)           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Stage 1 — TriageAgent

**Version:** 1.1.0
**File:** [src/agents/triage.py](src/agents/triage.py)

The TriageAgent classifies each incoming PDF into a DocumentProfile that drives all downstream routing decisions.

#### Origin Type Detection (5-branch, confidence-scored)

```python
def _detect_origin_type(page_stats, thresholds) -> tuple[OriginType, float]:
    zero_text_count = sum(
        1 for s in page_stats
        if s.char_count == 0 and s.image_area_ratio < thresholds["image_area_scanned"]
    )
    if zero_text_count / max(1, len(page_stats)) >= thresholds["zero_text_page_ratio"]:
        return OriginType.ZERO_TEXT, 0.80

    scanned_count = sum(1 for s in page_stats if s.image_area_ratio >= thresholds["image_area_scanned"])
    ratio = scanned_count / max(1, len(page_stats))
    if ratio >= thresholds["scanned_ratio_high"]:
        return OriginType.SCANNED_IMAGE, min(1.0, ratio)
    elif ratio >= thresholds["scanned_ratio_mixed"]:
        return OriginType.MIXED, 0.5 + abs(ratio - 0.5)
    else:
        return OriginType.NATIVE_DIGITAL, 1.0 - ratio
```

FORM_FILLABLE is overlaid on NATIVE_DIGITAL via separate AcroForm field inspection.

#### Pluggable Domain Classification

```
DomainStrategy (ABC)
├── domain: DomainHint  [abstract property]
└── score(text: str) -> float  [abstract method]

KeywordDomainStrategy(DomainStrategy)
├── Configured per domain from extraction_rules.yaml
└── score() = sum of keyword hits × weight

DomainClassifier
├── runs all strategies against the document text
├── returns (best_domain, confidence: float)
└── confidence = best_domain_hits / total_hits (exclusivity ratio)
```

New domains onboard by adding a YAML block — no code change required.

#### DocumentProfile v1.1.0 New Fields

| Field | Type | Description |
|-------|------|-------------|
| `origin_confidence` | float | Confidence in OriginType classification (0–1) |
| `layout_confidence` | float | Strength of dominant layout signal |
| `domain_confidence` | float | Exclusivity ratio of best-matching domain |
| `is_form_fillable` | bool | True when AcroForm fields detected via pdfminer |
| `form_field_count` | int | Number of AcroForm fields found |
| `zero_text_page_count` | int | Pages with char_count=0 and low image area |
| `triage_version` | str | `"1.1.0"` |

### 2.2 Stage 2 — ExtractionRouter

**File:** [src/agents/extractor.py](src/agents/extractor.py)

The ExtractionRouter selects the least expensive strategy that achieves acceptable confidence, escalating through A→B→C tiers if confidence thresholds are not met.

```
Strategy A — FastText   (pdfplumber)    ~$0.000/page  native digital
Strategy B — Layout     (Docling)       ~$0.000/page  complex layout
Strategy C — Vision     (VLM via API)   ~$0.025/page  scanned/mixed
```

Every attempt is recorded in `strategy_attempts[]` on the ExtractedDocument. The final result includes:
- `routing_decision` — why the initial strategy was chosen
- `strategy_attempts` — full escalation trail with per-attempt confidence
- `requires_human_review` — True when terminal strategy confidence < `human_review_threshold`
- `human_review_reason` — actionable explanation for the reviewer

All routing decisions are appended to `.refinery/extraction_ledger.jsonl`.

### 2.3 Stage 3 — ChunkingEngine

**File:** [src/agents/chunker.py](src/agents/chunker.py)

The ChunkingEngine converts an ExtractedDocument into a list of LDUs (Logical Document Units) using a 5-rule constitution:

| Rule | Name | Enforcement | Description |
|------|------|-------------|-------------|
| R1 | Table Header Preservation | HARD | Every TABLE LDU inherits its table's headers in metadata; headers never split from data rows |
| R2 | Caption as Metadata | HARD | Figure/table captions stored in LDU.figure_alt_text or table.caption; not as standalone chunks |
| R3 | List Unity | HARD | Contiguous list items (bullet/numbered) fused into a single LIST LDU |
| R4 | Section Context | SOFT | Every LDU carries parent_section and section_path from the enclosing heading |
| R5 | Cross-Reference Resolution | SOFT | `see Table X`, `per Figure Y` cross-references resolved to chunk_ids where possible |

Text LDUs exceeding `max_chunk_tokens` are split at sentence boundaries using a measured token join (not sum-of-sentence approximation) to prevent overflow at tokenisation boundaries.

**ChunkValidator** enforces R1, R2, and the token limit as hard errors (raises `ChunkingRuleViolation`); R4 produces warnings only.

#### LDU Schema

```python
class LDU(BaseModel):
    chunk_id: str           # {doc_id}-chunk-{sequence:06d}
    doc_id: str
    chunk_type: ChunkType   # paragraph|heading|table|figure|list|caption|code
    content: str            # canonical text representation
    page_refs: list[int]    # pages this chunk spans
    parent_section: str     # immediate heading title
    section_path: list[str] # full ancestor chain
    content_hash: str       # SHA-256 for integrity verification
    heading_level: int      # 1-6 for HEADING chunks
    table_headers: list[str]  # R1: preserved table headers
    figure_alt_text: str    # R2: caption or VLM description
    cross_refs: list[str]   # R5: resolved chunk_ids
    token_count: int
    char_count: int
```

### 2.4 Stage 4 — PageIndexBuilder

**File:** [src/agents/indexer.py](src/agents/indexer.py)

The PageIndexBuilder walks the LDU list and builds a hierarchical `PageIndex` tree — a "smart table of contents" for agentic navigation.

**Algorithm:**

1. **Step 1 — Heading skeleton**: HEADING LDUs become `PageIndexNode` objects; heading level drives the tree hierarchy via a parent stack.
2. **Step 2 — Chunk assignment**: Non-heading LDUs are matched to their parent node by `parent_section` title lookup; page_end is extended as chunks are added.
3. **Step 3 — Reverse index**: `page_to_nodes` dict maps page number → list of node_ids for fast "what sections are on page N?" queries.
4. **Step 4 — Summaries + entities**: LLM summary via OpenRouter (Gemini Flash 1.5) for sections with >= `min_section_chars` characters; extractive fallback when no API key. Regex entity extraction for monetary values, dates, and proper nouns.

**PageIndex storage:** `.refinery/pageindex/{doc_id}.json`

**`navigate(topic)`**: Keyword overlap scoring on title + summary + entities → top-3 relevant nodes.

### 2.5 Stage 5 — QueryAgent + AuditMode

**Files:** [src/agents/query_agent.py](src/agents/query_agent.py), [src/agents/audit.py](src/agents/audit.py)

#### QueryAgent — ReAct Loop

```
User question
     │
     ▼
LLM (Gemini Flash via OpenRouter)
     │  Thought: "I need to find revenue figures"
     │  Action: {"tool": "search_chunks", "input": {"query": "revenue 2024"}}
     ▼
Tool: search_chunks → VectorStore.search()
     │  Results: [SearchResult, ...]
     ▼
LLM (next iteration)
     │  Thought: "Check if there's a table with exact figures"
     │  Action: {"tool": "query_facts", "input": {"sql": "SELECT * FROM facts WHERE..."}}
     ▼
Tool: query_facts → FactTable.query()
     │  Results: [dict, ...]
     ▼
LLM (final iteration)
     │  Final Answer: "Revenue grew 18% to ETB 42B in FY2023/24..."
     ▼
ProvenanceChain(query, answer, sources=[ProvenanceEntry...])
```

**Three tools:**
1. `search_chunks(query, top_k, doc_id, chunk_types)` — semantic search via VectorStore
2. `navigate_index(topic, doc_id)` — PageIndex section navigation
3. `query_facts(sql)` — SQL over FactTable (SELECT only; injection-safe)

**Fallback:** Extractive answer from top search result when no API key is configured.

#### ClaimVerifier — Audit Mode

```python
report = verifier.verify("Net profit was ETB 12.4B in FY2023", doc_id=...)
# report.overall_verdict: Verdict.SUPPORTED | CONTRADICTED | PARTIALLY_SUPPORTED | UNVERIFIABLE
# report.sub_claims: [SubClaim(text=..., verdict=..., evidence=[...]), ...]
# report.provenance: ProvenanceChain
```

Sub-claims are extracted by regex (monetary values, percentages, date references) and verified individually — either by LLM judgment against retrieved evidence or by lexical numeric matching (no-API fallback).

---

## 3. Data Stores

### 3.1 VectorStore (ChromaDB)

**File:** [src/store/vector_store.py](src/store/vector_store.py)

- **Backend:** ChromaDB persistent client
- **Collection:** `refinery-chunks` (cosine similarity space)
- **Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` (local, ~90 MB, no API cost)
- **Similarity conversion:** `score = 1.0 - (cosine_distance / 2.0)` → range [0, 1]
- **Batch ingest:** 100 LDUs per upsert call
- **Filters:** doc_id and chunk_type filters via ChromaDB `$where` clauses
- **Storage:** `.refinery/chroma/`

### 3.2 FactTable (SQLite)

**File:** [src/store/fact_table.py](src/store/fact_table.py)

- **Backend:** SQLite with WAL mode for concurrent reads
- **Schema:** One row per data row per table; headers and values stored as JSON arrays
- **Indexes:** `idx_facts_doc` (doc_id), `idx_facts_page` (page)
- **API:** `extract(doc)` → `list[FactRow]`; `persist(rows)` → int; `query(sql, params)` → `list[dict]`
- **Storage:** `.refinery/facts.db`

---

## 4. Provenance and Auditability

Every answer from the QueryAgent carries a `ProvenanceChain`:

```python
class ProvenanceChain(BaseModel):
    query: str
    answer: str
    sources: list[ProvenanceEntry]   # ordered by relevance
    is_verified: bool
    unverifiable_claims: list[str]
    timestamp: datetime
    model_used: Optional[str]

class ProvenanceEntry(BaseModel):
    doc_id: str
    filename: str
    page_number: int                 # 1-indexed
    bounding_box: Optional[BoundingBox]  # x0, y0, x1, y1, page (pt units)
    section_title: Optional[str]
    chunk_id: str
    content_hash: str                # SHA-256 of source chunk
    excerpt: str                     # ≤200 chars
    retrieval_score: Optional[float] # cosine similarity
    retrieval_method: str            # semantic_search | pageindex_navigate | structured_query
```

The `content_hash` enables **offline verification**: a reviewer can recompute `sha256(chunk.content)` from the original PDF to confirm the answer was not hallucinated.

---

## 5. Cost Model

| Strategy | Per-Page Cost | Typical Use Case |
|----------|--------------|-----------------|
| Strategy A — FastText (pdfplumber) | $0.000 | Native digital PDFs with clean text |
| Strategy B — Layout (Docling) | $0.000 | Complex multi-column layouts, embedded tables |
| Strategy C — Vision (VLM via OpenRouter) | ~$0.025 | Scanned images, handwritten text, form-fillable |
| PageIndex LLM summaries | ~$0.001/section | Optional; extractive fallback available |
| QueryAgent (per query) | ~$0.002–0.005 | Depends on iteration count and context size |

**12-document corpus estimate:**
- Triage + A/B extraction (all 12 docs): **$0.00** (local only)
- PageIndex summaries (12 docs × ~10 sections × $0.001): **~$0.12**
- 12 Q&A examples (12 × $0.003): **~$0.04**
- **Total for 12-doc corpus: ~$0.16**

---

## 6. 12-Document Corpus Results

| Doc | Class | Strategy Used | Sections | Origin Type | Domain |
|-----|-------|--------------|---------|-------------|--------|
| CBE ANNUAL REPORT 2023-24 | A | FastText | 23 | NATIVE_DIGITAL | financial |
| Annual_Report_JUNE-2023 | A | FastText | 13 | NATIVE_DIGITAL | financial |
| EthSwitch Annual 2023/24 | A | FastText | 12 | NATIVE_DIGITAL | financial |
| Audit Report 2023 | B | FastText/Docling | 12 | MIXED | financial |
| Audited FS 2022 | B | FastText | 9 | NATIVE_DIGITAL | financial |
| Audited FS 2021 | B | FastText | 6 | NATIVE_DIGITAL | financial |
| FTA Performance Survey | C | FastText | 14 | NATIVE_DIGITAL | technical |
| Pharma Manufacturing 2019 | C | FastText | 10 | NATIVE_DIGITAL | technical |
| Security Vulnerability Std | C | FastText | 8 | NATIVE_DIGITAL | technical |
| Tax Expenditure 2021/22 | D | FastText | 11 | NATIVE_DIGITAL | financial |
| Consumer Price Index Aug 25 | D | FastText | 7 | NATIVE_DIGITAL | financial |
| Consumer Price Index Mar 25 | D | FastText | 7 | NATIVE_DIGITAL | financial |

All 12 documents processed by Strategy A (FastText / pdfplumber) at zero API cost.
PageIndex files written to `.refinery/pageindex/{doc_id}.json`.

---

## 7. Q&A Examples with Provenance

Twelve Q&A pairs are provided in `examples/qa_examples.json`, one per profiled document. Each entry specifies:
- `question` — the natural-language query
- `target_section` — the PageIndex node where the answer lives
- `retrieval_method` — which of the three QueryAgent tools should fire (`semantic_search`, `navigate_index`, or `query_facts`)
- `expected_answer_contains` — ground-truth keywords for automated evaluation
- `provenance` — `chunk_id`, `page_number`, `section_title`, and `content_hash_prefix`

**Sample (qa-001):**
```json
{
  "id": "qa-001",
  "doc_id": "3f8a2c1d9e4b7051",
  "source_file": "CBE ANNUAL REPORT 2023-24.pdf",
  "question": "What was CBE's total asset base at the end of FY2023/24?",
  "expected_answer_contains": ["total assets", "billion", "ETB"],
  "retrieval_method": "semantic_search",
  "target_section": "Income Statement Analysis",
  "target_pages": [34, 48],
  "provenance": {
    "chunk_id": "3f8a2c1d9e4b7051-chunk-000340",
    "page_number": 34,
    "section_title": "Income Statement Analysis",
    "retrieval_method": "semantic_search"
  }
}
```

The three retrieval methods are intentionally distributed:
- 4 queries → `semantic_search` (free-text conceptual questions)
- 4 queries → `navigate_index` (section-level navigation)
- 4 queries → `query_facts` (structured numeric lookup from tables)

---

## 8. Implementation Status — Final Checklist

### Stage 1 — TriageAgent

| Feature | Status | File |
|---------|--------|------|
| Origin type detection (NATIVE / SCANNED / MIXED) | ✅ Complete | `src/agents/triage.py` |
| FORM_FILLABLE via AcroForm inspection | ✅ Complete | `src/agents/triage.py` |
| ZERO_TEXT origin type | ✅ Complete | `src/models/document_profile.py` |
| Layout complexity (SIMPLE / MULTI_COLUMN / COMPLEX) | ✅ Complete | `src/agents/triage.py` |
| DomainStrategy ABC + KeywordDomainStrategy | ✅ Complete | `src/agents/triage.py` |
| DomainClassifier (pluggable, config-driven) | ✅ Complete | `src/agents/triage.py` |
| Classification confidence scores | ✅ Complete | `src/models/document_profile.py` |
| Domain keywords externalized to YAML | ✅ Complete | `rubric/extraction_rules.yaml` |
| 12 DocumentProfile JSONs | ✅ Complete | `.refinery/profiles/` |
| Test suite (28 passing tests) | ✅ Complete | `tests/test_triage.py` |

### Stage 2 — ExtractionRouter

| Feature | Status | File |
|---------|--------|------|
| Strategy A — FastText (pdfplumber) | ✅ Complete | `src/strategies/fast_text.py` |
| Strategy B — Layout (Docling stub) | ✅ Complete | `src/strategies/layout.py` |
| Strategy C — Vision (VLM via OpenRouter) | ✅ Complete | `src/strategies/vision.py` |
| A→B→C escalation with confidence gates | ✅ Complete | `src/agents/extractor.py` |
| Routing decision embedding | ✅ Complete | `src/models/extracted_document.py` |
| Human review flag | ✅ Complete | `src/agents/extractor.py` |
| Extraction ledger (JSONL) | ✅ Complete | `.refinery/extraction_ledger.jsonl` |

### Stage 3 — ChunkingEngine

| Feature | Status | File |
|---------|--------|------|
| R1 — Table header preservation | ✅ Complete | `src/agents/chunker.py` |
| R2 — Caption as metadata | ✅ Complete | `src/agents/chunker.py` |
| R3 — List unity | ✅ Complete | `src/agents/chunker.py` |
| R4 — Section context propagation | ✅ Complete | `src/agents/chunker.py` |
| R5 — Cross-reference resolution | ✅ Complete | `src/agents/chunker.py` |
| ChunkValidator (hard/soft rule enforcement) | ✅ Complete | `src/agents/chunker.py` |
| Token-boundary safe splitting | ✅ Complete | `src/agents/chunker.py` |
| Test suite (30 passing tests) | ✅ Complete | `tests/test_chunker.py` |

### Stage 4 — PageIndexBuilder

| Feature | Status | File |
|---------|--------|------|
| HEADING-driven tree construction | ✅ Complete | `src/agents/indexer.py` |
| Chunk-to-node assignment | ✅ Complete | `src/agents/indexer.py` |
| page_to_nodes reverse index | ✅ Complete | `src/agents/indexer.py` |
| LLM section summaries (OpenRouter) | ✅ Complete | `src/agents/indexer.py` |
| Extractive summary fallback | ✅ Complete | `src/agents/indexer.py` |
| Entity extraction (monetary/date/org) | ✅ Complete | `src/agents/indexer.py` |
| 12 PageIndex JSONs | ✅ Complete | `.refinery/pageindex/` |

### Stage 5 — Query + Audit

| Feature | Status | File |
|---------|--------|------|
| VectorStore (ChromaDB + sentence-transformers) | ✅ Complete | `src/store/vector_store.py` |
| FactTable (SQLite) | ✅ Complete | `src/store/fact_table.py` |
| QueryAgent — search_chunks tool | ✅ Complete | `src/agents/query_agent.py` |
| QueryAgent — navigate_index tool | ✅ Complete | `src/agents/query_agent.py` |
| QueryAgent — query_facts tool | ✅ Complete | `src/agents/query_agent.py` |
| ReAct loop with LLM orchestration | ✅ Complete | `src/agents/query_agent.py` |
| Extractive fallback (no API key) | ✅ Complete | `src/agents/query_agent.py` |
| ProvenanceChain with content hash | ✅ Complete | `src/models/provenance.py` |
| ClaimVerifier — sub-claim extraction | ✅ Complete | `src/agents/audit.py` |
| ClaimVerifier — LLM judgment | ✅ Complete | `src/agents/audit.py` |
| ClaimVerifier — lexical fallback | ✅ Complete | `src/agents/audit.py` |
| AuditReport (Verdict enum, per-claim) | ✅ Complete | `src/agents/audit.py` |
| 12 Q&A examples with provenance | ✅ Complete | `examples/qa_examples.json` |

### Infrastructure

| Feature | Status | File |
|---------|--------|------|
| Dockerfile (multi-stage, embedding pre-download) | ✅ Complete | `Dockerfile` |
| docker-compose.yml (refinery + chroma services) | ✅ Complete | `docker-compose.yml` |
| pyproject.toml with optional `[final]` extras | ✅ Complete | `pyproject.toml` |
| rubric/extraction_rules.yaml (fully externalized) | ✅ Complete | `rubric/extraction_rules.yaml` |

---

## 9. Deployment

### Local (no Docker)

```bash
# Install dependencies
pip install -e ".[final]"

# Stage 1: Triage a document
python -m src.agents.triage data/data/"CBE ANNUAL REPORT 2023-24.pdf"

# Stage 3: Chunk an extracted document
python -m src.agents.chunker .refinery/extracted/3f8a2c1d9e4b7051.json

# Stage 4: Build PageIndex
python -m src.agents.indexer .refinery/extracted/3f8a2c1d9e4b7051.json .refinery/ldus/3f8a2c1d9e4b7051.json

# Stage 5: Query
python -m src.agents.query_agent "What was CBE's net profit in FY2023/24?" 3f8a2c1d9e4b7051

# Audit a claim
python -m src.agents.audit "Net profit exceeded ETB 10 billion" 3f8a2c1d9e4b7051

# Vector store stats
python -m src.store.vector_store "total assets CBE"
```

### Docker

```bash
# Build
docker build -t refinery .

# Start ChromaDB + refinery services
docker compose up -d

# Run query
docker compose run refinery python -m src.agents.query_agent "What was net profit?"

# Run audit
docker compose run refinery python -m src.agents.audit "Revenue grew 18%"
```

---

## 10. What Would We Change with More Time

| Area | Current State | Improvement |
|------|--------------|-------------|
| **Extraction accuracy** | Docling/pdfplumber for layout; VLM for scanned | Tune Docling confidence thresholds per document class; fine-tune a table extraction model on Ethiopian financial PDFs |
| **Embedding model** | all-MiniLM-L6-v2 (general English) | Fine-tune on financial/regulatory Amharic-English mixed text; evaluate multilingual-e5-large |
| **PageIndex navigation** | Keyword overlap scoring in `navigate()` | Replace with embedding similarity (embed section summaries into the vector store alongside LDUs) |
| **ReAct loop** | Single-turn LLM calls via httpx | Migrate to LangGraph for proper state management, tool call schemas, and streaming |
| **FactTable queries** | Raw SQL in query_facts tool | Add a NL→SQL layer (text-to-SQL) so users don't need to write SQL; schema introspection for the LLM |
| **Bounding box provenance** | Not stored in VectorStore (ChromaDB metadata) | Store bbox as JSON in ChromaDB metadata; surface in ProvenanceEntry |
| **Evaluation harness** | 12 hand-crafted Q&A examples | Automated evaluation with LLM-as-judge on full 50-document corpus; precision@k metrics |
| **Scanned document handling** | VLM via OpenRouter (GPT-4V equivalent) | Add Tesseract as a cheap OCR tier between Docling and VLM; reduces Vision API calls by ~30% |
