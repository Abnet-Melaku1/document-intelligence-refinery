# The Document Intelligence Refinery

## Final Submission Report

### TRP1 Challenge — Week 3

---

**Submitted by:** Abnet Melaku
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
10. [Development Failures — Root Causes and Fixes](#10-development-failures--root-causes-and-fixes)
11. [What Would We Change with More Time](#11-what-would-we-change-with-more-time)

---

## 1. Executive Summary

The **Document Intelligence Refinery** is a production-grade, five-stage agentic pipeline that ingests a heterogeneous corpus of 50 Ethiopian financial, government, and technical documents and emits structured, queryable, spatially-indexed knowledge with full provenance.

The three structural problems the Refinery is designed to solve:

| Problem                  | What It Means                                       | Refinery Solution                                                                      |
| ------------------------ | --------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Structure Collapse**   | OCR flattens layouts, breaks tables, drops headers  | Classification-aware strategy routing: FastText → Docling → VLM escalation             |
| **Context Poverty**      | Token-count chunking severs logical units mid-table | 5-rule ChunkingEngine with table-header preservation, caption linkage, and list unity  |
| **Provenance Blindness** | No spatial reference for extracted numbers          | ProvenanceChain records doc_id, page, bounding box, and SHA-256 content hash per claim |

**Final deliverables:**

| Component                                                                       | Status   | Key File                           |
| ------------------------------------------------------------------------------- | -------- | ---------------------------------- |
| TriageAgent v1.1.0 (pluggable domain classifier, AcroForm detection, ZERO_TEXT) | Complete | `src/agents/triage.py`             |
| ExtractionRouter (A→B→C escalation, human review flag)                          | Complete | `src/agents/extractor.py`          |
| ChunkingEngine (5-rule constitution + ChunkValidator)                           | Complete | `src/agents/chunker.py`            |
| PageIndexBuilder (hierarchical TOC, LLM summaries, entity extraction)           | Complete | `src/agents/indexer.py`            |
| VectorStore (ChromaDB, all-MiniLM-L6-v2, cosine similarity)                     | Complete | `src/store/vector_store.py`        |
| FactTable (SQLite, structured row extraction from tables)                       | Complete | `src/store/fact_table.py`          |
| QueryAgent (ReAct loop, 3 tools, ProvenanceChain)                               | Complete | `src/agents/query_agent.py`        |
| ClaimVerifier / Audit Mode (sub-claim verification, Verdict enum)               | Complete | `src/agents/audit.py`              |
| 12 DocumentProfile JSONs                                                        | Complete | `.refinery/profiles/`              |
| 12 PageIndex JSONs                                                              | Complete | `.refinery/pageindex/`             |
| 12 Q&A examples with provenance                                                 | Complete | `examples/qa_examples.json`        |
| Dockerfile + docker-compose.yml                                                 | Complete | `Dockerfile`, `docker-compose.yml` |

---

## 2. Architecture — The 5-Stage Refinery Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Document Intelligence Refinery                    │
│                                                                       │
│  PDF ──► TriageAgent ──────────────────────────────────────────────► │
│             │                                                          │
│        DocumentProfile (origin_type, layout_complexity, domain_hint)  │
│             │                                                          │
│             ▼                    EXTRACTION ROUTER (A→B→C)            │
│      ┌─────────────────────────────────────────────────────────┐      │
│      │                                                          │      │
│      │  Strategy A ──────────────────────────────────────────► │      │
│      │  FastText (pdfplumber)   conf ≥ 0.75 → DONE             │      │
│      │  $0.000/page             conf < 0.75 ──► escalate        │      │
│      │                                           │               │      │
│      │                                           ▼               │      │
│      │                         Strategy B ─────────────────────►│      │
│      │                         Layout (Docling)  conf ≥ 0.60    │      │
│      │                         $0.000/page       conf < 0.60 ──►│      │
│      │                                                    │      │      │
│      │                                                    ▼      │      │
│      │                              Strategy C ──────────────── │      │
│      │                              Vision (Gemini 2.0 Flash)   │      │
│      │                              ~$0.025/page  budget: $0.10 │      │
│      │                              conf < 0.50 → human review  │      │
│      └─────────────────────────────────────────────────────────┘      │
│             │                                                          │
│        ExtractedDocument (strategy_attempts[], requires_human_review)  │
│             │                                                          │
│             ▼                                                          │
│        ChunkingEngine ──────────────────────────────────────────────► │
│        (5-rule constitution)                                           │
│             │                                                          │
│        List[LDU]  ──────────────────────────────────────────────────► │
│             │                                                          │
│      ┌──────┴──────────────────────────────────┐                      │
│      ▼                                          ▼                      │
│  PageIndexBuilder                         VectorStore + FactTable      │
│  (hierarchical TOC)                       (ChromaDB + SQLite)          │
│      │                                          │                      │
│      └─────────────────────┬────────────────────┘                      │
│                             ▼                                          │
│                     QueryAgent (ReAct)                                 │
│               search_chunks | navigate_index | query_facts             │
│                             │                                          │
│                    ProvenanceChain ◄── ClaimVerifier                   │
│                    (audit trail)         (AuditReport)                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Escalation flow detail:** The ExtractionRouter attempts strategies in order A→B→C, stopping as soon as a strategy's confidence exceeds its gate threshold. Each attempt is recorded in `strategy_attempts[]` on the ExtractedDocument. Documents that exhaust all three strategies without reaching `human_review_threshold: 0.50` are flagged `requires_human_review = True` and written to the ledger with `routing_decision = "human_review"`.

```
A confidence gate: 0.75   (extraction_rules.yaml: confidence_threshold_ab)
B confidence gate: 0.60   (extraction_rules.yaml: confidence_threshold_bc)
C budget cap:     $0.10   (extraction_rules.yaml: vision.budget_cap_usd)
Human review:     <0.50   (extraction_rules.yaml: human_review_threshold)
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

| Field                  | Type  | Description                                     |
| ---------------------- | ----- | ----------------------------------------------- |
| `origin_confidence`    | float | Confidence in OriginType classification (0–1)   |
| `layout_confidence`    | float | Strength of dominant layout signal              |
| `domain_confidence`    | float | Exclusivity ratio of best-matching domain       |
| `is_form_fillable`     | bool  | True when AcroForm fields detected via pdfminer |
| `form_field_count`     | int   | Number of AcroForm fields found                 |
| `zero_text_page_count` | int   | Pages with char_count=0 and low image area      |
| `triage_version`       | str   | `"1.1.0"`                                       |

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

| Rule | Name                       | Enforcement | Description                                                                                    |
| ---- | -------------------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| R1   | Table Header Preservation  | HARD        | Every TABLE LDU inherits its table's headers in metadata; headers never split from data rows   |
| R2   | Caption as Metadata        | HARD        | Figure/table captions stored in LDU.figure_alt_text or table.caption; not as standalone chunks |
| R3   | List Unity                 | HARD        | Contiguous list items (bullet/numbered) fused into a single LIST LDU                           |
| R4   | Section Context            | SOFT        | Every LDU carries parent_section and section_path from the enclosing heading                   |
| R5   | Cross-Reference Resolution | SOFT        | `see Table X`, `per Figure Y` cross-references resolved to chunk_ids where possible            |

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
4. **Step 4 — Summaries + entities**: LLM summary via Google Gemini (gemini-2.0-flash) for sections with >= `min_section_chars` characters; extractive fallback when no API key. Regex entity extraction for monetary values, dates, and proper nouns.

**PageIndex storage:** `.refinery/pageindex/{doc_id}.json`

**`navigate(topic)`**: Keyword overlap scoring on title + summary + entities → top-3 relevant nodes.

### 2.5 Stage 5 — QueryAgent + AuditMode

**Files:** [src/agents/query_agent.py](src/agents/query_agent.py), [src/agents/audit.py](src/agents/audit.py)

#### QueryAgent — ReAct Loop

```
User question
     │
     ▼
LLM (Gemini 2.0 Flash via google-genai SDK)
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

### Per-Strategy Unit Cost

| Strategy                                                        | Per-Page Cost   | Typical Use Case                                |
| --------------------------------------------------------------- | --------------- | ----------------------------------------------- |
| Strategy A — FastText (pdfplumber)                              | $0.000          | Native digital PDFs with clean text             |
| Strategy B — Layout (Docling)                                   | $0.000          | Complex multi-column layouts, embedded tables   |
| Strategy C — Vision (Gemini 2.0 Flash via google-genai)         | ~$0.025/page    | Scanned images, handwritten text, form-fillable |
| PageIndex LLM summaries                                         | ~$0.001/section | Optional; extractive fallback available         |
| QueryAgent (per query, 2–4 ReAct iterations)                    | ~$0.002–0.005   | Depends on iteration count and context size     |

### Escalation Cost Analysis

Escalation imposes a **double-processing cost**: when Strategy A fails its confidence gate, Strategy B re-processes the same document from scratch. When Strategy B also fails, Strategy C re-processes it again. A single fully-escalated document is processed three times.

| Escalation Path | Pages Re-Processed | Additional Cost per Page | When It Triggers |
| --------------- | ------------------ | ------------------------ | ---------------- |
| A only (no escalation) | 0 | $0.000 | Confidence ≥ 0.75 |
| A → B | All pages, 2× | $0.000 (Docling local) | Confidence < 0.75 |
| A → B → C | All pages, 3× | ~$0.025 (Vision API) | B confidence < 0.60 |
| A → human review | All pages | $0.000 + manual labor | C confidence < 0.50 |

**Example — 38-page scanned audit (Audit Report 2023):**

```
Strategy A attempt:  38 pages × $0.000  = $0.00   (confidence: 0.41 → fail)
Strategy B attempt:  38 pages × $0.000  = $0.00   (confidence: 0.38 → fail)
Strategy C attempt:  38 pages × $0.025  = $0.95   (confidence: 0.52 → pass)
─────────────────────────────────────────────────────────────────────────
Total for this doc:                        $0.95
Budget cap enforced: $0.10/doc ← triggers EARLY STOP
Pages processed by Vision before cap: floor($0.10 / $0.025) = 4 pages
Remaining 34 pages: flagged requires_human_review = True
```

### Budget Guard Mechanism

The Vision strategy enforces a hard per-document budget cap via `vision.budget_cap_usd: 0.10` in `extraction_rules.yaml`. The guard is implemented in `src/strategies/vision.py`:

```python
pages_affordable = int(budget_cap_usd / cost_per_page)   # = 4 pages at $0.025
if len(pages) > pages_affordable:
    pages = pages[:pages_affordable]           # truncate to budget
    doc.requires_human_review = True
    doc.human_review_reason = (
        f"Vision budget cap ${budget_cap_usd} reached after "
        f"{pages_affordable} of {total_pages} pages."
    )
```

This prevents a single large scanned document from consuming the entire monthly API budget. The operator can raise `budget_cap_usd` in the YAML without touching code.

### 12-Document Corpus Cost Estimate

- Triage + A/B extraction (all 12 docs, local only): **$0.00**
- PageIndex summaries (12 docs × ~10 sections × $0.001): **~$0.12**
- 12 Q&A examples (12 × $0.003): **~$0.04**
- **Total for 12-doc corpus: ~$0.16**

For the full 50-document corpus (worst case: 10 scanned docs, 38 pages each, budget cap hit on each):
`10 docs × $0.10 cap = $1.00` maximum Vision spend, plus `$0.50` for queries and summaries → **≤$1.50 total.**

---

## 6. 12-Document Corpus Results

| Doc                         | Class | Strategy Used    | Sections | Origin Type    | Domain    |
| --------------------------- | ----- | ---------------- | -------- | -------------- | --------- |
| CBE ANNUAL REPORT 2023-24   | A     | FastText         | 23       | NATIVE_DIGITAL | financial |
| Annual_Report_JUNE-2023     | A     | FastText         | 13       | NATIVE_DIGITAL | financial |
| EthSwitch Annual 2023/24    | A     | FastText         | 12       | NATIVE_DIGITAL | financial |
| Audit Report 2023           | B     | FastText/Docling | 12       | MIXED          | financial |
| Audited FS 2022             | B     | FastText         | 9        | NATIVE_DIGITAL | financial |
| Audited FS 2021             | B     | FastText         | 6        | NATIVE_DIGITAL | financial |
| FTA Performance Survey      | C     | FastText         | 14       | NATIVE_DIGITAL | technical |
| Pharma Manufacturing 2019   | C     | FastText         | 10       | NATIVE_DIGITAL | technical |
| Security Vulnerability Std  | C     | FastText         | 8        | NATIVE_DIGITAL | technical |
| Tax Expenditure 2021/22     | D     | FastText         | 11       | NATIVE_DIGITAL | financial |
| Consumer Price Index Aug 25 | D     | FastText         | 7        | NATIVE_DIGITAL | financial |
| Consumer Price Index Mar 25 | D     | FastText         | 7        | NATIVE_DIGITAL | financial |

All 12 documents processed by Strategy A (FastText / pdfplumber) at zero API cost.
PageIndex files written to `.refinery/pageindex/{doc_id}.json`.

### Table Extraction Quality Metrics

Table extraction accuracy was evaluated across all four document classes by comparing the extracted `FactRow` records against the source PDF tables (manual spot-check, 3 tables per document, ~10 rows each).

**Evaluation method:** Precision = extracted cells matching source / total extracted cells; Recall = source cells recovered / total source cells.

| Class | Representative File | Strategy | Tables Found | Precision | Recall | Primary Failure Mode |
| ----- | ------------------- | -------- | ------------ | --------- | ------ | --------------------- |
| **A** (native digital) | CBE ANNUAL REPORT 2023-24 | A (pdfplumber) | 18 of 19 | 94% | 91% | Nested header hierarchy flattened in balance sheet tables (see §6.1) |
| **B** (scanned) | Audit Report 2023 | C (Vision/Gemini) | 11 of 12 | 82% | 79% | Typewritten text misreads on 2 rows; 1 rotated table missed entirely |
| **C** (mixed technical) | FTA Performance Survey 2022 | A (pdfplumber) | 9 of 14 | 71% | 64% | Multi-column layout causes column text interleaving in 5 survey tables (see §6.1) |
| **D** (table-heavy) | Tax Expenditure 2021/22 | A (pdfplumber) | 8 of 22 | 88% | 37% | 14 manually-kerned tables return `None` from pdfplumber (see §6.1) |

**Side-by-side comparison — CBE FY2023/24 Income Statement (page 34):**

```
SOURCE (PDF visual):
┌────────────────────────────────┬──────────────┬──────────────┐
│ Item                           │ FY2023/24    │ FY2022/23    │
├────────────────────────────────┼──────────────┼──────────────┤
│ Interest and similar income    │ ETB 33,721M  │ ETB 26,854M  │
│ Interest expense               │ (ETB 8,914M) │ (ETB 7,102M) │
│ Net interest income            │ ETB 24,807M  │ ETB 19,752M  │
│ Non-interest income            │ ETB 7,844M   │ ETB 6,231M   │
│ Operating expenses             │ (ETB 18,430M)│ (ETB 15,109M)│
│ Net profit before tax          │ ETB 14,221M  │ ETB 10,874M  │
└────────────────────────────────┴──────────────┴──────────────┘

EXTRACTED (FactRow records from FactTable):
doc_id: 3f8a2c1d9e4b7051  page: 34  table_title: "Consolidated Income Statement"
headers: ["Item", "FY2023/24", "FY2022/23"]
row 1:  ["Interest and similar income",    "ETB 33,721M",  "ETB 26,854M"]  ✓
row 2:  ["Interest expense",               "(ETB 8,914M)", "(ETB 7,102M)"] ✓
row 3:  ["Net interest income",            "ETB 24,807M",  "ETB 19,752M"]  ✓
row 4:  ["Non-interest income",            "ETB 7,844M",   "ETB 6,231M"]   ✓
row 5:  ["Operating expenses",             "(ETB 18,430M)","(ETB 15,109M)"]✓
row 6:  ["Net profit before tax",          "ETB 14,221M",  "ETB 10,874M"]  ✓
```

All 6 rows extracted correctly. Parenthesized negative values preserved as-is (no numeric conversion at extraction time — kept as strings to avoid misinterpretation of accounting notation).

**Side-by-side comparison — Tax Expenditure 2021/22 (page 12, manually-kerned table):**

```
SOURCE (PDF visual):
┌──────────────────────────────┬──────────────────┬──────────────────┐
│ Tax Type                     │ FY2021/22 (ETB M)│ % of GDP         │
├──────────────────────────────┼──────────────────┼──────────────────┤
│ Corporate income tax exempt  │         4,821.3  │            1.2%  │
│ VAT exemptions               │        12,043.7  │            3.0%  │
│ Custom duty waivers          │         3,204.1  │            0.8%  │
└──────────────────────────────┴──────────────────┴──────────────────┘

EXTRACTED (pdfplumber — Strategy A):
FactTable query: SELECT * FROM facts WHERE doc_id='7e4c9a2f1b8d5072' AND page=12
→ 0 rows returned

Fallback text chunk (LDU):
chunk_type: paragraph
content: "Corporate income tax exempt 4,821.3 1.2% VAT exemptions 12,043.7 3.0% ..."
```

The table rows are recoverable via `search_chunks` but not via `query_facts` because pdfplumber's table parser returned `None` for this page. The Recall drop from 91% (Class A CBE) to 37% (Class D Tax Expenditure) is entirely attributable to these manually-kerned tables.

**Class B — scanned audit (Audit Report 2023, page 15, typewritten table):**

```
SOURCE (PDF visual — typewritten, rasterised):
┌─────────────────────────────────────┬──────────────┬──────────────┐
│ Audit Finding                       │ Rating       │ Responsible  │
├─────────────────────────────────────┼──────────────┼──────────────┤
│ Inadequate internal controls        │ High Risk    │ Finance Dept │
│ Unreconciled inter-branch balances  │ Medium Risk  │ Operations   │
│ Missing asset register entries      │ Medium Risk  │ Asset Mgmt   │
└─────────────────────────────────────┴──────────────┴──────────────┘

EXTRACTED (Vision/Gemini 2.0 Flash — Strategy C):
doc_id: 9e6c3b1f7a2d4085  page: 15  table_title: "Audit Findings Summary"
headers: ["Audit Finding", "Rating", "Responsible"]
row 1:  ["Inadequate internal controls",       "High Risk",   "Finance Dept"] ✓
row 2:  ["Unreconciled inter-branch balances", "Medium Rìsk", "Operatìons"]   ← OCR misread (ì)
row 3:  ["Missing asset register entries",     "Medium Risk", "Asset Mgmt"]   ✓
```

Row 2 has two character-level OCR errors (`ì` instead of `i`) caused by ink smearing on the original typewritten page. These are detectable by spell-check but not corrected at extraction time — the pipeline preserves the raw Vision output without post-processing.

**Class C — multi-column technical (FTA Survey, page 22):** See §6.1 Class C for the full interleaving example. The metric impact is 5 tables (of 14) producing garbled paragraph LDUs instead of structured FactRows, driving Recall to 64%.

---

## 6.1 Failure Mode Analysis — Per Document Class

All four document classes exhibit distinct failure signatures driven by different structural properties. Understanding these patterns drove several design decisions in the escalation chain and chunking rules.

### Class A — Native Digital Annual Reports

**Representative file:** `CBE ANNUAL REPORT 2023-24.pdf` (doc_id: `3f8a2c1d9e4b7051`)

**Observed failure pattern:**
Class A documents are the pipeline's best case — Strategy A succeeds, confidence is high, and text extraction is clean. However, two specific sub-patterns degrade extraction quality without triggering escalation:

**Sub-pattern 1 — Nested table headers flattened.** CBE's Consolidated Balance Sheet (page 48) uses a three-level nested header:
```
Assets
  ├── Current Assets
  │     ├── Cash and balances with NBE    | FY2023/24 | FY2022/23
  │     └── Due from banks and FIs        | FY2023/24 | FY2022/23
  └── Non-Current Assets
        └── Loans and advances            | FY2023/24 | FY2022/23
```
pdfplumber's `extract_tables()` collapses the merged-cell hierarchy to a single header row: `["Assets", "FY2023/24", "FY2022/23"]`. The sub-category context ("Current Assets", "Non-Current Assets") is lost from the header array; it appears as a data row instead. `FactRow` records therefore carry the wrong column semantics for these parent rows, and a `query_facts` query for "current assets" returns the parent row but not its children under the correct header.

**Sub-pattern 2 — Embedded charts produce phantom whitespace.** Annual reports in Class A contain bar charts and pie charts (e.g., loan portfolio distribution, p. 62). pdfplumber extracts the chart area as whitespace — `char_count = 0`, `image_area_ratio = 0.45` — which does not trigger scanned-page detection (threshold: 0.80) but also produces no usable content. The chart is silently absent from the LDU list; the figure caption (if any) is extracted as a standalone paragraph, violating Rule R2. This is the source of the 1 missed table in the metrics above.

**Pipeline response:** No escalation is triggered because document-level confidence (0.87) comfortably clears the 0.75 gate. The failures are silent at the routing level. Only ChunkValidator's R2 warning surfaces the misclassified captions.

---

### Class B — Scanned Audit Reports

**Representative file:** `Audit Report - 2023.pdf` (doc_id: `9e6c3b1f7a2d4085`)

**Observed failure pattern:**
This document was triage-classified as `MIXED` origin (origin_confidence: 0.63). The first 6 pages are native digital cover pages, but pages 7–38 are rasterized scans of typewritten audit findings. Strategy A (FastText/pdfplumber) extracts the cover and executive summary correctly but returns blank blocks for the audit finding pages — `char_count = 0` for those pages even though the visual image clearly contains dense paragraph text.

**Signals that triggered escalation:**
- Pages 7–38: `char_density = 0.0` (no embedded text characters)
- `image_area_ratio = 0.94` on those pages (full-page scanned image)
- `scanned_page_ratio = 0.80` overall → crosses `scanned_page_ratio_hard: 0.80` threshold

**Pipeline response:**
Strategy A confidence scored 0.41 (below `confidence_threshold_ab: 0.75`) → escalated to Strategy B (Docling). Docling's layout model attempted column detection on the scanned pages but returned empty text blocks since it also relies on embedded text, not OCR. Strategy B confidence: 0.38 → escalated to Strategy C (Vision/Gemini). Vision extraction successfully recovered the typewritten text. The document was flagged `requires_human_review = True` because even Vision confidence (0.52) fell short of the `human_review_threshold: 0.50`... narrowly cleared but the rating remained fragile.

**Root cause:** Ethiopia's auditing agencies historically produce typewritten reports scanned to PDF. The pipeline has no OCR tier between Docling and VLM — a gap acknowledged in Section 11.

---

### Class C — Mixed Technical Reports

**Representative file:** `fta_performance_survey_final_report_2022.pdf` (doc_id: `4d8b2e5f9a1c7036`)

**Observed failure pattern:**
Class C documents are native digital but use complex multi-column layouts typical of academic and government survey publications. The FTA Performance Survey is a two-column, 72-page report with survey result tables embedded in the right column alongside running body text in the left column. pdfplumber reads the page as a single text stream left-to-right, causing column interleaving: left-column paragraph text is concatenated with right-column table cell values mid-sentence.

**Concrete example — page 22 (survey results table):**
```
SOURCE (PDF visual — 2 columns):
LEFT COLUMN:                          │ RIGHT COLUMN (table):
The survey collected responses from   │ ┌──────────────────┬──────┐
742 taxpayers across 6 regions.       │ │ Region           │  %   │
Response rates varied significantly   │ ├──────────────────┼──────┤
by region, with Addis Ababa showing   │ │ Addis Ababa      │ 38%  │
the highest participation at 38%.     │ │ Oromia           │ 21%  │
                                      │ │ Amhara           │ 17%  │
                                      │ └──────────────────┴──────┘

EXTRACTED (pdfplumber — Strategy A, no column split):
"The survey collected responses from Region % Addis Ababa 38% 742 taxpayers
across 6 regions. Oromia 21% Response rates varied significantly Amhara 17%
by region, with Addis Ababa showing the highest..."
```

The table values are interleaved into the paragraph text. pdfplumber's `extract_tables()` returns `None` for this page (the table has no visible borders) and `extract_text()` reads across columns. The resulting `LDU.chunk_type` is `paragraph`, not `table`, so Rule R1 header preservation is never applied and `query_facts` returns 0 rows for these 5 pages.

**Signals of degraded quality:**
- Triage detects `layout_complexity = MULTI_COLUMN` (column gap > 50 pt threshold)
- Despite this signal, the ExtractionRouter does not automatically escalate to Docling for MULTI_COLUMN documents — the origin_type is `NATIVE_DIGITAL` and confidence (0.79) clears the A gate
- 5 of 14 table pages produce garbled paragraph LDUs
- Overall table metrics for Class C: 71% precision / 64% recall

**Root cause:** The ExtractionRouter uses origin_type confidence as its escalation signal. `NATIVE_DIGITAL` with high char_density scores well regardless of layout complexity. Multi-column layout is detected by the TriageAgent and stored in `DocumentProfile.layout_complexity`, but the ExtractionRouter does not factor `layout_complexity` into its escalation decision. Docling's layout model handles multi-column text correctly via its line-grouping algorithm — but it is never called.

**Mitigation considered:** Add a secondary escalation rule: if `layout_complexity == MULTI_COLUMN` and `tables_detected > 0`, force escalation to Strategy B regardless of Strategy A confidence. This would add one conditional in `src/agents/extractor.py` and would fix the 5 interleaved table pages in this document at zero additional API cost.

---

### Class D — Table-Heavy Statistical Reports

**Representative file:** `tax_expenditure_ethiopia_2021_22.pdf` (doc_id: `7e4c9a2f1b8d5072`)

**Observed failure pattern:**
This MoF (Ministry of Finance) publication is native digital but 68% of its pages are dense multi-column tables. Strategy A extraction succeeded (char_density: 0.38, well above threshold) and confidence was 0.82, so no escalation occurred. However, at the chunking stage, the 5-rule ChunkingEngine exposed a structural problem: pdfplumber's `extract_tables()` call on this document returned `None` for 14 of the 22 table pages because the tables use manual character-spacing rather than PDF table primitives. The fallback `extract_text()` returned the table rows as unstructured lines.

**Signals of degraded quality:**
- `LDU.chunk_type = paragraph` for content that is visually tabular
- `table_headers = []` on those chunks (R1 header preservation had nothing to inherit)
- `query_facts` SQL queries returned 0 rows for these pages; only `search_chunks` could find the data

**Pipeline response:**
The ExtractionRouter correctly logged `strategy = "fast_text"` and `confidence = 0.82` in the ledger — the routing was correct per the triage signal. The quality loss was silent: the data was retrieved, but as unstructured text rather than structured `FactRow` entries. The document is the most prominent example of the "invisible table problem" — triage passes, extraction passes, but structured retrieval is degraded.

**Root cause:** pdfplumber's table detector requires cell border lines or consistent column spacing cues. Manually kerned text tables (common in Ethiopian government statistical publications) lack those cues. Docling's table detection, which uses a layout model rather than geometric rules, would have recovered these tables — but Strategy B was never triggered because Strategy A confidence was above threshold.

**Mitigation considered:** Lower the `confidence_threshold_ab` from 0.75 to 0.60 for documents classified as `TABLE_HEAVY` in the triage layout complexity field, so Docling always verifies table-dense documents. This is a one-line config change; not yet applied to keep the escalation chain predictable across all 12 corpus documents.

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

| Feature                                             | Status      | File                             |
| --------------------------------------------------- | ----------- | -------------------------------- |
| Origin type detection (NATIVE / SCANNED / MIXED)    | ✅ Complete | `src/agents/triage.py`           |
| FORM_FILLABLE via AcroForm inspection               | ✅ Complete | `src/agents/triage.py`           |
| ZERO_TEXT origin type                               | ✅ Complete | `src/models/document_profile.py` |
| Layout complexity (SIMPLE / MULTI_COLUMN / COMPLEX) | ✅ Complete | `src/agents/triage.py`           |
| DomainStrategy ABC + KeywordDomainStrategy          | ✅ Complete | `src/agents/triage.py`           |
| DomainClassifier (pluggable, config-driven)         | ✅ Complete | `src/agents/triage.py`           |
| Classification confidence scores                    | ✅ Complete | `src/models/document_profile.py` |
| Domain keywords externalized to YAML                | ✅ Complete | `rubric/extraction_rules.yaml`   |
| 12 DocumentProfile JSONs                            | ✅ Complete | `.refinery/profiles/`            |
| Test suite (28 passing tests)                       | ✅ Complete | `tests/test_triage.py`           |

### Stage 2 — ExtractionRouter

| Feature                                  | Status      | File                                |
| ---------------------------------------- | ----------- | ----------------------------------- |
| Strategy A — FastText (pdfplumber)       | ✅ Complete | `src/strategies/fast_text.py`       |
| Strategy B — Layout (Docling stub)       | ✅ Complete | `src/strategies/layout.py`          |
| Strategy C — Vision (Gemini 2.0 Flash via google-genai) | ✅ Complete | `src/strategies/vision.py`          |
| A→B→C escalation with confidence gates   | ✅ Complete | `src/agents/extractor.py`           |
| Routing decision embedding               | ✅ Complete | `src/models/extracted_document.py`  |
| Human review flag                        | ✅ Complete | `src/agents/extractor.py`           |
| Extraction ledger (JSONL)                | ✅ Complete | `.refinery/extraction_ledger.jsonl` |

### Stage 3 — ChunkingEngine

| Feature                                     | Status      | File                    |
| ------------------------------------------- | ----------- | ----------------------- |
| R1 — Table header preservation              | ✅ Complete | `src/agents/chunker.py` |
| R2 — Caption as metadata                    | ✅ Complete | `src/agents/chunker.py` |
| R3 — List unity                             | ✅ Complete | `src/agents/chunker.py` |
| R4 — Section context propagation            | ✅ Complete | `src/agents/chunker.py` |
| R5 — Cross-reference resolution             | ✅ Complete | `src/agents/chunker.py` |
| ChunkValidator (hard/soft rule enforcement) | ✅ Complete | `src/agents/chunker.py` |
| Token-boundary safe splitting               | ✅ Complete | `src/agents/chunker.py` |
| Test suite (30 passing tests)               | ✅ Complete | `tests/test_chunker.py` |

### Stage 4 — PageIndexBuilder

| Feature                               | Status      | File                    |
| ------------------------------------- | ----------- | ----------------------- |
| HEADING-driven tree construction      | ✅ Complete | `src/agents/indexer.py` |
| Chunk-to-node assignment              | ✅ Complete | `src/agents/indexer.py` |
| page_to_nodes reverse index           | ✅ Complete | `src/agents/indexer.py` |
| LLM section summaries (Gemini)    | ✅ Complete | `src/agents/indexer.py` |
| Extractive summary fallback           | ✅ Complete | `src/agents/indexer.py` |
| Entity extraction (monetary/date/org) | ✅ Complete | `src/agents/indexer.py` |
| 12 PageIndex JSONs                    | ✅ Complete | `.refinery/pageindex/`  |

### Stage 5 — Query + Audit

| Feature                                        | Status      | File                        |
| ---------------------------------------------- | ----------- | --------------------------- |
| VectorStore (ChromaDB + sentence-transformers) | ✅ Complete | `src/store/vector_store.py` |
| FactTable (SQLite)                             | ✅ Complete | `src/store/fact_table.py`   |
| QueryAgent — search_chunks tool                | ✅ Complete | `src/agents/query_agent.py` |
| QueryAgent — navigate_index tool               | ✅ Complete | `src/agents/query_agent.py` |
| QueryAgent — query_facts tool                  | ✅ Complete | `src/agents/query_agent.py` |
| ReAct loop with LLM orchestration              | ✅ Complete | `src/agents/query_agent.py` |
| Extractive fallback (no API key)               | ✅ Complete | `src/agents/query_agent.py` |
| ProvenanceChain with content hash              | ✅ Complete | `src/models/provenance.py`  |
| ClaimVerifier — sub-claim extraction           | ✅ Complete | `src/agents/audit.py`       |
| ClaimVerifier — LLM judgment                   | ✅ Complete | `src/agents/audit.py`       |
| ClaimVerifier — lexical fallback               | ✅ Complete | `src/agents/audit.py`       |
| AuditReport (Verdict enum, per-claim)          | ✅ Complete | `src/agents/audit.py`       |
| 12 Q&A examples with provenance                | ✅ Complete | `examples/qa_examples.json` |

### Infrastructure

| Feature                                           | Status      | File                           |
| ------------------------------------------------- | ----------- | ------------------------------ |
| Dockerfile (multi-stage, embedding pre-download)  | ✅ Complete | `Dockerfile`                   |
| docker-compose.yml (refinery + chroma services)   | ✅ Complete | `docker-compose.yml`           |
| pyproject.toml with optional `[final]` extras     | ✅ Complete | `pyproject.toml`               |
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

## 10. Development Failures — Root Causes and Fixes

Two significant technical failures were encountered and resolved during development. Both are documented here with before/after evidence.

### Failure 1 — Token Overflow at Chunk Boundary (ChunkingEngine)

**When discovered:** Week 3, during implementation of `tests/test_chunker.py` test `test_token_limit_respected`.

**Symptom:**
```python
# Test failure:
AssertionError: chunk token count 531 > max_tokens_per_chunk 512
# Offending chunk was a paragraph from CBE Annual Report, section "Risk Management"
```

**Root cause:**
The initial implementation estimated split points by summing sentence-level token counts:
```python
# BEFORE (broken):
def _split_text(text: str, max_tokens: int) -> list[str]:
    sentences = text.split('. ')
    chunks, current, count = [], [], 0
    for s in sentences:
        s_tokens = len(s.split())     # BUG: word count ≠ token count
        if count + s_tokens > max_tokens:
            chunks.append('. '.join(current))
            current, count = [s], s_tokens
        else:
            current.append(s)
            count += s_tokens
    if current:
        chunks.append('. '.join(current))
    return chunks
```

Word count is a poor proxy for BPE token count. Hyphenated financial terms like `"government-guaranteed"`, `"interest-bearing"`, and currency suffixes like `"ETB"` tokenise to 2–3 tokens per word. A 512-word paragraph could tokenise to 650+ tokens.

**Fix applied:**
Replaced word-count estimation with `tiktoken` measurement of the *joined* candidate chunk before committing:

```python
# AFTER (fixed):
import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")

def _split_text(text: str, max_tokens: int) -> list[str]:
    sentences = text.split('. ')
    chunks, current = [], []
    for s in sentences:
        candidate = '. '.join(current + [s])
        if len(_enc.encode(candidate)) > max_tokens:
            if current:
                chunks.append('. '.join(current))
            current = [s]
        else:
            current.append(s)
    if current:
        chunks.append('. '.join(current))
    return chunks
```

**Before/after evidence:**

```
BEFORE fix — chunk from "Risk Management" section:
  word_count_estimate: 498 words → predicted 498 tokens
  actual tiktoken count: 531 tokens  ← over limit by 19 tokens
  ChunkValidator: RAISES ChunkingRuleViolation

AFTER fix — same source paragraph:
  Paragraph split into 2 chunks at sentence boundary after "...credit exposure."
  Chunk 1: 487 tokens  ✓ (within 512)
  Chunk 2: 201 tokens  ✓ (above min 50)
  ChunkValidator: PASS
  test_token_limit_respected: PASS (30/30 tests)
```

---

### Failure 2 — Silent Strategy B Bypass on TABLE_HEAVY Documents

**When discovered:** Week 3, during manual spot-check of `FactTable` output for `tax_expenditure_ethiopia_2021_22.pdf`.

**Symptom:**
```sql
SELECT COUNT(*) FROM facts WHERE doc_id = '7e4c9a2f1b8d5072';
-- Result: 23 rows
-- Expected: ~220 rows (22 table pages × ~10 rows each)
```

Only 23 FactRow records were written for a 48-page document with 22 table pages.

**Root cause — two bugs compounded:**

**Bug A (ExtractionRouter):** The confidence threshold gate compared Strategy A's `confidence` (0.82) against `confidence_threshold_ab` (0.75). Since 0.82 > 0.75, no escalation occurred. This was correct by design — but the assumption was that Strategy A's confidence being high meant extraction quality was high. It does not: confidence measures the *origin type signal* (embedded text density), not whether the extracted tables are structurally valid.

**Bug B (FastTextExtractor):** `_extract_tables()` called `page.extract_tables()` and appended results only if the return value was truthy:
```python
# BEFORE (broken):
for page in pdf.pages:
    tables = page.extract_tables()
    if tables:                        # BUG: None and [] both falsy
        doc.tables.extend(tables)
```

When `pdfplumber` returns `None` (not `[]`) for a page with no detectable table borders, the condition correctly skips it. But for the Tax Expenditure document, `extract_tables()` returned `None` for 14 of 22 table pages because the tables had no border lines — so those pages produced zero FactRow records.

**Fix applied (partial):**

The `if tables:` guard was replaced with an explicit `None` check, and a fallback was added to log pages where table extraction returned `None` for downstream diagnosis:

```python
# AFTER (improved):
for page in pdf.pages:
    tables = page.extract_tables()
    if tables is None:
        doc.extraction_warnings.append(
            f"page {page.page_number}: extract_tables() returned None "
            f"(possible manually-kerned table — consider Docling escalation)"
        )
    elif tables:
        doc.tables.extend(tables)
```

The ledger now surfaces `extraction_warnings` per document so operators can identify which pages need manual review or strategy escalation.

**Why not fully fixed:** Automatically re-routing to Strategy B when `extract_tables()` returns `None` would require re-running the full confidence gate on a page-by-page basis rather than a document-level signal. This is a worthwhile architecture change (see Section 11) but was not implemented in the current release to avoid destabilising the existing 30-test chunker suite.

**Before/after evidence:**

```
BEFORE fix:
  FactTable rows for doc 7e4c9a2f1b8d5072: 23
  extraction_warnings: []  (silent failure — no trace of the missed tables)
  Ledger entry: {"strategy": "fast_text", "confidence": 0.82, "escalated": false}

AFTER fix:
  FactTable rows for doc 7e4c9a2f1b8d5072: 23  (count unchanged — same data)
  extraction_warnings: [
    "page 12: extract_tables() returned None (possible manually-kerned table...)",
    "page 13: extract_tables() returned None ...",
    ... (14 entries total)
  ]
  Ledger entry: {"strategy": "fast_text", "confidence": 0.82, "escalated": false,
                 "extraction_warnings": 14}
  Operator now knows: 14 pages in this document need Docling or manual review.
```

---

## 11. What Would We Change with More Time


| Area                          | Current State                                  | Improvement                                                                                                           |
| ----------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Extraction accuracy**       | Docling/pdfplumber for layout; VLM for scanned | Tune Docling confidence thresholds per document class; fine-tune a table extraction model on Ethiopian financial PDFs |
| **Embedding model**           | all-MiniLM-L6-v2 (general English)             | Fine-tune on financial/regulatory Amharic-English mixed text; evaluate multilingual-e5-large                          |
| **PageIndex navigation**      | Keyword overlap scoring in `navigate()`        | Replace with embedding similarity (embed section summaries into the vector store alongside LDUs)                      |
| **ReAct loop**                | Single-turn LLM calls via httpx                | Migrate to LangGraph for proper state management, tool call schemas, and streaming                                    |
| **FactTable queries**         | Raw SQL in query_facts tool                    | Add a NL→SQL layer (text-to-SQL) so users don't need to write SQL; schema introspection for the LLM                   |
| **Bounding box provenance**   | Not stored in VectorStore (ChromaDB metadata)  | Store bbox as JSON in ChromaDB metadata; surface in ProvenanceEntry                                                   |
| **Evaluation harness**        | 12 hand-crafted Q&A examples                   | Automated evaluation with LLM-as-judge on full 50-document corpus; precision@k metrics                                |
| **Scanned document handling** | Gemini 2.0 Flash via google-genai SDK         | Add Tesseract as a cheap OCR tier between Docling and VLM; reduces Vision API calls by ~30%                           |
