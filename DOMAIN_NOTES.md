# Domain Notes — Document Intelligence Refinery

> Phase 0 deliverable. Documents the extraction strategy decision tree, failure modes observed
> across the corpus, and the pipeline architecture diagram.

---

## 1. Extraction Strategy Decision Tree

```
                        ┌──────────────────────┐
                        │   Incoming Document   │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Triage Agent             │
                    │  (character density + bbox)  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    origin_type=               origin_type=          origin_type=
    native_digital             mixed                 scanned_image
              │                    │                     │
              ▼                    ▼                     ▼
   layout_complexity?      ──► Strategy B          Strategy C
              │             (Docling layout)      (VLM Vision)
   ┌──────────┴──────────┐
   │                     │
single_column      multi_column /
   │               table_heavy /
   ▼               figure_heavy
Strategy A              │
(pdfplumber)            ▼
   │              Strategy B
   │              (Docling)
   ▼
confidence_score?
   │
   ├── HIGH (≥0.75) ──► pass to Chunking Engine
   │
   └── LOW  (<0.75) ──► escalate to Strategy B
                              │
                        confidence_score?
                              │
                        ├── HIGH ──► pass
                        └── LOW  ──► escalate to Strategy C (VLM)
```

### Confidence Signal Formula (Strategy A)

```
confidence = (
    0.40 * char_density_score      +  # chars / page_area (normalized 0-1)
    0.30 * (1 - image_area_ratio)  +  # penalize image-dominated pages
    0.20 * font_metadata_score     +  # presence of embedded font data
    0.10 * whitespace_ratio_score     # reasonable whitespace = structured text
)
```

**Thresholds** (defined in `rubric/extraction_rules.yaml`):
- `char_density_min`: 0.05 chars/pt² → below this, page is likely scanned
- `image_area_max`: 0.50 → if images > 50% of page area, escalate
- `confidence_threshold_ab`: 0.75 → below triggers A→B escalation
- `confidence_threshold_bc`: 0.60 → below triggers B→C escalation

---

## 2. Failure Modes Observed Across Document Classes

### Class A — Native Digital Annual Reports (CBE, EthSwitch)

| Failure Mode | Example | Root Cause | Fix |
|---|---|---|---|
| Multi-column merge | CBE 2023-24 p.12: two columns of text merged into one stream | pdfplumber reads left-to-right without column detection | Route to Strategy B (Docling) |
| Table header loss | Income statement: header row lost when table spans page break | pdfplumber `extract_tables()` misses continued tables | Use Docling's table continuation detection |
| Footnote interleaving | Footnotes inserted mid-paragraph in text stream | Bounding box overlap between body and footnote regions | Filter by y-coordinate threshold |

### Class B — Scanned Audit Reports (DBE, Government Statements)

| Failure Mode | Example | Root Cause | Fix |
|---|---|---|---|
| Zero character stream | Audit Report 2023 p.1-40: `len(chars) == 0` on all pages | Pure image PDF — no embedded text | Detect via char_density < 0.001, route to VLM |
| OCR number confusion | Financial figures: "8" misread as "B", "0" as "O" | Poor scan quality + standard OCR | Prompt VLM with domain hint (financial) for digit-aware extraction |
| Rotated pages | Some pages at 90°/270° | Scanning orientation inconsistency | Detect via aspect ratio heuristic, apply rotation correction |

### Class C — Mixed Technical Reports (FTA, Pharmaceutical)

| Failure Mode | Example | Root Cause | Fix |
|---|---|---|---|
| Table-text interleave | FTA report: narrative paragraph split by embedded table | pdfplumber returns text blocks and table cells without ordering | Docling's reading order reconstruction |
| Hierarchical section loss | Section 3.2.1 numbering lost in extraction | Naive text extraction strips numbering context | Preserve section hierarchy as parent_section metadata |
| Figure caption orphaned | Chart caption stored 3 blocks away from figure bbox | Caption is spatially below figure but logically linked | Link by proximity: caption within 50pt below figure bbox |

### Class D — Table-Heavy Fiscal Data (Tax Expenditure, CPI)

| Failure Mode | Example | Root Cause | Fix |
|---|---|---|---|
| Multi-year table merge | Tax expenditure: 3-year columns collapsed to 1 | Column boundary detection fails on wide tables | Docling structured table JSON preserves column headers |
| Thousands separator confusion | "4,200" parsed as two numbers | Locale-specific number formatting | Post-process: validate numeric cells against regex `[\d,\.]+` |
| Merged cells | CPI table: category spans 3 rows | pdfplumber returns merged cells as single block | Docling table model handles cell spans explicitly |

---

## 3. Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Document Intelligence Refinery                     │
│                                                                      │
│  INPUT                                                               │
│  ─────                                                               │
│  PDFs (native + scanned) │ Excel/CSV │ Word/PPTX │ Images           │
│                                   │                                  │
│                                   ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 1: TRIAGE AGENT                                        │  │
│  │                                                                │  │
│  │  pdfplumber char density  →  origin_type                      │  │
│  │  bbox column analysis     →  layout_complexity                │  │
│  │  keyword classifier       →  domain_hint                      │  │
│  │                                                                │  │
│  │  OUTPUT: DocumentProfile → .refinery/profiles/{doc_id}.json   │  │
│  └───────────────────────────────┬────────────────────────────────┘  │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 2: STRUCTURE EXTRACTION LAYER                          │  │
│  │                                                                │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │   │ Strategy A   │  │ Strategy B   │  │  Strategy C      │   │  │
│  │   │ FastText     │  │ Layout       │  │  Vision          │   │  │
│  │   │ (pdfplumber) │  │ (Docling)    │  │  (Gemini Flash)  │   │  │
│  │   │ Cost: Low    │  │ Cost: Medium │  │  Cost: High      │   │  │
│  │   └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │  │
│  │          │                 │                   │              │  │
│  │          └─────────────────┴───────────────────┘              │  │
│  │                            │                                  │  │
│  │          ┌─────────────────▼──────────────────┐               │  │
│  │          │  Confidence Gate + Escalation Guard │               │  │
│  │          │  (reads extraction_rules.yaml)      │               │  │
│  │          └─────────────────┬──────────────────┘               │  │
│  │                            │                                  │  │
│  │  OUTPUT: ExtractedDocument + .refinery/extraction_ledger.jsonl │  │
│  └───────────────────────────────┬────────────────────────────────┘  │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 3: SEMANTIC CHUNKING ENGINE                            │  │
│  │                                                                │  │
│  │  5 Chunking Rules (Constitution):                             │  │
│  │  R1: Table cells never split from header                      │  │
│  │  R2: Figure caption stored as parent figure metadata          │  │
│  │  R3: Numbered lists kept as single LDU                        │  │
│  │  R4: Section headers stored as parent_section on children     │  │
│  │  R5: Cross-references resolved as chunk relationships         │  │
│  │                                                                │  │
│  │  ChunkValidator enforces all rules before emit                │  │
│  │  OUTPUT: List[LDU] with content_hash + bbox + page_refs      │  │
│  └───────────────────────────────┬────────────────────────────────┘  │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 4: PAGEINDEX BUILDER                                   │  │
│  │                                                                │  │
│  │  Section tree → LLM summaries (fast model) → navigation index │  │
│  │  OUTPUT: PageIndex tree → .refinery/pageindex/{doc_id}.json   │  │
│  └───────────────────────────────┬────────────────────────────────┘  │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 5: QUERY INTERFACE AGENT (LangGraph)                   │  │
│  │                                                                │  │
│  │  Tool 1: pageindex_navigate  → section-targeted retrieval     │  │
│  │  Tool 2: semantic_search     → vector store (ChromaDB)        │  │
│  │  Tool 3: structured_query    → SQLite FactTable               │  │
│  │                                                                │  │
│  │  Every answer: ProvenanceChain (doc + page + bbox + hash)     │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. VLM vs OCR Decision Boundary

| Signal | Threshold | Decision |
|--------|-----------|----------|
| char_density < 0.001 chars/pt² | Hard cutoff | → Vision (scanned) |
| image_area_ratio > 0.80 | Hard cutoff | → Vision |
| Strategy A confidence < 0.75 | Soft escalation | → Layout (Docling) |
| Strategy B confidence < 0.60 | Soft escalation | → Vision |
| handwriting_detected = True | Hard cutoff | → Vision |

**Cost tradeoff articulation (for client conversations):**
- Strategy A: ~$0.00 (local, CPU only, <1s/page)
- Strategy B: ~$0.00 (local Docling model, ~3-8s/page, GPU optional)
- Strategy C: ~$0.001-0.003/page (Gemini Flash via OpenRouter, budget capped at $0.10/doc)

At 400-page document scale: A = $0, B = $0, C = $0.40–$1.20. The escalation guard ensures C is only triggered for pages where A and B genuinely fail.

---

## 5. Key Insights from Tooling Research

### MinerU Architecture Insight
MinerU uses a cascade of specialized models — not one general model. PDF-Extract-Kit handles layout detection, then separate models handle formulas, tables, and reading order. The lesson: **specialization beats generalization** at document scale. We apply this same principle in our strategy router.

### Docling's DoclingDocument Representation
Docling's unified `DoclingDocument` object encodes structure (headings, sections), text blocks with bounding boxes, tables as proper row/column objects, and figures with captions — all in one traversable schema. Our `ExtractedDocument` Pydantic model mirrors this structure so all three strategies emit the same schema.

### PageIndex vs Naive Vector Search
On a 400-page financial report, embedding all chunks and doing cosine similarity retrieval surfaces chunks from the wrong section even when their text is similar. PageIndex solves this by first navigating to the correct section (O(log n) tree traversal), then doing vector search within that scope. This is the same insight as a database index — don't scan the full table when you can index into the right partition first.

### Chunr's Semantic Boundary Insight
Chunkr's key innovation: chunk boundaries must respect document semantic units (paragraph, table, caption), not token counts. A 512-token window that bisects a financial table produces hallucinated answers on every query about that table because the model sees half a header and half a data row. Our ChunkingEngine enforces this via 5 explicit rules.
