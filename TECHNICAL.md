# Document Intelligence Refinery — Technical Deep Dive

This document walks through every file in execution order, showing exactly what
each class does, how data flows between stages, and which lines to open when
explaining the project.

---

## Table of Contents

1. [The Data Contracts — Start Here](#1-the-data-contracts--start-here)
2. [Stage 1 — TriageAgent](#2-stage-1--triageagent)
3. [Stage 2 — ExtractionRouter](#3-stage-2--extractionrouter)
4. [Stage 3 — ChunkingEngine](#4-stage-3--chunkingengine)
5. [Stage 4 — PageIndexBuilder](#5-stage-4--pageindexbuilder)
6. [Stage 5 — QueryAgent](#6-stage-5--queryagent)
7. [Stage 5b — ClaimVerifier (Audit Mode)](#7-stage-5b--claimverifier-audit-mode)
8. [Data Stores](#8-data-stores)
9. [Configuration — rubric/extraction_rules.yaml](#9-configuration--rubricextraction_rulesyaml)
10. [End-to-End Trace: One PDF, One Query](#10-end-to-end-trace-one-pdf-one-query)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. The Data Contracts — Start Here

Before opening any agent file, open the **models** — they are the contracts
every stage must satisfy. Everything else is implementation detail.

### `src/models/document_profile.py` — Triage output

```
OriginType (enum)
  native_digital | scanned_image | mixed | form_fillable | zero_text

LayoutComplexity (enum)
  simple | multi_column | table_heavy | complex

DomainHint (enum)
  financial | legal | technical | medical | general

ExtractionCost (enum)
  free_tier | needs_layout_model | needs_vision_api | unknown

PageStats (BaseModel)          ← per-page measurements from pdfplumber
  page_num, char_count, char_density, image_area_ratio,
  table_count, estimated_columns, is_likely_scanned

DocumentProfile (BaseModel)    ← the output of Stage 1
  doc_id, filename, page_count
  origin_type, origin_confidence      ← What kind of PDF?
  layout_complexity, layout_confidence
  domain_hint, domain_confidence      ← What subject area?
  estimated_extraction_cost
  is_form_fillable, form_field_count
  zero_text_page_count
  avg_char_density, avg_image_area_ratio
  scanned_page_count, page_stats
  triage_version = "1.1.0"
```

**Key methods:**
- `DocumentProfile.save()` → writes to `.refinery/profiles/{doc_id}.json`
- `DocumentProfile.load(doc_id)` → reads it back

---

### `src/models/extracted_document.py` — Extraction output

```
TextBlock        text + BoundingBox + reading_order + font_size + is_heading
TableData        caption + headers[] + rows[][] + cells[] + BoundingBox
FigureBlock      figure_id + caption + alt_text + BoundingBox

ExtractionStrategy (enum)   fast_text | layout | vision

StrategyAttempt              one strategy's execution record
  strategy, confidence_score, escalated, escalation_reason,
  escalation_detail, cost_usd, processing_time_seconds

RoutingDecision              why the router chose the starting strategy
  initial_strategy, selection_reason, strategy_chain
  origin_type, layout_complexity, avg_char_density ...

ExtractedDocument            the output of Stage 2
  doc_id, filename, page_count
  text_blocks[], tables[], figures[]   ← content in reading order
  strategy_used, confidence_score
  routing_decision                     ← why this strategy?
  strategy_attempts[]                  ← full escalation trail
  requires_human_review, human_review_reason
  warnings[]
```

**Key property:**
- `ExtractedDocument.full_text` → all text blocks joined in reading order

---

### `src/models/ldu.py` — Chunking output

```
ChunkType (enum)
  paragraph | heading | table | figure | list | caption | code

LDURelationship              a resolved cross-reference
  source_chunk_id, target_chunk_id, relationship_type, context

LDU (BaseModel)              one Logical Document Unit
  chunk_id        "{doc_id}-chunk-{sequence:06d}"
  doc_id
  chunk_type
  content         canonical text of this chunk
  page_refs[]     all pages this chunk spans
  parent_section  immediate heading title (R4)
  section_path[]  full ancestor chain  ["Ch3", "3.2 Analysis"]
  content_hash    SHA-256 of content (computed on model_validate)
  heading_level   1-6 for HEADING chunks
  table_headers[] R1: preserved column headers
  figure_alt_text R2: caption or VLM description
  cross_refs[]    R5: resolved LDURelationship list
  token_count
  char_count
```

---

### `src/models/page_index.py` — PageIndex output

```
PageIndexNode
  node_id, title, level, page_start, page_end
  summary         2-3 sentence LLM summary
  key_entities[]  monetary values, dates, orgs
  data_types_present[]  "tables" | "figures" | "lists"
  chunk_ids[]     LDU chunk_ids belonging to this section
  parent_node_id, child_node_ids[]

PageIndex
  doc_id, filename, page_count
  nodes{}         dict[node_id → PageIndexNode]
  root_node_ids[] top-level sections
  page_to_nodes{} dict[page_number → [node_id, ...]]
```

**Key methods:**
- `PageIndex.navigate(topic)` → top-3 nodes by keyword score
- `PageIndex.save()` → `.refinery/pageindex/{doc_id}.json`
- `PageIndex.load(doc_id)` → reads it back

---

### `src/models/provenance.py` — Query output

```
ProvenanceEntry              one source citation
  doc_id, filename
  page_number                1-indexed
  bounding_box               Optional BoundingBox
  section_title
  chunk_id
  content_hash               SHA-256 — enables offline verification
  excerpt                    ≤200 chars
  retrieval_score            cosine similarity 0-1
  retrieval_method           "semantic_search" | "pageindex_navigate" | "structured_query"

ProvenanceChain              the complete answer + audit trail
  query, answer
  sources[]                  ordered by relevance
  is_verified
  unverifiable_claims[]
  timestamp, model_used, total_tokens_used
```

---

## 2. Stage 1 — TriageAgent

**File:** `src/agents/triage.py`
**Input:** PDF file path (string)
**Output:** `DocumentProfile` saved to `.refinery/profiles/{doc_id}.json`

### How to open and explain it

Open `src/agents/triage.py`. Jump to line 484 (`class TriageAgent`). The
`run()` method at line 500 is the entry point — it calls 6 sub-functions in
order:

```
TriageAgent.run(file_path)
  │
  ├─ _doc_id(file_path)                       line 210
  │    sha256(absolute_path)[:16] → deterministic, path-stable doc_id
  │
  ├─ _analyze_page(page, i, thresholds)        line 220  [once per page]
  │    pdfplumber extracts:
  │      char_count, char_density (chars/pt²)
  │      image_area_ratio (image pixels / page area)
  │      table_count (pdfplumber.find_tables())
  │      estimated_columns (text bbox clustering)
  │    returns PageStats
  │
  ├─ _detect_form_fillable(pdf, thresholds)    line 269
  │    pdf.doc.catalog["AcroForm"]["Fields"] → count interactive fields
  │    returns (is_form_fillable: bool, field_count: int, confidence: float)
  │
  ├─ _detect_origin_type(page_stats, thresholds)  line 307
  │    5-branch logic:
  │      zero_text_count ≥ 60% pages  → ZERO_TEXT (not scanned — truly blank)
  │      image_area_ratio ≥ 80%       → SCANNED_IMAGE
  │      20% ≤ ratio < 80%            → MIXED
  │      else                         → NATIVE_DIGITAL
  │    FORM_FILLABLE overlaid after if AcroForm fields found
  │    returns (OriginType, confidence: float)
  │
  ├─ _detect_layout_complexity(pdf, page_stats, thresholds)  line 381
  │    table_page_ratio ≥ 30%         → TABLE_HEAVY
  │    estimated_columns ≥ 2          → MULTI_COLUMN
  │    both above thresholds          → COMPLEX
  │    else                           → SIMPLE
  │    returns (LayoutComplexity, col_count: int, confidence: float)
  │
  ├─ _sample_text(pdf, page_stats)              line 442
  │    first 5000 chars from non-scanned pages
  │
  └─ DomainClassifier.classify(sample_text)    line 156
       runs all DomainStrategy.score() → picks highest
       confidence = best_score / total_score
       returns (DomainHint, confidence: float)
```

### The Pluggable Domain Classifier (lines 107–208)

This is worth explaining carefully — it demonstrates a real design pattern:

```python
# Abstract base class — the contract
class DomainStrategy(ABC):           # line 107
    @property
    @abstractmethod
    def domain(self) -> DomainHint: ...

    @abstractmethod
    def score(self, text: str) -> float: ...

# Default implementation — keyword counting
class KeywordDomainStrategy(DomainStrategy):  # line 126
    def score(self, text: str) -> float:
        return sum(text.lower().count(kw) for kw in self.keywords) * self.weight

# Orchestrator — calls all strategies, picks winner
class DomainClassifier:              # line 146
    def classify(self, text: str) -> tuple[DomainHint, float]:
        scores = {s.domain: s.score(text) for s in self.strategies}
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        confidence = scores[best] / total   # exclusivity ratio
        return best if scores[best] > 0 else DomainHint.GENERAL, confidence
```

**The key insight:** To add a new domain (e.g. `agricultural`), you add YAML:
```yaml
domains:
  agricultural:
    weight: 1.0
    keywords: [crop, harvest, irrigation, fertilizer, ...]
```
`_build_domain_classifier(config)` at line 175 reads it and builds
`KeywordDomainStrategy` instances at runtime. **Zero code changes.**

### Configuration loading (lines 89–106)

```python
def _load_thresholds(config: dict) -> dict:
    triage = config.get("triage", {})
    return {
        "scanned_ratio_high": triage.get("scanned_ratio_high", 0.80),
        "zero_text_page_ratio": triage.get("zero_text_page_ratio", 0.60),
        "form_fillable_min_fields": triage.get("form_fillable_min_fields", 1),
        ...
    }
```

Every numeric decision in TriageAgent comes from this dict — never hardcoded.

---

## 3. Stage 2 — ExtractionRouter

**File:** `src/agents/extractor.py`
**Input:** `DocumentProfile` + PDF file path
**Output:** `ExtractedDocument` + entry appended to `.refinery/extraction_ledger.jsonl`

### How to open and explain it

Jump to line 182 (`class ExtractionRouter`). The router:

```
ExtractionRouter.run(file_path, profile)
  │
  ├─ _build_routing_decision(profile)          line 67
  │    reads profile → picks initial strategy from strategy_chain
  │    "SCANNED_IMAGE → start at Vision"
  │    "NATIVE_DIGITAL + TABLE_HEAVY → start at Layout"
  │    "NATIVE_DIGITAL + SIMPLE → start at FastText"
  │
  └─ for strategy in strategy_chain:
       attempt = strategy.extract(file_path)
       record StrategyAttempt(strategy, confidence, escalated=False/True)
       │
       ├─ if confidence ≥ threshold → accept, break
       └─ if confidence < threshold → escalate (try next strategy)
                                      StrategyAttempt.escalated = True

  After loop:
    doc.routing_decision = routing_decision
    doc.strategy_attempts = all_attempts
    doc.requires_human_review = (final_confidence < human_review_threshold)
    _write_ledger_entry(doc)
```

### Ledger entry (line 115)

Every document produces one JSONL line in `.refinery/extraction_ledger.jsonl`:
```json
{
  "doc_id": "3f8a2c1d...",
  "filename": "CBE ANNUAL REPORT 2023-24.pdf",
  "strategy_used": "fast_text",
  "confidence_score": 0.87,
  "escalation_count": 0,
  "requires_human_review": false,
  "cost_usd": 0.0,
  "timestamp": "2026-03-06T09:00:00"
}
```

### The three strategies (`src/strategies/`)

| File | Class | What it does |
|------|-------|-------------|
| `fast_text.py` | `FastTextExtractor` | `pdfplumber.open()` → iterate pages → `page.extract_text()`, `page.extract_tables()`, `page.images` |
| `layout.py` | `LayoutExtractor` | Docling `DocumentConverter` → layout-aware bounding boxes for multi-column text |
| `vision.py` | `VisionExtractor` | Renders each page as PNG → sends to Gemini 2.0 Flash via google-genai → parses response |

All three inherit from `BaseExtractionStrategy` in `src/strategies/base.py`,
which enforces the common `extract(file_path) → ExtractedDocument` interface.

---

## 4. Stage 3 — ChunkingEngine

**File:** `src/agents/chunker.py`
**Input:** `ExtractedDocument`
**Output:** `list[LDU]`

### How to open and explain it

Jump to line 240 (`class ChunkingEngine`). The `run()` method does two passes:

```
ChunkingEngine.run(doc: ExtractedDocument) → list[LDU]
  │
  ├─ Pass 1: Build LDUs
  │    merge all content into one reading-order stream:
  │      text_blocks + tables + figures
  │    sorted by reading_order field
  │
  │    for each item:
  │      if TextBlock and is_heading:
  │          → ChunkType.HEADING, update current_section / section_path
  │      if TextBlock and _is_list_item(text):  ← line 121
  │          → accumulate into list buffer (R3: List Unity)
  │          flush list buffer when non-list item arrives
  │      if TextBlock (paragraph):
  │          → _split_text(text, max_tokens)  if too long (line 129)
  │          → each split → ChunkType.PARAGRAPH
  │          → attach parent_section, section_path (R4: Section Context)
  │      if TableData:
  │          → ChunkType.TABLE
  │          → ldu.table_headers = table.headers  (R1: Header Preservation)
  │          → ldu.figure_alt_text = table.caption  (R2: Caption as Metadata)
  │      if FigureBlock:
  │          → ChunkType.FIGURE
  │          → ldu.figure_alt_text = figure.caption or figure.alt_text (R2)
  │
  ├─ Pass 2: Cross-reference resolution (R5)
  │    build name_registry = {"Table 1": chunk_id, "Figure 3": chunk_id, ...}
  │    for each paragraph LDU:
  │      _extract_xrefs(ldu.content, name_registry)  ← line 89
  │      regex: r"(?:see|per|refer to|as shown in)\s+(Table|Figure|Section)\s+[\dA-Z]+"
  │      if match found in registry → create LDURelationship
  │      ldu.cross_refs.append(relationship)
  │
  └─ ChunkValidator.validate(chunks)  ← line 181
       HARD checks → raise ChunkingRuleViolation:
         token_count > max_chunk_tokens
         TABLE chunk missing table_headers
         FIGURE chunk missing figure_alt_text when caption exists
       SOFT checks → return warnings:
         PARAGRAPH with empty parent_section (R4 miss)
```

### The token-safe text splitter (lines 129–165)

This is the most technically subtle part. Explain why it matters:

```python
def _split_text(text, max_tokens, overlap_tokens=20):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []

    for sent in sentences:
        # CORRECT: measure the JOINED candidate, not sum of parts
        candidate = " ".join(current + [sent])
        if _count_tokens(candidate) > max_tokens and current:
            chunks.append(" ".join(current))
            # overlap: keep last N tokens of previous chunk
            current = current[-overlap_tokens:] + [sent]
        else:
            current.append(sent)

    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Why the naive approach fails:** If you sum individual sentence token counts,
tokenisation boundary effects (BPE merges across word boundaries) cause the
joined text to exceed `max_tokens`. By measuring `_count_tokens(" ".join(...))`
at every step you get the exact count of what will actually be embedded.

### Heading vs. list item guard (lines 121–127)

```python
def _is_list_item(text: str) -> bool:
    return bool(re.match(r'^(\s*[-•*◦▪]|\s*\d+[.)]\s)', text.strip()))
```

**Important:** The chunker checks `not tb.is_heading` before calling this.
Without the guard, `"1. Introduction"` (a heading) would be consumed into
the list accumulator and lose its heading status.

---

## 5. Stage 4 — PageIndexBuilder

**File:** `src/agents/indexer.py`
**Input:** `doc_id`, `filename`, `page_count`, `list[LDU]`
**Output:** `PageIndex` saved to `.refinery/pageindex/{doc_id}.json`

### How to open and explain it

Jump to line 142 (`class PageIndexBuilder`). The `run()` method has 4 steps:

```
PageIndexBuilder.run(doc_id, filename, page_count, ldus) → PageIndex
  │
  ├─ Step 1: Heading skeleton (lines ~183–223)
  │    walk ldus, filter HEADING chunks only
  │    maintain a stack for parent tracking:
  │      stack[-1] = current parent node
  │      when new heading.level ≤ stack[-1].level → pop until parent found
  │    each heading → PageIndexNode(title, level, page_start, page_end=page_start)
  │    wire parent/child links
  │    add to index.root_node_ids or parent.child_node_ids
  │
  ├─ Step 2: Assign content chunks (lines ~226–255)
  │    build section_title_to_node dict from index.nodes
  │    for each non-heading LDU:
  │      match by ldu.parent_section → section_title_to_node[parent_section]
  │      fallback: walk ldu.section_path in reverse until match
  │      node.chunk_ids.append(ldu.chunk_id)
  │      extend node.page_end and all ancestor page_ends
  │
  ├─ Step 3: Reverse index (lines ~259–264)
  │    for each node:
  │      for pg in range(node.page_start, node.page_end + 1):
  │          index.page_to_nodes[pg].append(node.node_id)
  │
  └─ Step 4: Summaries + entities (lines ~268–302)
       for each heading node:
         section_text = concat first 20 child chunks (excluding figures)
         if len(section_text) < min_section_chars:
             node.summary = node.title   ← too short
         elif GEMINI_API_KEY:
             node.summary = _llm_summary(section_text, node.title, model, ...)
         else:
             node.summary = _extractive_summary(section_text)  ← first 2 sentences
         node.key_entities = _extract_entities(section_text)
         node.data_types_present = detect tables/figures in section_ldus
```

### Entity extraction (lines 46–78)

Three regex patterns running on each section's text:

```python
_MONEY_RE  = r'(?:ETB|USD|EUR|birr)?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?'
_DATE_RE   = r'\b(?:FY\s*)?(?:19|20)\d{2}(?:/\d{2,4})? | Q[1-4]\s+\d{4}\b'
_ORG_RE    = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}(?:\s+[A-Z]{2,})*)  \b'
```

Results: up to 5 of each type, capped at 15 total per section.

### `navigate(topic)` — how section search works (page_index.py line 96)

```python
def navigate(self, topic: str) -> list[PageIndexNode]:
    topic_words = set(topic.lower().split())
    for node in self.nodes.values():
        score = 0.0
        score += len(topic_words & title_words) * 2.0      # title match: weight 2
        score += len(topic_words & summary_words) * 1.0    # summary match: weight 1
        for entity in node.key_entities:
            if any(w in entity.lower() for w in topic_words):
                score += 1.5                                 # entity hit: weight 1.5
    return top-3 nodes by score
```

---

## 6. Stage 5 — QueryAgent

**File:** `src/agents/query_agent.py`
**Input:** natural-language question + optional doc_id
**Output:** `ProvenanceChain` (answer + source citations)

### How to open and explain it

Jump to line 90 (`class QueryAgent`). The `answer()` method branches on whether
an API key is set:

```
QueryAgent.answer(question, doc_id) → ProvenanceChain
  │
  ├─ if GEMINI_API_KEY:
  │    QueryAgent._react_answer(question, doc_id)
  │
  └─ else:
       QueryAgent._extractive_answer(question, doc_id)
           VectorStore.search(query=question, top_k=5)
           return top result content as answer + sources
```

### The ReAct loop (lines ~136–190)

```
_react_answer:
  messages = [system_prompt, user: question]

  for iteration in range(MAX_ITERATIONS=6):
    LLM response → _parse_react_response(content)
      │  parses "Thought: ..." and "Action: {...}" or "Final Answer: ..."
      │
      ├─ if step.final_answer:
      │    search vector store for supporting sources
      │    return _build_chain(question, answer, search_results)
      │
      └─ if step.tool_name:
           _dispatch_tool(tool_name, tool_input)
             "search_chunks"   → _tool_search_chunks()  → VectorStore.search()
             "navigate_index"  → _tool_navigate_index() → PageIndex.navigate()
             "query_facts"     → _tool_query_facts()    → FactTable.query()
           append tool result to messages
           continue loop
```

### The three tools

**Tool 1 — search_chunks** (line ~152)
```python
def _tool_search_chunks(self, inp: ChunkSearchInput) -> ToolResult:
    results = self._vs.search(
        query=inp.query,
        top_k=inp.top_k,
        doc_id=inp.doc_id,          # optional doc filter
        chunk_types=inp.chunk_types  # e.g. ["table", "paragraph"]
    )
    return ToolResult(tool_name="search_chunks", success=True, data=[r.model_dump() for r in results])
```

**Tool 2 — navigate_index** (line ~166)
```python
def _tool_navigate_index(self, inp: NavigateInput) -> ToolResult:
    index = self._load_page_index(inp.doc_id)  # lazy-loaded + cached
    nodes = index.navigate(inp.topic)
    return ToolResult(data=[{
        "node_id": n.node_id,
        "title": n.title,
        "page_range": n.page_range_str(),
        "summary": n.summary,
        "chunk_ids": n.chunk_ids[:10],
    } for n in nodes])
```

**Tool 3 — query_facts** (line ~183)
```python
def _tool_query_facts(self, inp: FactQueryInput) -> ToolResult:
    # Safety: SELECT only — no INSERT/UPDATE/DELETE
    if not inp.sql.strip().upper().startswith("SELECT"):
        return ToolResult(success=False, error="Only SELECT queries are permitted")
    rows = self._ft.query(inp.sql, tuple(inp.params))
    return ToolResult(data=rows)
```

### System prompt (line ~254)

The LLM receives this before the user question:
```
You are a document intelligence assistant with access to three tools:
1. search_chunks — semantic search:
   Action: {"tool": "search_chunks", "input": {"query": "...", "top_k": 5}}
2. navigate_index — navigate document sections:
   Action: {"tool": "navigate_index", "input": {"topic": "...", "doc_id": "..."}}
3. query_facts — SQL over structured tables:
   Action: {"tool": "query_facts", "input": {"sql": "SELECT ..."}}

Always respond in this format:
Thought: <your reasoning>
Action: <JSON tool call> OR Final Answer: <your answer>

Ground every claim in retrieved content.
```

### ProvenanceChain builder (line ~225)

```python
def _build_chain(self, question, answer, search_results, doc_id):
    entries = []
    for result in search_results:
        content_hash = hashlib.sha256(result.content.encode()).hexdigest()
        entries.append(ProvenanceEntry(
            doc_id=result.doc_id,
            page_number=result.page_refs[0],
            section_title=result.parent_section,
            chunk_id=result.chunk_id,
            content_hash=content_hash,      # ← verifiable offline
            excerpt=result.content[:200],
            retrieval_score=result.score,
            retrieval_method="semantic_search",
        ))
    return ProvenanceChain(query=question, answer=answer, sources=entries, is_verified=True)
```

---

## 7. Stage 5b — ClaimVerifier (Audit Mode)

**File:** `src/agents/audit.py`
**Input:** natural-language claim + optional doc_id
**Output:** `AuditReport` with per-sub-claim verdicts

### How to open and explain it

Jump to line 100 (`class ClaimVerifier`). The `verify()` method:

```
ClaimVerifier.verify(claim, doc_id) → AuditReport
  │
  ├─ 1. Retrieve evidence
  │    VectorStore.search(query=claim, top_k=8)
  │
  ├─ 2. Extract sub-claims
  │    _extract_sub_claims(claim)        ← line ~134
  │      _NUMBER_RE  finds: "ETB 42 billion", "18%", "12.4B"
  │      _DATE_RE    finds: "FY2023/24", "Q3 2024"
  │      each match + ±20 char context → SubClaim(text=context)
  │
  ├─ 3. Judge each sub-claim
  │    if GEMINI_API_KEY:
  │        _llm_judge(claim, sub_claims, evidence)
  │            for each sub_claim:
  │              prompt LLM: "Is '{sub_claim.text}' supported by this evidence?"
  │              parse JSON: {"verdict": "supported", "confidence": 0.9, "evidence_excerpt": "..."}
  │              → SubClaim.verdict = Verdict.SUPPORTED
  │    else:
  │        _lexical_judge(sub_claims, evidence)
  │            extract numbers from sub_claim
  │            check if those exact numbers appear in evidence_text
  │            num_hits ≥ len(numbers) → SUPPORTED (conf 0.4–0.7)
  │            keyword hits ≥ half → PARTIALLY_SUPPORTED
  │            else → UNVERIFIABLE
  │
  └─ 4. Aggregate verdict
       contradicted > 0         → overall CONTRADICTED
       supported == total       → overall SUPPORTED
       supported + partial > 0  → overall PARTIALLY_SUPPORTED
       else                     → overall UNVERIFIABLE
```

### Verdict enum (line 43)

```python
class Verdict(str, Enum):
    SUPPORTED           = "supported"
    CONTRADICTED        = "contradicted"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNVERIFIABLE        = "unverifiable"
```

---

## 8. Data Stores

### VectorStore — `src/store/vector_store.py`

```
VectorStore.__init__(persist_dir=".refinery/chroma")
  chromadb.PersistentClient(path=persist_dir)
  DefaultEmbeddingFunction()           ← all-MiniLM-L6-v2, downloads once
  client.get_or_create_collection(
      name="refinery-chunks",
      metadata={"hnsw:space": "cosine"}
  )

VectorStore.ingest(ldus, doc_id) → int
  skip FIGURE ldus with no alt_text
  build ids[], documents[], metadatas[]
  upsert in batches of 100
  metadatas store: doc_id, chunk_type, page, page_refs (JSON),
                   parent_section, section_path (JSON), content_hash

VectorStore.search(query, top_k, doc_id, chunk_types) → list[SearchResult]
  build where-filter:
    {"doc_id": {"$eq": doc_id}}         if doc_id provided
    {"chunk_type": {"$in": chunk_types}} if chunk_types provided
    {"$and": [...]}                      if both
  col.query(query_texts=[query], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"])
  score = round(max(0.0, 1.0 - distance / 2.0), 4)
  return sorted by score descending
```

**Why `1.0 - distance/2.0`?**
ChromaDB returns cosine distance in [0, 2] (0=identical, 2=opposite).
Dividing by 2 gives [0, 1], then `1 - x` inverts so higher = more similar.

### FactTable — `src/store/fact_table.py`

```
SQLite schema:
  facts(
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      doc_id      TEXT,
      table_idx   INTEGER,   index within doc.tables[]
      page        INTEGER,
      row_idx     INTEGER,   0-based row within table
      caption     TEXT,
      headers     TEXT,      JSON array: ["Revenue", "FY2023", "FY2024"]
      values      TEXT,      JSON array: ["ETB 42B", "ETB 36B"]
      source_file TEXT,
      created_at  TEXT
  )
  Indexes: idx_facts_doc (doc_id), idx_facts_page (page)

FactTable.extract(doc) → list[FactRow]
  for each table in doc.tables:
    skip if no headers or no rows
    for each row in table.rows:
      pad values to len(headers) with ""
      → FactRow(doc_id, table_idx, page, row_idx, caption, headers, values, source_file)

FactTable.persist(rows) → int
  DELETE existing rows for these doc_ids (idempotent re-runs)
  INSERT rows in batch
  WAL mode for concurrent reads
```

---

## 9. Configuration — `rubric/extraction_rules.yaml`

This is the **single source of truth** for all numeric thresholds. Open it
alongside any agent to see exactly where each number comes from.

```yaml
triage:
  scanned_ratio_high: 0.80      # ≥80% pages have high image area → SCANNED_IMAGE
  scanned_ratio_mixed: 0.20     # 20%–80% → MIXED
  image_area_scanned: 0.80      # a page is "scanned" if image_area_ratio ≥ this
  zero_text_page_ratio: 0.60    # ≥60% truly blank pages → ZERO_TEXT
  form_fillable_min_fields: 1   # ≥1 AcroForm field → FORM_FILLABLE
  table_page_ratio_high: 0.30   # ≥30% of pages have tables → table_heavy
  multi_column_min: 2           # ≥2 estimated columns → multi_column

extraction:
  fast_text:
    min_confidence: 0.75        # below → escalate to Docling
  layout:
    min_confidence: 0.80        # below → escalate to Vision
  human_review_threshold: 0.60  # below this after all tiers → flag

chunking:
  max_chunk_tokens: 512         # hard cap per LDU (tiktoken cl100k_base)
  chunk_overlap_tokens: 20      # sentences kept from prev chunk on split
  min_heading_font_size: 11.0   # below this → not treated as heading

pageindex:
  summary_model: gemini-2.0-flash
  summary_max_tokens: 150
  min_section_chars: 100        # sections with fewer chars skip LLM summary

query:
  model: gemini-2.0-flash

domains:
  financial:
    weight: 1.0
    keywords: [revenue, profit, assets, liabilities, equity, ...]
  medical:
    weight: 1.2    # boosted — medical terms are distinctive
    keywords: [patient, clinical, diagnosis, ...]
```

---

## 10. End-to-End Trace: One PDF, One Query

Follow exactly what happens when you run:
```bash
uv run python -m src.agents.triage "data/data/CBE ANNUAL REPORT 2023-24.pdf"
```

```
1. triage.py __main__ block (line 602)
   agent = TriageAgent()
     _load_config("rubric/extraction_rules.yaml")
     _load_thresholds(config) → thresholds dict
     _build_domain_classifier(config) → DomainClassifier with 4 strategies

2. agent.run("data/data/CBE ANNUAL REPORT 2023-24.pdf")
   doc_id = sha256("...absolute path...")[:16] = "cafb11ca016fe487"
   pdfplumber.open(file_path) → 161 pages

3. Per-page analysis (161 iterations):
   page 1: char_count=1840, image_area_ratio=0.02, table_count=0
   page 34: char_count=320, image_area_ratio=0.04, table_count=3
   ...

4. _detect_form_fillable → pdf.doc.catalog → no "AcroForm" → (False, 0, 0.0)

5. _detect_origin_type:
   zero_text_count = 0/161 = 0% < 60% → not ZERO_TEXT
   scanned_count = 19/161 = 11.8% < 20% → NATIVE_DIGITAL (conf=0.88)

6. _detect_layout_complexity:
   table_pages = 96/161 = 59.6% ≥ 30% → TABLE_HEAVY (conf=1.0)
   col_count = 1

7. DomainClassifier.classify(first 5000 chars):
   financial score = 847, technical = 28, legal = 12, medical = 3
   confidence = 847 / (847+28+12+3) = 0.55 → DomainHint.FINANCIAL

8. ExtractionCost = NEEDS_LAYOUT_MODEL (TABLE_HEAVY triggers layout strategy)

9. DocumentProfile saved → .refinery/profiles/cafb11ca016fe487.json
```

Then for a query:
```bash
uv run python -m src.agents.query_agent "What was net profit?" cafb11ca016fe487
```

```
1. QueryAgent.__init__()
   VectorStore() → loads ChromaDB collection "refinery-chunks"
   _load_page_index("cafb11ca016fe487") → .refinery/pageindex/cafb11ca016fe487.json

2. QueryAgent.answer("What was net profit?", doc_id="cafb11ca016fe487")
   no GEMINI_API_KEY → _extractive_answer()

3. VectorStore.search("What was net profit?", top_k=5, doc_id="cafb11ca016fe487")
   embed query → cosine search → 5 results sorted by score

4. _build_chain(question, answer=results[0].content[:500], search_results)
   for each result:
     content_hash = sha256(result.content)
     ProvenanceEntry(page_number=34, chunk_id="cafb11...-chunk-000340", ...)

5. return ProvenanceChain(
     query="What was net profit?",
     answer="<first 500 chars of top matching chunk>",
     sources=[5 × ProvenanceEntry],
     is_verified=True
   )
```

---

## 11. Key Design Decisions

### Why pdfplumber for Stage 1 triage, not just file metadata?

File metadata (PDF info dict) is unreliable — documents can be copy-scanned
or have incorrect metadata. pdfplumber's per-page `char_density` and
`image_area_ratio` are ground-truth signals computed from actual content.

### Why SHA-256 doc_id instead of UUID?

`doc_id = sha256(absolute_path)[:16]` is deterministic — the same file always
gets the same ID. This makes re-runs idempotent (upsert not duplicate) and
makes document identity stable across machines if the relative path is
consistent.

### Why not LangGraph for the ReAct loop?

LangGraph requires a full stateful graph with typed tool schemas. For this
submission, a simple `for iteration in range(MAX_ITERATIONS)` loop with
`httpx` achieves the same result with zero extra dependencies. The design is
identical in concept — the `TECHNICAL.md` section 11 of FINAL_REPORT.md
explicitly lists migrating to LangGraph as the first improvement.

### Why ChromaDB + sentence-transformers instead of OpenAI embeddings?

`all-MiniLM-L6-v2` runs locally — no API calls, no latency, no cost. The
model downloads once (~90 MB) and runs on CPU. For a 12-document corpus the
quality is indistinguishable from OpenAI embeddings in casual evaluation.

### Why SQLite for FactTable instead of ChromaDB metadata?

ChromaDB metadata values must be strings, integers, or floats — not nested
arrays. Table headers and row values are variable-length arrays. SQLite with
JSON columns handles this naturally and allows arbitrary SQL queries without
schema migrations when table widths vary.

### Why `content_hash` on every LDU?

The `content_hash = sha256(ldu.content)` is computed at model validation time
(Pydantic `@model_validator`). It enables a reviewer to re-extract the same
chunk from the original PDF and verify the hash matches — proving the answer
was not hallucinated or tampered with after extraction. This is the foundation
of the audit trail.

### Why confidence scores on every classification?

Single-label classifiers produce silent failures: a document that is 50%
scanned and 50% native gets classified as MIXED, but there's no signal about
how confident that classification is. A confidence of 0.51 vs 0.95 tells
downstream stages how much to trust the routing decision, and surfaces
borderline cases for human review.
