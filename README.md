# Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous document corpora and emits structured, queryable, spatially-indexed knowledge.

Solves three enterprise failure modes: **Structure Collapse**, **Context Poverty**, and **Provenance Blindness**.

## Architecture

```
Document Input
     │
     ▼
┌─────────────────────┐
│  Stage 1: Triage    │  → DocumentProfile (origin, layout, domain, cost estimate)
│  Agent              │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2: Structure │  → Strategy A (pdfplumber) / B (Docling) / C (VLM)
│  Extraction Layer   │    with confidence-gated escalation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3: Semantic  │  → List[LDU] with content_hash + provenance
│  Chunking Engine    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 4: PageIndex │  → Hierarchical navigation tree with LLM summaries
│  Builder            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 5: Query     │  → Natural language Q&A with ProvenanceChain citations
│  Interface Agent    │
└─────────────────────┘
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
# Clone the repo
git clone <repo-url>
cd document-intelligence-refinery

# Install with uv (recommended)
uv venv --python 3.11
uv pip install -e "."
```

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here        # Get free at aistudio.google.com
VLM_BUDGET_CAP_USD=0.10             # max spend per document
```

> **No API key needed** for Stages 1–3 (triage, extraction, chunking) and
> vector search — those are fully local. `GEMINI_API_KEY` is only required
> for LLM-powered section summaries (Stage 4) and the QueryAgent ReAct loop
> (Stage 5). A free Gemini API key gives 1,500 requests/day.

### Run Triage on a Document

```bash
uv run python -m src.agents.triage "data/data/CBE ANNUAL REPORT 2023-24.pdf"
```

### Run Full Extraction Pipeline

```bash
uv run python -m src.agents.extractor "data/data/CBE ANNUAL REPORT 2023-24.pdf"
```

### Query a Document

```bash
# Extractive mode (no API key needed)
uv run python -m src.agents.query_agent "What was net profit?" <doc_id>

# Full ReAct mode with Gemini
$env:GEMINI_API_KEY="AIza..."
uv run python -m src.agents.query_agent "What was net profit?" <doc_id>
```

### Run Tests

```bash
uv run pytest tests/test_triage.py tests/test_chunker.py -v
```

## Project Structure

```
src/
├── models/          # Pydantic schemas for all pipeline data
├── agents/          # Triage Agent and ExtractionRouter
└── strategies/      # FastText, Layout, Vision extractors

rubric/
└── extraction_rules.yaml   # All thresholds and chunking rules (config, not code)

.refinery/
├── profiles/        # DocumentProfile JSON per document
└── extraction_ledger.jsonl  # Audit log for every extraction run

tests/               # Unit tests
```

## Extraction Strategy Decision

| Condition | Strategy | Tool | Cost |
|-----------|----------|------|------|
| native_digital + single_column | A — Fast Text | pdfplumber | Low |
| multi_column OR table_heavy OR mixed | B — Layout-Aware | Docling | Medium |
| scanned_image OR confidence < threshold | C — Vision | Gemini 2.0 Flash (google-genai) | High |

Thresholds are defined in `rubric/extraction_rules.yaml` and documented in `DOMAIN_NOTES.md`.

## Corpus Support

Validated against 50 Ethiopian financial and government PDF documents across four classes:

- **Class A**: Native digital annual reports (CBE, EthSwitch, ETS)
- **Class B**: Scanned audit reports (DBE, government financial statements)
- **Class C**: Mixed technical assessment reports (FTA, pharmaceutical studies)
- **Class D**: Table-heavy fiscal data reports (tax expenditure, consumer price index)
