"""Stage 3: ChunkingEngine — Semantic Chunking with 5-Rule Constitution.

Takes an ExtractedDocument and produces a list of LDUs (Logical Document Units)
that are RAG-ready, rule-compliant chunks with full spatial provenance.

Five rules enforced (all thresholds read from rubric/extraction_rules.yaml):

  R1 — Table header protection:
       TABLE LDUs always include the header row; a table LDU with no headers
       is a hard validator violation.

  R2 — Caption as metadata:
       Figure captions are stored in LDU.figure_caption, never emitted as a
       standalone CAPTION LDU. ChunkValidator raises on any CAPTION chunk.

  R3 — List unity:
       Consecutive list-item text blocks are merged into a single LIST LDU
       unless the merged text exceeds rule_r3_list_max_tokens, in which case
       it is split at item boundaries (not mid-item).

  R4 — Section context:
       Every non-heading LDU carries parent_section (most recent heading text)
       and section_path (full ancestor chain). ChunkValidator warns when missing.

  R5 — Cross-reference resolution:
       "Table N", "Figure N", "Section N.N" patterns in paragraph/list text are
       detected and stored as LDURelationship objects pointing to known chunk IDs.
       Unresolvable references carry target_chunk_id="unresolved".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import yaml

from src.models.extracted_document import (
    ExtractedDocument,
    FigureBlock,
    TableData,
    TextBlock,
)
from src.models.ldu import LDU, LDURelationship, ChunkType

# ---------------------------------------------------------------------------
# Token counting (tiktoken when available, char estimate otherwise)
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_chunking_config(rules_path: str = "rubric/extraction_rules.yaml") -> dict:
    """Read the chunking section from extraction_rules.yaml."""
    path = Path(rules_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
            return data.get("chunking", {})
    return {}


# ---------------------------------------------------------------------------
# Cross-reference detection (R5)
# ---------------------------------------------------------------------------

_XREF_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bTable\s+\d+(?:\.\d+)?', re.IGNORECASE), "references_table"),
    (re.compile(r'\bFigure\s+\d+(?:\.\d+)?', re.IGNORECASE), "references_figure"),
    (re.compile(r'\b(?:see\s+)?(?:Section|Chapter|Annex)\s+[\d]+(?:\.[\d]+)*', re.IGNORECASE), "see_also"),
]


def _extract_xrefs(text: str, name_registry: dict[str, str]) -> list[LDURelationship]:
    """Find cross-reference patterns and resolve to chunk IDs where possible."""
    refs: list[LDURelationship] = []
    seen: set[str] = set()

    for pattern, rel_type in _XREF_PATTERNS:
        for match in pattern.finditer(text):
            anchor = match.group(0).strip()
            key = anchor.lower()
            if key in seen:
                continue
            seen.add(key)
            target_id = name_registry.get(key, "unresolved")
            refs.append(LDURelationship(
                target_chunk_id=target_id,
                relationship_type=rel_type,
                anchor_text=anchor,
            ))

    return refs


# ---------------------------------------------------------------------------
# List item detection (R3)
# ---------------------------------------------------------------------------

_LIST_ITEM_RE = re.compile(
    r'^[\s]*(?:[\u2022\u25cf\u2013\u2014\-\*\+]|\d+[\.\)]|[a-zA-Z][\.\)])\s',
    re.MULTILINE,
)


def _is_list_item(text: str) -> bool:
    return bool(_LIST_ITEM_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Text splitting (long paragraphs)
# ---------------------------------------------------------------------------

def _split_text(text: str, max_tokens: int, overlap_tokens: int = 20) -> list[str]:
    """Split text at sentence boundaries into chunks of at most max_tokens.

    Measures token count on the actual joined string (not sum of individual
    sentence counts) to avoid tokenisation boundary discrepancies.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current: list[str] = []

    for sent in sentences:
        candidate = " ".join(current + [sent]) if current else sent
        if _count_tokens(candidate) > max_tokens and current:
            chunks.append(" ".join(current))
            # Build overlap from the tail of current (measured as joined string)
            overlap: list[str] = []
            for s in reversed(current):
                trial = " ".join([s] + overlap)
                if _count_tokens(trial) <= overlap_tokens:
                    overlap.insert(0, s)
                else:
                    break
            current = overlap
        current.append(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


# ---------------------------------------------------------------------------
# ChunkingRuleViolation
# ---------------------------------------------------------------------------

class ChunkingRuleViolation(Exception):
    """Raised by ChunkValidator when a hard rule is breached."""

    def __init__(self, rule: str, chunk_id: str, detail: str):
        self.rule = rule
        self.chunk_id = chunk_id
        self.detail = detail
        super().__init__(f"[{rule}] chunk={chunk_id}: {detail}")


# ---------------------------------------------------------------------------
# ChunkValidator
# ---------------------------------------------------------------------------

class ChunkValidator:
    """Re-reads the LDU list and enforces the 5-rule constitution.

    Hard violations (R1, R2, token overrun) → ChunkingRuleViolation.
    Soft violations (R4 missing parent_section) → returned as warning strings.
    """

    def __init__(self, cfg: Optional[dict] = None):
        c = cfg or {}
        self.max_tokens = c.get("max_tokens_per_chunk", 512)
        self.r1 = c.get("rule_r1_table_header_protection", True)
        self.r2 = c.get("rule_r2_caption_as_metadata", True)
        self.r4 = c.get("rule_r4_section_context", True)

    def validate(self, chunks: list[LDU]) -> list[str]:
        """Validate chunks. Returns list of soft-warning strings.

        Raises ChunkingRuleViolation on any hard breach.
        """
        warnings: list[str] = []

        for chunk in chunks:
            # Token upper-bound (hard)
            if chunk.token_count > self.max_tokens:
                raise ChunkingRuleViolation(
                    "TOKEN_LIMIT", chunk.chunk_id,
                    f"token_count={chunk.token_count} > max={self.max_tokens}",
                )

            # R1: TABLE must carry table_headers (hard)
            if self.r1 and chunk.chunk_type == ChunkType.TABLE:
                if not chunk.table_headers:
                    raise ChunkingRuleViolation(
                        "R1_TABLE_HEADER", chunk.chunk_id,
                        "TABLE LDU has no table_headers — header row protection violated",
                    )

            # R2: no standalone CAPTION chunks (hard)
            if self.r2 and chunk.chunk_type == ChunkType.CAPTION:
                raise ChunkingRuleViolation(
                    "R2_CAPTION_METADATA", chunk.chunk_id,
                    "CAPTION LDU must not be standalone — store as figure metadata",
                )

            # R4: non-heading chunks should carry parent_section (soft)
            if self.r4 and chunk.chunk_type not in (ChunkType.HEADING,):
                if chunk.parent_section is None:
                    warnings.append(
                        f"[R4] {chunk.chunk_id} ({chunk.chunk_type.value}): "
                        f"no parent_section set"
                    )

        return warnings


# ---------------------------------------------------------------------------
# ChunkingEngine
# ---------------------------------------------------------------------------

class ChunkingEngine:
    """Stage 3 agent — converts ExtractedDocument into a validated List[LDU].

    All thresholds are read from rubric/extraction_rules.yaml at construction.
    To change chunking behaviour, edit the YAML — no code changes needed.
    """

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml"):
        self._cfg = _load_chunking_config(rules_path)
        self.max_tokens = self._cfg.get("max_tokens_per_chunk", 512)
        self.min_tokens = self._cfg.get("min_tokens_per_chunk", 50)
        self.overlap = self._cfg.get("chunk_overlap_tokens", 20)
        self.list_max_tokens = self._cfg.get("rule_r3_list_max_tokens", 512)
        self.validator = ChunkValidator(self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, doc: ExtractedDocument) -> list[LDU]:
        """Chunk *doc* into validated LDUs.

        Returns the LDU list. R4 soft warnings are printed to stderr.
        Raises ChunkingRuleViolation on hard rule breaches.
        """
        chunks: list[LDU] = []
        seq = 0

        # R4 state — track current heading context
        current_section: Optional[str] = None
        current_path: list[str] = []

        # R5 name registry: "table 1" / "figure 2" → chunk_id
        name_registry: dict[str, str] = {}
        table_seq = 0
        figure_seq = 0

        # R3 list accumulator
        list_buffer: list[TextBlock] = []

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _make_id() -> str:
            nonlocal seq
            cid = LDU.make_chunk_id(doc.doc_id, seq)
            seq += 1
            return cid

        def _flush_list() -> None:
            """Emit accumulated list items as one (or split) LIST LDU."""
            nonlocal list_buffer
            if not list_buffer:
                return

            combined = "\n".join(b.text.strip() for b in list_buffer)
            pages = sorted(set(b.bbox.page for b in list_buffer))

            if _count_tokens(combined) > self.list_max_tokens:
                # R3: split at item boundaries rather than mid-item
                items_text = [b.text.strip() for b in list_buffer]
                current_items: list[str] = []
                current_tok = 0
                for item_text in items_text:
                    itok = _count_tokens(item_text)
                    if current_tok + itok > self.max_tokens and current_items:
                        _emit_list(current_items, pages, list_buffer[0])
                        current_items = []
                        current_tok = 0
                    current_items.append(item_text)
                    current_tok += itok
                if current_items:
                    _emit_list(current_items, pages, list_buffer[0])
            else:
                cid = _make_id()
                chunks.append(LDU(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    content=combined,
                    chunk_type=ChunkType.LIST,
                    token_count=_count_tokens(combined),
                    page_refs=pages,
                    bounding_box=list_buffer[0].bbox if len(list_buffer) == 1 else None,
                    parent_section=current_section,
                    section_path=list(current_path),
                    content_hash="",
                ))

            list_buffer = []

        def _emit_list(items: list[str], pages: list[int], first_block: TextBlock) -> None:
            text = "\n".join(items)
            cid = _make_id()
            chunks.append(LDU(
                chunk_id=cid,
                doc_id=doc.doc_id,
                content=text,
                chunk_type=ChunkType.LIST,
                token_count=_count_tokens(text),
                page_refs=pages,
                bounding_box=first_block.bbox,
                parent_section=current_section,
                section_path=list(current_path),
                content_hash="",
            ))

        # ------------------------------------------------------------------
        # Merge all content into one reading-order stream
        # ------------------------------------------------------------------

        stream: list[tuple[int, object]] = []
        for tb in doc.text_blocks:
            stream.append((tb.reading_order, tb))
        for td in doc.tables:
            stream.append((td.reading_order, td))
        for fg in doc.figures:
            stream.append((fg.reading_order, fg))
        stream.sort(key=lambda x: x[0])

        # ------------------------------------------------------------------
        # Pass 1: build LDUs
        # ------------------------------------------------------------------

        for _, item in stream:

            # ---- TextBlock ----
            if isinstance(item, TextBlock):
                tb: TextBlock = item

                # R3: accumulate list items — but headings take priority even if text starts with "1."
                if (
                    not tb.is_heading
                    and self._cfg.get("rule_r3_list_unity", True)
                    and _is_list_item(tb.text)
                ):
                    list_buffer.append(tb)
                    continue

                _flush_list()  # end of any previous list run

                if tb.is_heading:
                    # R4: update section context
                    level = tb.heading_level or 1
                    title = tb.text.strip()
                    if level == 1:
                        current_path = [title]
                    elif level == 2:
                        current_path = current_path[:1] + [title]
                    else:
                        current_path = current_path[:2] + [title]
                    current_section = title

                    cid = _make_id()
                    chunks.append(LDU(
                        chunk_id=cid,
                        doc_id=doc.doc_id,
                        content=title,
                        chunk_type=ChunkType.HEADING,
                        token_count=_count_tokens(title),
                        page_refs=[tb.bbox.page],
                        bounding_box=tb.bbox,
                        parent_section=None,
                        section_path=list(current_path),
                        heading_level=level,
                        content_hash="",
                    ))

                else:
                    # Paragraph — split if needed
                    parts = _split_text(tb.text, self.max_tokens, self.overlap)
                    for part in parts:
                        cid = _make_id()
                        chunks.append(LDU(
                            chunk_id=cid,
                            doc_id=doc.doc_id,
                            content=part,
                            chunk_type=ChunkType.PARAGRAPH,
                            token_count=_count_tokens(part),
                            page_refs=[tb.bbox.page],
                            bounding_box=tb.bbox,
                            parent_section=current_section,
                            section_path=list(current_path),
                            content_hash="",
                        ))

            # ---- TableData ----
            elif isinstance(item, TableData):
                _flush_list()
                td: TableData = item
                table_seq += 1

                # R1: ensure headers are present; fall back to first row
                if td.headers:
                    headers = td.headers
                    data_rows = td.rows
                else:
                    headers = td.rows[0] if td.rows else []
                    data_rows = td.rows[1:] if len(td.rows) > 1 else []

                # Format as readable text
                header_line = " | ".join(headers) if headers else "(no headers)"
                row_lines = [" | ".join(str(c) for c in row) for row in data_rows]
                table_text = header_line + "\n" + "\n".join(row_lines)
                if td.caption:
                    table_text = f"{td.caption}\n{table_text}"

                cid = _make_id()
                name_registry[f"table {table_seq}"] = cid

                chunks.append(LDU(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    content=table_text,
                    chunk_type=ChunkType.TABLE,
                    token_count=_count_tokens(table_text),
                    page_refs=[td.page],
                    bounding_box=td.bbox,
                    parent_section=current_section,
                    section_path=list(current_path),
                    table_headers=headers,
                    table_rows=data_rows[:20],  # cap stored rows at 20
                    content_hash="",
                ))

            # ---- FigureBlock ----
            elif isinstance(item, FigureBlock):
                _flush_list()
                fg: FigureBlock = item
                figure_seq += 1

                # R2: caption → metadata field only, not standalone LDU
                figure_content = fg.alt_text or f"[Figure {figure_seq}: {fg.figure_id}]"

                cid = _make_id()
                name_registry[f"figure {figure_seq}"] = cid

                chunks.append(LDU(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    content=figure_content,
                    chunk_type=ChunkType.FIGURE,
                    token_count=_count_tokens(figure_content),
                    page_refs=[fg.page],
                    bounding_box=fg.bbox,
                    parent_section=current_section,
                    section_path=list(current_path),
                    figure_caption=fg.caption,     # R2: stored here, not as LDU
                    figure_alt_text=fg.alt_text,
                    content_hash="",
                ))

        # Flush any trailing list items
        _flush_list()

        # ------------------------------------------------------------------
        # Pass 2: resolve cross-references (R5)
        # ------------------------------------------------------------------

        if self._cfg.get("rule_r5_cross_reference_resolution", True):
            for i, chunk in enumerate(chunks):
                if chunk.chunk_type in (ChunkType.PARAGRAPH, ChunkType.LIST):
                    xrefs = _extract_xrefs(chunk.content, name_registry)
                    if xrefs:
                        chunks[i].relationships = xrefs

        # ------------------------------------------------------------------
        # Validate and return
        # ------------------------------------------------------------------

        warnings = self.validator.validate(chunks)
        for w in warnings:
            print(f"[ChunkValidator] {w}", file=sys.stderr)

        return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from rich.console import Console
    from rich.table import Table as RichTable

    console = Console()

    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.agents.chunker <path/to/extracted_doc.json>[/red]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        from src.models.extracted_document import ExtractedDocument as ED
        doc = ED.model_validate_json(f.read())

    engine = ChunkingEngine()
    ldus = engine.run(doc)

    t = RichTable(title=f"LDUs — {doc.filename} ({len(ldus)} chunks)", show_header=True)
    t.add_column("chunk_id", style="cyan", no_wrap=True)
    t.add_column("type", style="magenta")
    t.add_column("tokens", justify="right")
    t.add_column("page(s)", justify="right")
    t.add_column("parent_section", style="green")
    t.add_column("content[:60]", style="white")

    for ldu in ldus:
        t.add_row(
            ldu.chunk_id,
            ldu.chunk_type.value,
            str(ldu.token_count),
            ",".join(str(p) for p in ldu.page_refs),
            ldu.parent_section or "—",
            ldu.content[:60].replace("\n", " "),
        )

    console.print(t)
    console.print(f"\n[bold green]✓ {len(ldus)} LDUs produced[/bold green]")
