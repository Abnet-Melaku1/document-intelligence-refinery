"""ExtractedDocument — normalized output of Stage 2 Structure Extraction Layer.

All three extraction strategies (FastText, Layout, Vision) must produce this schema.
This is the internal contract between the extraction layer and the chunking engine.

Routing metadata (RoutingDecision, StrategyAttempt) is embedded directly in the
ExtractedDocument so that every downstream stage — chunker, indexer, query agent —
can inspect exactly how the document was routed and whether it needs human review.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Spatial coordinates in PDF point units (1 pt = 1/72 inch).

    Origin is bottom-left for pdfplumber; top-left for Docling.
    We normalize to top-left origin (PDF standard coordinates).
    """
    x0: float  # Left edge
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge
    page: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}


class TextBlock(BaseModel):
    """A contiguous block of extracted text with spatial provenance."""
    text: str
    bbox: BoundingBox
    reading_order: int = Field(description="Position in document reading order (0-indexed)")
    font_size: Optional[float] = None
    is_heading: bool = False
    heading_level: Optional[int] = None  # 1=H1, 2=H2, etc.
    section_path: list[str] = Field(
        default_factory=list,
        description="Ancestor section titles e.g. ['3. Results', '3.2 Analysis']"
    )


class TableCell(BaseModel):
    """Single cell within an extracted table."""
    row: int
    col: int
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


class TableData(BaseModel):
    """Structured table extracted with row/column semantics preserved."""
    caption: Optional[str] = None
    bbox: BoundingBox
    page: int
    headers: list[str] = Field(description="Column header labels")
    rows: list[list[str]] = Field(description="Data rows as list of cell values")
    cells: list[TableCell] = Field(default_factory=list, description="Full cell objects with span info")
    reading_order: int = 0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)


class FigureBlock(BaseModel):
    """An embedded figure (image, chart, diagram) with its caption."""
    figure_id: str
    bbox: BoundingBox
    page: int
    caption: Optional[str] = None
    caption_bbox: Optional[BoundingBox] = None
    alt_text: Optional[str] = None  # VLM-generated description
    reading_order: int = 0


class ExtractionStrategy(str, Enum):
    FAST_TEXT = "fast_text"       # Strategy A: pdfplumber
    LAYOUT = "layout"             # Strategy B: Docling
    VISION = "vision"             # Strategy C: VLM


# ---------------------------------------------------------------------------
# Routing metadata — embedded in every ExtractedDocument
# ---------------------------------------------------------------------------

class EscalationReason(str, Enum):
    """Why a strategy attempt did not produce an acceptable result."""
    LOW_CONFIDENCE = "low_confidence"       # Confidence below threshold
    EXCEPTION_RAISED = "exception_raised"   # Strategy raised an unhandled exception
    BUDGET_EXHAUSTED = "budget_exhausted"   # Vision budget cap reached mid-document


class StrategyAttempt(BaseModel):
    """Record of one strategy's execution — whether it succeeded or triggered escalation.

    All attempts are preserved in order so reviewers can see the full escalation trail.
    """
    strategy: ExtractionStrategy
    confidence_score: float = Field(ge=0.0, le=1.0)
    escalated: bool = Field(
        description="True if this attempt's confidence was below threshold and the router moved on"
    )
    escalation_reason: Optional[EscalationReason] = Field(
        default=None,
        description="Why escalation was triggered. None when escalated=False."
    )
    escalation_detail: Optional[str] = Field(
        default=None,
        description="Human-readable detail: threshold value, exception message, etc."
    )
    cost_usd: float = 0.0
    processing_time_seconds: Optional[float] = None


class RoutingDecision(BaseModel):
    """The router's initial strategy selection and its justification.

    Captures the DocumentProfile signals that drove the routing choice so that
    the decision is reproducible and auditable without re-running triage.
    """
    initial_strategy: ExtractionStrategy
    selection_reason: str = Field(
        description="Human-readable explanation of why this strategy was the starting point"
    )
    # Key DocumentProfile signals that determined the starting strategy
    origin_type: str
    layout_complexity: str
    estimated_extraction_cost: str
    avg_char_density: float
    avg_image_area_ratio: float
    scanned_page_count: int
    # The full ordered chain the router will attempt (first = cheapest that should work)
    strategy_chain: list[ExtractionStrategy]


class ExtractedDocument(BaseModel):
    """Normalized extraction output — the common schema all strategies produce.

    Adapters for Docling, MinerU, and VLM output must all produce this model.

    Routing metadata fields (routing_decision, strategy_attempts, requires_human_review)
    are populated by ExtractionRouter after extraction completes. Individual strategy
    implementations leave them at defaults.
    """

    doc_id: str
    filename: str
    page_count: int

    # Content blocks in reading order
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    figures: list[FigureBlock] = Field(default_factory=list)

    # Extraction metadata
    strategy_used: ExtractionStrategy
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall extraction confidence")
    cost_estimate_usd: float = Field(default=0.0, description="Estimated API cost (0 for local strategies)")
    processing_time_seconds: Optional[float] = None
    extraction_version: str = "1.0.0"

    # Routing provenance — populated by ExtractionRouter, not by individual strategies
    routing_decision: Optional[RoutingDecision] = Field(
        default=None,
        description="Why the initial strategy was selected and the full strategy chain"
    )
    strategy_attempts: list[StrategyAttempt] = Field(
        default_factory=list,
        description="Every strategy attempted in order, with individual confidence scores"
    )

    # Human review flag — set when final tier confidence is still below human_review_threshold
    requires_human_review: bool = Field(
        default=False,
        description="True when even the terminal strategy did not achieve acceptable confidence"
    )
    human_review_reason: Optional[str] = Field(
        default=None,
        description="Explanation of why human review is needed (shown in CLI and ledger)"
    )

    # Warnings produced during extraction
    warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """All text blocks concatenated in reading order."""
        sorted_blocks = sorted(self.text_blocks, key=lambda b: b.reading_order)
        return "\n\n".join(b.text for b in sorted_blocks)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def figure_count(self) -> int:
        return len(self.figures)

    @property
    def escalation_count(self) -> int:
        """Number of strategies that escalated (= total attempts - 1 if final succeeded)."""
        return sum(1 for a in self.strategy_attempts if a.escalated)
