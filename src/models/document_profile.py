"""DocumentProfile — output of Stage 1 Triage Agent.

Governs which extraction strategy all downstream stages will use.
Stored as JSON in .refinery/profiles/{doc_id}.json.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class OriginType(str, Enum):
    """How the document content was originally created."""
    NATIVE_DIGITAL = "native_digital"   # Characters embedded in PDF
    SCANNED_IMAGE = "scanned_image"     # Pure image — needs OCR
    MIXED = "mixed"                     # Some pages digital, some scanned
    FORM_FILLABLE = "form_fillable"     # Interactive PDF with AcroForm fields
    ZERO_TEXT = "zero_text"             # No extractable text on any page —
                                        # blank, decorative, or purely graphical;
                                        # even Vision models may yield little


class LayoutComplexity(str, Enum):
    """Structural complexity of the document layout."""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"         # > 30% of pages contain tables
    FIGURE_HEAVY = "figure_heavy"       # > 30% of pages contain figures
    MIXED = "mixed"                     # Combination of the above


class DomainHint(str, Enum):
    """Content domain — influences extraction prompt strategy for VLM."""
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"


class ExtractionCost(str, Enum):
    """Estimated extraction cost tier based on triage analysis."""
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"       # Strategy A
    NEEDS_LAYOUT_MODEL = "needs_layout_model"           # Strategy B
    NEEDS_VISION_MODEL = "needs_vision_model"           # Strategy C


class PageStats(BaseModel):
    """Per-page statistics collected during triage."""
    page_number: int
    char_count: int
    char_density: float = Field(description="Characters per point² of page area")
    image_area_ratio: float = Field(description="Fraction of page area covered by images (0-1)")
    has_tables: bool = False
    has_figures: bool = False
    is_likely_scanned: bool = False


class DocumentProfile(BaseModel):
    """Complete classification of a document produced by the Triage Agent.

    This profile is the single source of truth for all downstream routing decisions.
    Classification confidence scores allow the router to weight triage signals
    and decide how aggressively to escalate between extraction strategies.
    """

    # Identity
    doc_id: str = Field(description="SHA256 hash of file content (first 16 chars)")
    filename: str
    file_path: str
    page_count: int
    file_size_bytes: int

    # Classification dimensions
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: str = Field(default="en", description="ISO 639-1 language code")
    language_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    domain_hint: DomainHint
    estimated_extraction_cost: ExtractionCost

    # Classification confidence scores (0.0 = ambiguous, 1.0 = certain)
    # These are emitted by the triage classifier and consumed by ExtractionRouter
    # to decide initial strategy selection and escalation aggressiveness.
    origin_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence in origin_type; low when scanned ratio is near a threshold",
    )
    layout_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence in layout_complexity; low when multiple signals tie",
    )
    domain_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Confidence in domain_hint; ratio of best-domain keyword hits to total hits. "
            "0.0 means no domain keywords found (defaults to GENERAL)."
        ),
    )

    # Form-fillable detection
    is_form_fillable: bool = Field(
        default=False,
        description="True when PDF contains interactive AcroForm fields",
    )
    form_field_count: int = Field(
        default=0,
        description="Number of AcroForm fields detected; 0 when is_form_fillable is False",
    )

    # Evidence behind classification
    avg_char_density: float = Field(description="Mean char density across all pages")
    avg_image_area_ratio: float = Field(description="Mean image area ratio across all pages")
    scanned_page_count: int = Field(default=0, description="Pages with near-zero character stream")
    zero_text_page_count: int = Field(
        default=0,
        description="Pages with char_count == 0 (no extractable characters at all)",
    )
    table_page_count: int = Field(default=0, description="Pages with detected tables")
    column_count_estimate: int = Field(default=1, description="Estimated number of text columns")

    # Per-page breakdown
    page_stats: list[PageStats] = Field(default_factory=list)

    # Metadata
    triage_version: str = "1.1.0"
    processing_time_seconds: Optional[float] = None

    @classmethod
    def profile_path(cls, doc_id: str, base_dir: str = ".refinery/profiles") -> Path:
        return Path(base_dir) / f"{doc_id}.json"

    def save(self, base_dir: str = ".refinery/profiles") -> Path:
        path = self.profile_path(self.doc_id, base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, doc_id: str, base_dir: str = ".refinery/profiles") -> "DocumentProfile":
        path = cls.profile_path(doc_id, base_dir)
        return cls.model_validate_json(path.read_text())
