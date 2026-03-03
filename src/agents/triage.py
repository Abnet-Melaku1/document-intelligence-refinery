"""Stage 1: Triage Agent — Document Classifier.

Produces a DocumentProfile that governs all downstream extraction strategy decisions.
Uses pdfplumber for character density analysis, image area measurement, and
bounding box layout analysis. No ML model required — pure heuristics.

Classification dimensions:
  - origin_type: native_digital | scanned_image | mixed | form_fillable
  - layout_complexity: single_column | multi_column | table_heavy | figure_heavy | mixed
  - domain_hint: financial | legal | technical | medical | general
  - estimated_extraction_cost: fast_text_sufficient | needs_layout_model | needs_vision_model
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

import pdfplumber
import yaml

from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LayoutComplexity,
    OriginType,
    PageStats,
)

# ---------------------------------------------------------------------------
# Default thresholds (can be overridden by extraction_rules.yaml)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "char_density_min": 0.05,          # chars/pt² — below = likely scanned
    "char_density_scanned": 0.001,     # chars/pt² — hard scanned cutoff
    "image_area_scanned": 0.80,        # fraction — image dominates page
    "image_area_max": 0.50,            # fraction — triggers layout concern
    "scanned_page_ratio": 0.40,        # fraction of pages that are scanned → doc is mixed
    "scanned_page_ratio_hard": 0.80,   # fraction → doc is fully scanned
    "table_page_ratio": 0.30,          # fraction of pages with tables → table_heavy
    "multi_column_gap_min": 50.0,      # pt — min gap between text columns
    "multi_column_count": 2,           # detected columns → multi_column
}


def _load_thresholds(rules_path: Optional[str] = None) -> dict:
    """Load thresholds from extraction_rules.yaml, falling back to defaults."""
    if rules_path is None:
        rules_path = "rubric/extraction_rules.yaml"

    path = Path(rules_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            if data and "triage" in data:
                return {**DEFAULT_THRESHOLDS, **data["triage"]}

    return DEFAULT_THRESHOLDS.copy()


def _doc_id(file_path: str) -> str:
    """First 16 chars of SHA256 of file content."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-page analysis
# ---------------------------------------------------------------------------

def _analyze_page(page: pdfplumber.page.Page, page_num: int) -> PageStats:
    """Compute extraction-relevant statistics for a single PDF page."""
    page_area = page.width * page.height  # in pt²

    # Character stream
    chars = page.chars
    char_count = len(chars)
    char_density = char_count / page_area if page_area > 0 else 0.0

    # Image area
    images = page.images
    image_area = sum(
        (img.get("width", 0) * img.get("height", 0)) for img in images
    )
    image_area_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0.0

    # Table detection (lightweight — checks if extract_tables returns anything)
    try:
        tables = page.find_tables()
        has_tables = len(tables) > 0
    except Exception:
        has_tables = False

    # Figure detection (images that are large enough to be figures)
    has_figures = any(
        (img.get("width", 0) * img.get("height", 0)) > 10_000
        for img in images
    )

    t = DEFAULT_THRESHOLDS
    is_likely_scanned = (
        char_density < t["char_density_scanned"]
        or image_area_ratio > t["image_area_scanned"]
    )

    return PageStats(
        page_number=page_num,
        char_count=char_count,
        char_density=round(char_density, 6),
        image_area_ratio=round(image_area_ratio, 4),
        has_tables=has_tables,
        has_figures=has_figures,
        is_likely_scanned=is_likely_scanned,
    )


# ---------------------------------------------------------------------------
# Origin type detection
# ---------------------------------------------------------------------------

def _detect_origin_type(page_stats: list[PageStats], thresholds: dict) -> OriginType:
    """Classify document origin from per-page character density and image area."""
    total = len(page_stats)
    if total == 0:
        return OriginType.SCANNED_IMAGE

    scanned_count = sum(1 for p in page_stats if p.is_likely_scanned)
    scanned_ratio = scanned_count / total

    if scanned_ratio >= thresholds["scanned_page_ratio_hard"]:
        return OriginType.SCANNED_IMAGE
    elif scanned_ratio >= thresholds["scanned_page_ratio"]:
        return OriginType.MIXED
    else:
        return OriginType.NATIVE_DIGITAL


# ---------------------------------------------------------------------------
# Layout complexity detection
# ---------------------------------------------------------------------------

def _estimate_column_count(page: pdfplumber.page.Page) -> int:
    """Estimate number of text columns via x-coordinate gap analysis."""
    chars = page.chars
    if not chars:
        return 1

    # Collect x0 positions of word starts
    x0_positions = sorted(set(round(c["x0"]) for c in chars if c["text"].strip()))
    if len(x0_positions) < 4:
        return 1

    # Find large gaps in x0 distribution — gaps > multi_column_gap_min suggest column breaks
    gaps = []
    for i in range(1, len(x0_positions)):
        gap = x0_positions[i] - x0_positions[i - 1]
        if gap > DEFAULT_THRESHOLDS["multi_column_gap_min"]:
            gaps.append(gap)

    # Heuristic: one big gap = 2 columns, two gaps = 3 columns, etc.
    return min(len(gaps) + 1, 4)


def _detect_layout_complexity(
    pdf: pdfplumber.PDF,
    page_stats: list[PageStats],
    thresholds: dict,
) -> tuple[LayoutComplexity, int]:
    """Classify layout complexity and return (complexity, estimated_column_count)."""
    total = len(page_stats)
    if total == 0:
        return LayoutComplexity.SINGLE_COLUMN, 1

    table_count = sum(1 for p in page_stats if p.has_tables)
    figure_count = sum(1 for p in page_stats if p.has_figures)
    table_ratio = table_count / total
    figure_ratio = figure_count / total

    # Sample up to 5 non-scanned pages for column detection
    sample_pages = [
        pdf.pages[p.page_number - 1]
        for p in page_stats
        if not p.is_likely_scanned
    ][:5]

    col_counts = [_estimate_column_count(p) for p in sample_pages]
    avg_cols = sum(col_counts) / len(col_counts) if col_counts else 1
    estimated_cols = round(avg_cols)

    has_multi_column = estimated_cols >= thresholds["multi_column_count"]
    is_table_heavy = table_ratio >= thresholds["table_page_ratio"]
    is_figure_heavy = figure_ratio >= thresholds["table_page_ratio"]

    flags = sum([has_multi_column, is_table_heavy, is_figure_heavy])

    if flags >= 2:
        return LayoutComplexity.MIXED, estimated_cols
    elif is_table_heavy:
        return LayoutComplexity.TABLE_HEAVY, estimated_cols
    elif is_figure_heavy:
        return LayoutComplexity.FIGURE_HEAVY, estimated_cols
    elif has_multi_column:
        return LayoutComplexity.MULTI_COLUMN, estimated_cols
    else:
        return LayoutComplexity.SINGLE_COLUMN, 1


# ---------------------------------------------------------------------------
# Domain hint classifier (keyword-based, pluggable)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[DomainHint, list[str]] = {
    DomainHint.FINANCIAL: [
        "revenue", "balance sheet", "income statement", "profit", "loss",
        "assets", "liabilities", "equity", "fiscal", "budget", "expenditure",
        "audit", "financial statements", "birr", "usd", "eur", "cash flow",
        "earnings", "dividends", "tax", "quarter", "annual report",
    ],
    DomainHint.LEGAL: [
        "whereas", "hereinafter", "pursuant", "jurisdiction", "plaintiff",
        "defendant", "clause", "article", "regulation", "act", "law",
        "court", "tribunal", "legal", "contract", "agreement", "compliance",
    ],
    DomainHint.TECHNICAL: [
        "algorithm", "architecture", "implementation", "system", "software",
        "hardware", "protocol", "specification", "api", "database", "network",
        "assessment", "technical", "methodology", "framework", "performance",
        "pharmaceutical", "manufacturing", "chemical", "process", "standard",
    ],
    DomainHint.MEDICAL: [
        "patient", "clinical", "diagnosis", "treatment", "dosage", "medical",
        "hospital", "health", "disease", "therapy", "pharmaceutical",
        "epidemiology", "mortality", "morbidity",
    ],
}


def _detect_domain_hint(pdf: pdfplumber.PDF, page_stats: list[PageStats]) -> DomainHint:
    """Keyword-frequency domain classifier.

    Samples text from first 5 non-scanned pages and scores against domain keyword lists.
    Falls back to GENERAL if no domain scores above threshold.
    """
    sample_text = ""
    scanned_flags = {p.page_number: p.is_likely_scanned for p in page_stats}

    for i, page in enumerate(pdf.pages[:10], start=1):
        if scanned_flags.get(i, False):
            continue
        text = page.extract_text() or ""
        sample_text += text.lower() + "\n"
        if len(sample_text) > 5000:
            break

    if not sample_text.strip():
        return DomainHint.GENERAL

    scores: dict[DomainHint, int] = {d: 0 for d in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            scores[domain] += sample_text.count(kw)

    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        return DomainHint.GENERAL
    return best


# ---------------------------------------------------------------------------
# Extraction cost estimation
# ---------------------------------------------------------------------------

def _estimate_cost(
    origin_type: OriginType,
    layout_complexity: LayoutComplexity,
) -> ExtractionCost:
    """Map classification dimensions to extraction cost tier."""
    if origin_type == OriginType.SCANNED_IMAGE:
        return ExtractionCost.NEEDS_VISION_MODEL

    if layout_complexity in (
        LayoutComplexity.MULTI_COLUMN,
        LayoutComplexity.TABLE_HEAVY,
        LayoutComplexity.FIGURE_HEAVY,
        LayoutComplexity.MIXED,
    ):
        return ExtractionCost.NEEDS_LAYOUT_MODEL

    if origin_type == OriginType.MIXED:
        return ExtractionCost.NEEDS_LAYOUT_MODEL

    return ExtractionCost.FAST_TEXT_SUFFICIENT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TriageAgent:
    """Stage 1 agent — classifies a document and produces a DocumentProfile."""

    def __init__(self, rules_path: Optional[str] = None):
        self.thresholds = _load_thresholds(rules_path)

    def run(self, file_path: str) -> DocumentProfile:
        """Analyze a PDF and return its DocumentProfile.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            DocumentProfile ready to save to .refinery/profiles/{doc_id}.json
        """
        start = time.perf_counter()
        path = Path(file_path)

        doc_id = _doc_id(file_path)
        file_size = path.stat().st_size

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)

            # Analyze each page
            page_stats = [
                _analyze_page(page, i + 1)
                for i, page in enumerate(pdf.pages)
            ]

            # Classification
            origin_type = _detect_origin_type(page_stats, self.thresholds)
            layout_complexity, col_count = _detect_layout_complexity(
                pdf, page_stats, self.thresholds
            )
            domain_hint = _detect_domain_hint(pdf, page_stats)
            extraction_cost = _estimate_cost(origin_type, layout_complexity)

        # Aggregate stats
        avg_char_density = (
            sum(p.char_density for p in page_stats) / len(page_stats)
            if page_stats else 0.0
        )
        avg_image_area_ratio = (
            sum(p.image_area_ratio for p in page_stats) / len(page_stats)
            if page_stats else 0.0
        )
        scanned_count = sum(1 for p in page_stats if p.is_likely_scanned)
        table_count = sum(1 for p in page_stats if p.has_tables)

        elapsed = time.perf_counter() - start

        return DocumentProfile(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path.resolve()),
            page_count=page_count,
            file_size_bytes=file_size,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            domain_hint=domain_hint,
            estimated_extraction_cost=extraction_cost,
            avg_char_density=round(avg_char_density, 6),
            avg_image_area_ratio=round(avg_image_area_ratio, 4),
            scanned_page_count=scanned_count,
            table_page_count=table_count,
            column_count_estimate=col_count,
            page_stats=page_stats,
            processing_time_seconds=round(elapsed, 3),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.agents.triage <path/to/document.pdf>[/red]")
        sys.exit(1)

    agent = TriageAgent()
    profile = agent.run(sys.argv[1])
    saved = profile.save()

    table = Table(title=f"DocumentProfile — {profile.filename}", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("doc_id", profile.doc_id)
    table.add_row("pages", str(profile.page_count))
    table.add_row("origin_type", profile.origin_type.value)
    table.add_row("layout_complexity", profile.layout_complexity.value)
    table.add_row("domain_hint", profile.domain_hint.value)
    table.add_row("extraction_cost", profile.estimated_extraction_cost.value)
    table.add_row("avg_char_density", f"{profile.avg_char_density:.6f} chars/pt²")
    table.add_row("avg_image_area_ratio", f"{profile.avg_image_area_ratio:.2%}")
    table.add_row("scanned_pages", f"{profile.scanned_page_count}/{profile.page_count}")
    table.add_row("table_pages", f"{profile.table_page_count}/{profile.page_count}")
    table.add_row("est_columns", str(profile.column_count_estimate))
    table.add_row("triage_time", f"{profile.processing_time_seconds}s")
    table.add_row("saved_to", str(saved))

    console.print(table)
