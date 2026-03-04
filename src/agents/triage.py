"""Stage 1: Triage Agent — Document Classifier.

Produces a DocumentProfile that governs all downstream extraction strategy decisions.
Uses pdfplumber for character density analysis, image area measurement, and
bounding box layout analysis. No ML model required — pure heuristics.

Classification dimensions:
  - origin_type: native_digital | scanned_image | mixed | form_fillable | zero_text
  - layout_complexity: single_column | multi_column | table_heavy | figure_heavy | mixed
  - domain_hint: financial | legal | technical | medical | general
  - estimated_extraction_cost: fast_text_sufficient | needs_layout_model | needs_vision_model

Extensibility:
  - Domain classification is fully pluggable via DomainStrategy subclasses.
  - Keywords and weights are loaded from rubric/extraction_rules.yaml — adding a new
    domain requires only a YAML edit; no code changes needed.
  - All numeric thresholds are read from the same config file.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
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
# Default thresholds — used when extraction_rules.yaml is absent or incomplete
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict = {
    "char_density_min": 0.05,
    "char_density_scanned": 0.001,
    "image_area_scanned": 0.80,
    "image_area_max": 0.50,
    "scanned_page_ratio": 0.40,
    "scanned_page_ratio_hard": 0.80,
    "zero_text_page_ratio": 0.60,
    "form_fillable_min_fields": 1,
    "table_page_ratio": 0.30,
    "figure_page_ratio": 0.30,
    "multi_column_gap_min": 50.0,
    "multi_column_count": 2,
}

# Default domain keywords — used when the domains section is absent from config
_DEFAULT_DOMAIN_KEYWORDS: dict[DomainHint, list[str]] = {
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


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(rules_path: Optional[str] = None) -> dict:
    """Load full extraction_rules.yaml. Returns empty dict on missing file."""
    path = Path(rules_path or "rubric/extraction_rules.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_thresholds(config: dict) -> dict:
    """Merge triage section from config over defaults."""
    return {**DEFAULT_THRESHOLDS, **config.get("triage", {})}


# ---------------------------------------------------------------------------
# Pluggable domain classification
# ---------------------------------------------------------------------------

class DomainStrategy(ABC):
    """Abstract base for a single-domain classification strategy.

    Subclass to add non-keyword signals (e.g. regex patterns, font analysis).
    Register instances in _build_domain_classifier().
    """

    @property
    @abstractmethod
    def domain(self) -> DomainHint:
        """The domain this strategy classifies for."""
        ...

    @abstractmethod
    def score(self, text: str) -> float:
        """Return a non-negative relevance score for the given text."""
        ...


class KeywordDomainStrategy(DomainStrategy):
    """Score a domain by weighted keyword frequency in sampled text.

    Keywords are matched as lowercase substrings. Multi-word phrases count as one hit.
    A weight > 1.0 amplifies the score for domains with distinctive but rare vocabulary.
    """

    def __init__(self, domain: DomainHint, keywords: list[str], weight: float = 1.0):
        self._domain = domain
        self.keywords = [kw.lower() for kw in keywords]
        self.weight = weight

    @property
    def domain(self) -> DomainHint:
        return self._domain

    def score(self, text: str) -> float:
        return sum(text.count(kw) for kw in self.keywords) * self.weight


class DomainClassifier:
    """Orchestrates multiple DomainStrategy instances to classify domain and confidence.

    Confidence is the fraction of total keyword hits belonging to the winning domain.
    A score of 0.0 means no domain keywords were found (defaults to GENERAL).
    """

    def __init__(self, strategies: list[DomainStrategy]):
        self.strategies = strategies

    def classify(self, text: str) -> tuple[DomainHint, float]:
        """Return (domain, confidence) for the given text sample."""
        if not text.strip():
            return DomainHint.GENERAL, 0.0

        scores: dict[DomainHint, float] = {}
        for s in self.strategies:
            scores[s.domain] = s.score(text)

        total = sum(scores.values())
        if total == 0:
            return DomainHint.GENERAL, 0.0

        best = max(scores, key=lambda d: scores[d])
        # Confidence = exclusivity: how much of the total signal belongs to best domain
        confidence = round(scores[best] / total, 3)
        return best, confidence


def _build_domain_classifier(config: dict) -> DomainClassifier:
    """Construct DomainClassifier from config, falling back to defaults per domain.

    Config schema (domains section of extraction_rules.yaml):
        domains:
          financial:
            weight: 1.0
            keywords: [revenue, profit, ...]
    """
    domain_cfg = config.get("domains", {})
    strategies: list[DomainStrategy] = []

    for hint in DomainHint:
        if hint == DomainHint.GENERAL:
            continue  # GENERAL is the fallback — not a scored strategy

        name = hint.value
        if name in domain_cfg:
            entry = domain_cfg[name]
            keywords = entry.get("keywords", [])
            weight = float(entry.get("weight", 1.0))
        else:
            keywords = _DEFAULT_DOMAIN_KEYWORDS.get(hint, [])
            weight = 1.0

        if keywords:
            strategies.append(KeywordDomainStrategy(hint, keywords, weight))

    return DomainClassifier(strategies)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _doc_id(file_path: str) -> str:
    """First 16 chars of SHA256 of file content."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-page analysis
# ---------------------------------------------------------------------------

def _analyze_page(
    page: pdfplumber.page.Page,
    page_num: int,
    thresholds: dict,
) -> PageStats:
    """Compute extraction-relevant statistics for a single PDF page."""
    page_area = page.width * page.height  # in pt²

    chars = page.chars
    char_count = len(chars)
    char_density = char_count / page_area if page_area > 0 else 0.0

    images = page.images
    image_area = sum(
        (img.get("width", 0) * img.get("height", 0)) for img in images
    )
    image_area_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0.0

    try:
        tables = page.find_tables()
        has_tables = len(tables) > 0
    except Exception:
        has_tables = False

    has_figures = any(
        (img.get("width", 0) * img.get("height", 0)) > 10_000
        for img in images
    )

    is_likely_scanned = (
        char_density < thresholds["char_density_scanned"]
        or image_area_ratio > thresholds["image_area_scanned"]
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
# Form-fillable detection
# ---------------------------------------------------------------------------

def _detect_form_fillable(pdf: pdfplumber.PDF, thresholds: dict) -> tuple[bool, int, float]:
    """Detect if PDF has interactive AcroForm fields.

    Returns:
        (is_form_fillable, field_count, confidence)
        confidence scales with field count — more fields = more certain.
    """
    min_fields = int(thresholds.get("form_fillable_min_fields", 1))
    try:
        catalog = pdf.doc.catalog
        if "AcroForm" not in catalog:
            return False, 0, 0.0

        from pdfminer.pdftypes import resolve1  # pdfminer is pdfplumber's backend

        acroform = resolve1(catalog["AcroForm"])
        if not isinstance(acroform, dict):
            return False, 0, 0.0

        raw_fields = acroform.get("Fields", [])
        field_list = resolve1(raw_fields) if raw_fields else []
        field_count = len(field_list) if isinstance(field_list, list) else 0

        if field_count < min_fields:
            return False, field_count, 0.0

        # Confidence scales with field count (more fields → more certain)
        confidence = round(min(0.60 + field_count * 0.02, 1.0), 3)
        return True, field_count, confidence

    except Exception:
        return False, 0, 0.0


# ---------------------------------------------------------------------------
# Origin type detection
# ---------------------------------------------------------------------------

def _detect_origin_type(
    page_stats: list[PageStats],
    thresholds: dict,
) -> tuple[OriginType, float]:
    """Classify document origin. Returns (origin_type, confidence).

    Confidence is higher when the measured ratio is far from decision thresholds.
    Order of precedence: ZERO_TEXT > SCANNED_IMAGE > MIXED > NATIVE_DIGITAL.
    Form-fillable is detected separately (requires PDF catalog access).
    """
    total = len(page_stats)
    if total == 0:
        return OriginType.SCANNED_IMAGE, 1.0

    # Zero-text pages: no chars AND low image area (truly blank/decorative, not scanned).
    # Pages with no text but high image area are classified as scanned — they may be OCR-able.
    zero_text_count = sum(
        1 for p in page_stats
        if p.char_count == 0 and p.image_area_ratio < thresholds.get("image_area_scanned", 0.80)
    )
    scanned_count = sum(1 for p in page_stats if p.is_likely_scanned)
    zero_text_ratio = zero_text_count / total
    scanned_ratio = scanned_count / total

    zt_threshold = thresholds.get("zero_text_page_ratio", 0.60)
    hard = thresholds["scanned_page_ratio_hard"]
    soft = thresholds["scanned_page_ratio"]

    # --- Zero-text: no chars AND no significant images (blank/graphical decoration) ---
    if zero_text_ratio >= zt_threshold:
        conf = round(min(0.50 + (zero_text_ratio - zt_threshold) * 2.0, 1.0), 3)
        return OriginType.ZERO_TEXT, conf

    # --- Fully scanned ---
    if scanned_ratio >= hard:
        conf = round(min(0.60 + (scanned_ratio - hard) * 2.5, 1.0), 3)
        return OriginType.SCANNED_IMAGE, conf

    # --- Mixed ---
    if scanned_ratio >= soft:
        mid = (soft + hard) / 2
        distance_from_mid = abs(scanned_ratio - mid)
        half_span = (hard - soft) / 2
        conf = round(0.50 + (distance_from_mid / half_span) * 0.25, 3)
        return OriginType.MIXED, min(conf, 1.0)

    # --- Native digital ---
    conf = round(min(0.60 + (soft - scanned_ratio) * 1.5, 1.0), 3)
    return OriginType.NATIVE_DIGITAL, conf


# ---------------------------------------------------------------------------
# Layout complexity detection
# ---------------------------------------------------------------------------

def _estimate_column_count(page: pdfplumber.page.Page, thresholds: dict) -> int:
    """Estimate number of text columns via x-coordinate gap analysis."""
    chars = page.chars
    if not chars:
        return 1

    x0_positions = sorted(set(round(c["x0"]) for c in chars if c["text"].strip()))
    if len(x0_positions) < 4:
        return 1

    gap_min = thresholds.get("multi_column_gap_min", 50.0)
    gaps = [
        x0_positions[i] - x0_positions[i - 1]
        for i in range(1, len(x0_positions))
        if x0_positions[i] - x0_positions[i - 1] > gap_min
    ]
    return min(len(gaps) + 1, 4)


def _detect_layout_complexity(
    pdf: pdfplumber.PDF,
    page_stats: list[PageStats],
    thresholds: dict,
) -> tuple[LayoutComplexity, int, float]:
    """Classify layout complexity. Returns (complexity, estimated_col_count, confidence)."""
    total = len(page_stats)
    if total == 0:
        return LayoutComplexity.SINGLE_COLUMN, 1, 1.0

    table_count = sum(1 for p in page_stats if p.has_tables)
    figure_count = sum(1 for p in page_stats if p.has_figures)
    table_ratio = table_count / total
    figure_ratio = figure_count / total

    sample_pages = [
        pdf.pages[p.page_number - 1]
        for p in page_stats
        if not p.is_likely_scanned
    ][:5]

    col_counts = [_estimate_column_count(p, thresholds) for p in sample_pages]
    avg_cols = sum(col_counts) / len(col_counts) if col_counts else 1.0
    estimated_cols = round(avg_cols)

    table_thresh = thresholds.get("table_page_ratio", 0.30)
    figure_thresh = thresholds.get("figure_page_ratio", 0.30)
    col_thresh = int(thresholds.get("multi_column_count", 2))

    is_table_heavy = table_ratio >= table_thresh
    is_figure_heavy = figure_ratio >= figure_thresh
    has_multi_column = estimated_cols >= col_thresh

    flags = sum([is_table_heavy, is_figure_heavy, has_multi_column])

    # Confidence: strength of the dominant signal relative to its threshold
    dominant_signal = max(
        table_ratio / table_thresh if table_thresh > 0 else 0,
        figure_ratio / figure_thresh if figure_thresh > 0 else 0,
        avg_cols / col_thresh if col_thresh > 0 else 0,
    )
    confidence = round(min(dominant_signal * 0.60, 1.0), 3)

    if flags >= 2:
        return LayoutComplexity.MIXED, estimated_cols, confidence
    elif is_table_heavy:
        return LayoutComplexity.TABLE_HEAVY, estimated_cols, confidence
    elif is_figure_heavy:
        return LayoutComplexity.FIGURE_HEAVY, estimated_cols, confidence
    elif has_multi_column:
        return LayoutComplexity.MULTI_COLUMN, estimated_cols, confidence
    else:
        # Single column — confidence based on how far below multi-column threshold
        conf_single = round(min(1.0, 1.0 - (avg_cols - 1) / max(col_thresh - 1, 1)), 3)
        return LayoutComplexity.SINGLE_COLUMN, 1, max(conf_single, 0.5)


# ---------------------------------------------------------------------------
# Domain detection (text sampling)
# ---------------------------------------------------------------------------

def _sample_text(pdf: pdfplumber.PDF, page_stats: list[PageStats], max_chars: int = 5000) -> str:
    """Sample up to max_chars of text from the first 10 non-scanned pages."""
    scanned = {p.page_number: p.is_likely_scanned for p in page_stats}
    text = ""
    for i, page in enumerate(pdf.pages[:10], start=1):
        if scanned.get(i, False):
            continue
        text += (page.extract_text() or "").lower() + "\n"
        if len(text) >= max_chars:
            break
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Extraction cost estimation
# ---------------------------------------------------------------------------

def _estimate_cost(origin_type: OriginType, layout_complexity: LayoutComplexity) -> ExtractionCost:
    if origin_type in (OriginType.SCANNED_IMAGE, OriginType.ZERO_TEXT):
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

    if origin_type == OriginType.FORM_FILLABLE:
        return ExtractionCost.NEEDS_LAYOUT_MODEL  # Form fields need structured parsing

    return ExtractionCost.FAST_TEXT_SUFFICIENT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TriageAgent:
    """Stage 1 agent — classifies a document and produces a DocumentProfile.

    All numeric thresholds and domain keywords are read from
    rubric/extraction_rules.yaml at construction time. Pass rules_path to
    override the config location (useful in tests).

    Domain detection is pluggable: add a DomainStrategy subclass and register
    it in _build_domain_classifier(), or simply add keywords to the YAML config.
    """

    def __init__(self, rules_path: Optional[str] = None):
        self._config = _load_config(rules_path)
        self.thresholds = _load_thresholds(self._config)
        self._domain_classifier = _build_domain_classifier(self._config)

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

            # 1. Per-page stats (uses loaded thresholds — no hardcoded defaults)
            page_stats = [
                _analyze_page(page, i + 1, self.thresholds)
                for i, page in enumerate(pdf.pages)
            ]

            # 2. Form-fillable detection (requires PDF catalog access)
            is_form_fillable, form_field_count, form_confidence = _detect_form_fillable(
                pdf, self.thresholds
            )

            # 3. Origin classification
            origin_type, origin_confidence = _detect_origin_type(page_stats, self.thresholds)

            # Promote to FORM_FILLABLE if AcroForm fields found
            if is_form_fillable and origin_type == OriginType.NATIVE_DIGITAL:
                origin_type = OriginType.FORM_FILLABLE
                origin_confidence = form_confidence

            # 4. Layout complexity
            layout_complexity, col_count, layout_confidence = _detect_layout_complexity(
                pdf, page_stats, self.thresholds
            )

            # 5. Domain classification (pluggable, config-driven)
            sample_text = _sample_text(pdf, page_stats)
            domain_hint, domain_confidence = self._domain_classifier.classify(sample_text)

        # 6. Extraction cost estimate
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
        zero_text_count = sum(
            1 for p in page_stats
            if p.char_count == 0 and p.image_area_ratio < self.thresholds.get("image_area_scanned", 0.80)
        )
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
            # Classification confidences
            origin_confidence=origin_confidence,
            layout_confidence=layout_confidence,
            domain_confidence=domain_confidence,
            # Form-fillable evidence
            is_form_fillable=is_form_fillable,
            form_field_count=form_field_count,
            # Aggregate evidence
            avg_char_density=round(avg_char_density, 6),
            avg_image_area_ratio=round(avg_image_area_ratio, 4),
            scanned_page_count=scanned_count,
            zero_text_page_count=zero_text_count,
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
    table.add_row("origin_type", f"{profile.origin_type.value}  (conf={profile.origin_confidence:.2f})")
    table.add_row("layout_complexity", f"{profile.layout_complexity.value}  (conf={profile.layout_confidence:.2f})")
    table.add_row("domain_hint", f"{profile.domain_hint.value}  (conf={profile.domain_confidence:.2f})")
    table.add_row("extraction_cost", profile.estimated_extraction_cost.value)
    table.add_row("form_fillable", f"{profile.is_form_fillable}  ({profile.form_field_count} fields)")
    table.add_row("avg_char_density", f"{profile.avg_char_density:.6f} chars/pt²")
    table.add_row("avg_image_area_ratio", f"{profile.avg_image_area_ratio:.2%}")
    table.add_row("scanned_pages", f"{profile.scanned_page_count}/{profile.page_count}")
    table.add_row("zero_text_pages", f"{profile.zero_text_page_count}/{profile.page_count}")
    table.add_row("table_pages", f"{profile.table_page_count}/{profile.page_count}")
    table.add_row("est_columns", str(profile.column_count_estimate))
    table.add_row("triage_time", f"{profile.processing_time_seconds}s")
    table.add_row("saved_to", str(saved))

    console.print(table)
