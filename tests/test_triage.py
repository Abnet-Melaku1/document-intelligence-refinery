"""Unit tests for the TriageAgent classification logic.

Tests verify that:
1. origin_type is correctly classified from char density and image area signals
2. layout_complexity detection works for single vs multi-column
3. domain_hint classifier scores financial keywords above threshold
4. extraction cost is mapped correctly from classification dimensions
5. DocumentProfile can round-trip through JSON serialization

Tests use only pdfplumber's Page interface, mocked via simple objects — no actual PDFs needed.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LayoutComplexity,
    OriginType,
    PageStats,
)
from src.agents.triage import (
    _detect_origin_type,
    _detect_layout_complexity,
    _detect_domain_hint,
    _estimate_cost,
    DEFAULT_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helper: build PageStats objects for testing
# ---------------------------------------------------------------------------

def make_page_stats(
    char_density: float,
    image_area_ratio: float,
    has_tables: bool = False,
    has_figures: bool = False,
    page_number: int = 1,
) -> PageStats:
    """Create a PageStats for a single page with given signals."""
    t = DEFAULT_THRESHOLDS
    is_scanned = (
        char_density < t["char_density_scanned"]
        or image_area_ratio > t["image_area_scanned"]
    )
    return PageStats(
        page_number=page_number,
        char_count=int(char_density * 10000),
        char_density=char_density,
        image_area_ratio=image_area_ratio,
        has_tables=has_tables,
        has_figures=has_figures,
        is_likely_scanned=is_scanned,
    )


# ---------------------------------------------------------------------------
# origin_type detection tests
# ---------------------------------------------------------------------------

class TestOriginTypeDetection:

    def test_native_digital_high_density(self):
        """All pages with high char density → NATIVE_DIGITAL."""
        pages = [make_page_stats(char_density=1.2, image_area_ratio=0.1) for _ in range(10)]
        result = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert result == OriginType.NATIVE_DIGITAL

    def test_scanned_zero_chars(self):
        """All pages with zero chars and near-full image area → SCANNED_IMAGE."""
        pages = [make_page_stats(char_density=0.0, image_area_ratio=0.99) for _ in range(10)]
        result = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert result == OriginType.SCANNED_IMAGE

    def test_mixed_partial_scanned(self):
        """~50% scanned pages → MIXED."""
        digital_pages = [make_page_stats(1.0, 0.05, page_number=i) for i in range(1, 6)]
        scanned_pages = [make_page_stats(0.0, 0.95, page_number=i) for i in range(6, 11)]
        pages = digital_pages + scanned_pages
        result = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert result == OriginType.MIXED

    def test_mostly_digital_small_scanned_minority(self):
        """< 40% scanned → still NATIVE_DIGITAL (below scanned_page_ratio threshold)."""
        digital_pages = [make_page_stats(1.0, 0.05, page_number=i) for i in range(1, 8)]
        scanned_pages = [make_page_stats(0.0, 0.99, page_number=i) for i in range(8, 11)]
        pages = digital_pages + scanned_pages  # 30% scanned
        result = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert result == OriginType.NATIVE_DIGITAL

    def test_empty_pages_defaults_to_scanned(self):
        """Empty page list returns SCANNED_IMAGE (safe fallback)."""
        result = _detect_origin_type([], DEFAULT_THRESHOLDS)
        assert result == OriginType.SCANNED_IMAGE

    def test_borderline_scanned_threshold(self):
        """Pages right at the char_density_scanned threshold → scanned."""
        pages = [
            make_page_stats(char_density=0.0009, image_area_ratio=0.1)  # Below 0.001
            for _ in range(10)
        ]
        result = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert result == OriginType.SCANNED_IMAGE


# ---------------------------------------------------------------------------
# layout_complexity detection tests
# ---------------------------------------------------------------------------

class TestLayoutComplexityDetection:

    def _make_mock_pdf(self, pages_chars: list, pages_tables: list) -> MagicMock:
        """Build a mock pdfplumber PDF with fake pages."""
        mock_pdf = MagicMock()
        mock_pages = []
        for chars, has_table in zip(pages_chars, pages_tables):
            p = MagicMock()
            p.chars = chars
            p.width = 612.0
            p.height = 792.0
            p.find_tables.return_value = [MagicMock()] if has_table else []
            mock_pages.append(p)
        mock_pdf.pages = mock_pages
        return mock_pdf

    def test_table_heavy_classification(self):
        """Majority table pages → TABLE_HEAVY."""
        pages = [make_page_stats(0.8, 0.05, has_tables=True) for _ in range(7)]
        pages += [make_page_stats(0.8, 0.05, has_tables=False) for _ in range(3)]

        mock_chars = [{"x0": 50.0, "text": "A", "fontname": "Arial"}] * 100
        mock_pdf = self._make_mock_pdf(
            [mock_chars] * 10,
            [True] * 7 + [False] * 3,
        )

        with patch("src.agents.triage._estimate_column_count", return_value=1):
            result, cols = _detect_layout_complexity(mock_pdf, pages, DEFAULT_THRESHOLDS)

        assert result == LayoutComplexity.TABLE_HEAVY

    def test_single_column_clean_doc(self):
        """Single column, no tables, no figures → SINGLE_COLUMN."""
        pages = [make_page_stats(1.2, 0.05, has_tables=False) for _ in range(10)]
        mock_chars = [{"x0": 72.0, "text": "A", "fontname": "Arial"}] * 100
        mock_pdf = self._make_mock_pdf([mock_chars] * 10, [False] * 10)

        with patch("src.agents.triage._estimate_column_count", return_value=1):
            result, cols = _detect_layout_complexity(mock_pdf, pages, DEFAULT_THRESHOLDS)

        assert result == LayoutComplexity.SINGLE_COLUMN
        assert cols == 1

    def test_multi_column_detected(self):
        """Column count ≥ 2 → MULTI_COLUMN."""
        pages = [make_page_stats(0.9, 0.05) for _ in range(10)]
        mock_chars = [{"x0": 50.0, "text": "A", "fontname": "Arial"}] * 100
        mock_pdf = self._make_mock_pdf([mock_chars] * 10, [False] * 10)

        with patch("src.agents.triage._estimate_column_count", return_value=2):
            result, cols = _detect_layout_complexity(mock_pdf, pages, DEFAULT_THRESHOLDS)

        assert result == LayoutComplexity.MULTI_COLUMN
        assert cols == 2


# ---------------------------------------------------------------------------
# domain_hint classifier tests
# ---------------------------------------------------------------------------

class TestDomainHintDetection:

    def _make_mock_pdf_with_text(self, text: str, page_count: int = 3) -> MagicMock:
        """Build a mock PDF where each page returns the given text."""
        mock_pdf = MagicMock()
        mock_pages = []
        for _ in range(page_count):
            p = MagicMock()
            p.extract_text.return_value = text
            mock_pages.append(p)
        mock_pdf.pages = mock_pages
        return mock_pdf

    def test_financial_keywords_detected(self):
        """Text with financial keywords → FINANCIAL."""
        text = """
        Annual Report FY 2023-24. Revenue increased to $4.2 billion.
        Balance sheet assets total 12.8 billion birr. Fiscal year ended June 30.
        Income statement shows net profit of 840 million. Audit completed.
        Dividends declared at 15% of equity. Cash flow from operations: 2.1B.
        """
        pages = [make_page_stats(1.0, 0.05) for _ in range(3)]
        mock_pdf = self._make_mock_pdf_with_text(text)
        result = _detect_domain_hint(mock_pdf, pages)
        assert result == DomainHint.FINANCIAL

    def test_technical_keywords_detected(self):
        """Text with technical keywords → TECHNICAL."""
        text = """
        Assessment report for the implementation framework. Methodology based on
        technical specifications. System architecture review. API documentation.
        Performance benchmarks. Database schema. Protocol compliance standard.
        Implementation assessment findings. Technical recommendations.
        """
        pages = [make_page_stats(1.0, 0.05) for _ in range(3)]
        mock_pdf = self._make_mock_pdf_with_text(text)
        result = _detect_domain_hint(mock_pdf, pages)
        assert result == DomainHint.TECHNICAL

    def test_empty_text_returns_general(self):
        """No extractable text (scanned) → GENERAL fallback."""
        pages = [make_page_stats(0.0, 0.95, page_number=i) for i in range(1, 4)]
        mock_pdf = self._make_mock_pdf_with_text("")
        result = _detect_domain_hint(mock_pdf, pages)
        assert result == DomainHint.GENERAL

    def test_ambiguous_text_returns_non_general(self):
        """Text with moderate keywords still picks the highest-scoring domain."""
        text = "balance sheet revenue assets liabilities equity audit fiscal"
        pages = [make_page_stats(1.0, 0.05) for _ in range(3)]
        mock_pdf = self._make_mock_pdf_with_text(text)
        result = _detect_domain_hint(mock_pdf, pages)
        assert result == DomainHint.FINANCIAL


# ---------------------------------------------------------------------------
# Extraction cost estimation tests
# ---------------------------------------------------------------------------

class TestExtractionCostEstimation:

    def test_scanned_always_vision(self):
        """Scanned documents always need vision model regardless of layout."""
        for layout in LayoutComplexity:
            result = _estimate_cost(OriginType.SCANNED_IMAGE, layout)
            assert result == ExtractionCost.NEEDS_VISION_MODEL

    def test_native_single_column_is_fast(self):
        """Native digital + single column → fast text sufficient."""
        result = _estimate_cost(OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN)
        assert result == ExtractionCost.FAST_TEXT_SUFFICIENT

    def test_native_multi_column_needs_layout(self):
        """Native digital + multi-column → layout model."""
        result = _estimate_cost(OriginType.NATIVE_DIGITAL, LayoutComplexity.MULTI_COLUMN)
        assert result == ExtractionCost.NEEDS_LAYOUT_MODEL

    def test_native_table_heavy_needs_layout(self):
        """Native digital + table-heavy → layout model."""
        result = _estimate_cost(OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY)
        assert result == ExtractionCost.NEEDS_LAYOUT_MODEL

    def test_mixed_origin_needs_layout(self):
        """Mixed origin with single column → still needs layout (safe escalation)."""
        result = _estimate_cost(OriginType.MIXED, LayoutComplexity.SINGLE_COLUMN)
        assert result == ExtractionCost.NEEDS_LAYOUT_MODEL


# ---------------------------------------------------------------------------
# DocumentProfile serialization test
# ---------------------------------------------------------------------------

class TestDocumentProfileSerialization:

    def test_round_trip_json(self):
        """DocumentProfile serializes to JSON and back without data loss."""
        profile = DocumentProfile(
            doc_id="abc123def456789a",
            filename="test.pdf",
            file_path="/data/test.pdf",
            page_count=10,
            file_size_bytes=1024000,
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            domain_hint=DomainHint.FINANCIAL,
            estimated_extraction_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
            avg_char_density=1.24,
            avg_image_area_ratio=0.05,
            scanned_page_count=0,
            table_page_count=3,
            column_count_estimate=1,
        )

        json_str = profile.model_dump_json()
        restored = DocumentProfile.model_validate_json(json_str)

        assert restored.doc_id == profile.doc_id
        assert restored.origin_type == profile.origin_type
        assert restored.layout_complexity == profile.layout_complexity
        assert restored.domain_hint == profile.domain_hint
        assert restored.estimated_extraction_cost == profile.estimated_extraction_cost
        assert restored.avg_char_density == profile.avg_char_density
