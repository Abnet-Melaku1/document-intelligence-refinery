"""Unit tests for the TriageAgent classification logic.

Tests verify that:
1. origin_type is correctly classified from char density and image area signals
2. zero_text and form_fillable origin types are detected
3. layout_complexity detection works for single vs multi-column
4. DomainClassifier scores financial keywords above other domains
5. Confidence scores are emitted for each classification dimension
6. extraction cost is mapped correctly from classification dimensions
7. DocumentProfile can round-trip through JSON serialization

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
    DEFAULT_THRESHOLDS,
    DomainClassifier,
    KeywordDomainStrategy,
    _build_domain_classifier,
    _detect_layout_complexity,
    _detect_origin_type,
    _estimate_cost,
    _sample_text,
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
    char_count: int = -1,
) -> PageStats:
    """Create a PageStats for a single page with given signals."""
    t = DEFAULT_THRESHOLDS
    is_scanned = (
        char_density < t["char_density_scanned"]
        or image_area_ratio > t["image_area_scanned"]
    )
    computed_char_count = int(char_density * 10000) if char_count < 0 else char_count
    return PageStats(
        page_number=page_number,
        char_count=computed_char_count,
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
        origin, conf = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert origin == OriginType.NATIVE_DIGITAL
        assert conf >= 0.6

    def test_scanned_zero_chars(self):
        """All pages with zero chars and near-full image area → SCANNED_IMAGE."""
        pages = [make_page_stats(char_density=0.0, image_area_ratio=0.99) for _ in range(10)]
        origin, conf = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert origin == OriginType.SCANNED_IMAGE
        assert conf >= 0.6

    def test_mixed_partial_scanned(self):
        """~50% scanned pages → MIXED."""
        digital = [make_page_stats(1.0, 0.05, page_number=i) for i in range(1, 6)]
        scanned = [make_page_stats(0.0, 0.95, page_number=i) for i in range(6, 11)]
        origin, conf = _detect_origin_type(digital + scanned, DEFAULT_THRESHOLDS)
        assert origin == OriginType.MIXED

    def test_mostly_digital_small_scanned_minority(self):
        """< 40% scanned → still NATIVE_DIGITAL (below scanned_page_ratio threshold)."""
        digital = [make_page_stats(1.0, 0.05, page_number=i) for i in range(1, 8)]
        scanned = [make_page_stats(0.0, 0.99, page_number=i) for i in range(8, 11)]
        origin, conf = _detect_origin_type(digital + scanned, DEFAULT_THRESHOLDS)
        assert origin == OriginType.NATIVE_DIGITAL

    def test_empty_pages_defaults_to_scanned(self):
        """Empty page list returns SCANNED_IMAGE (safe fallback)."""
        origin, conf = _detect_origin_type([], DEFAULT_THRESHOLDS)
        assert origin == OriginType.SCANNED_IMAGE

    def test_borderline_scanned_threshold(self):
        """Pages right below char_density_scanned threshold → is_scanned=True → SCANNED_IMAGE."""
        pages = [
            make_page_stats(char_density=0.0009, image_area_ratio=0.1)
            for _ in range(10)
        ]
        origin, conf = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert origin == OriginType.SCANNED_IMAGE

    def test_zero_text_origin(self):
        """≥60% of pages with char_count == 0 → ZERO_TEXT."""
        zero = [make_page_stats(0.0, 0.05, char_count=0, page_number=i) for i in range(1, 8)]
        digital = [make_page_stats(1.0, 0.05, page_number=i) for i in range(8, 11)]
        origin, conf = _detect_origin_type(zero + digital, DEFAULT_THRESHOLDS)
        assert origin == OriginType.ZERO_TEXT
        assert conf >= 0.5

    def test_confidence_is_float_between_0_and_1(self):
        """Confidence score is always in [0.0, 1.0]."""
        pages = [make_page_stats(0.8, 0.1) for _ in range(5)]
        _, conf = _detect_origin_type(pages, DEFAULT_THRESHOLDS)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# layout_complexity detection tests
# ---------------------------------------------------------------------------

class TestLayoutComplexityDetection:

    def _make_mock_pdf(self, pages_chars: list, pages_tables: list) -> MagicMock:
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
            result, cols, conf = _detect_layout_complexity(mock_pdf, pages, DEFAULT_THRESHOLDS)

        assert result == LayoutComplexity.TABLE_HEAVY
        assert 0.0 <= conf <= 1.0

    def test_single_column_clean_doc(self):
        """Single column, no tables, no figures → SINGLE_COLUMN."""
        pages = [make_page_stats(1.2, 0.05, has_tables=False) for _ in range(10)]
        mock_chars = [{"x0": 72.0, "text": "A", "fontname": "Arial"}] * 100
        mock_pdf = self._make_mock_pdf([mock_chars] * 10, [False] * 10)

        with patch("src.agents.triage._estimate_column_count", return_value=1):
            result, cols, conf = _detect_layout_complexity(mock_pdf, pages, DEFAULT_THRESHOLDS)

        assert result == LayoutComplexity.SINGLE_COLUMN
        assert cols == 1

    def test_multi_column_detected(self):
        """Column count ≥ 2 → MULTI_COLUMN."""
        pages = [make_page_stats(0.9, 0.05) for _ in range(10)]
        mock_chars = [{"x0": 50.0, "text": "A", "fontname": "Arial"}] * 100
        mock_pdf = self._make_mock_pdf([mock_chars] * 10, [False] * 10)

        with patch("src.agents.triage._estimate_column_count", return_value=2):
            result, cols, conf = _detect_layout_complexity(mock_pdf, pages, DEFAULT_THRESHOLDS)

        assert result == LayoutComplexity.MULTI_COLUMN
        assert cols == 2


# ---------------------------------------------------------------------------
# DomainClassifier tests (replaces old _detect_domain_hint tests)
# ---------------------------------------------------------------------------

class TestDomainClassifier:

    def _make_classifier(self) -> DomainClassifier:
        """Build a DomainClassifier using default config (no file needed)."""
        return _build_domain_classifier({})

    def test_financial_keywords_detected(self):
        """Text with financial keywords → FINANCIAL with non-zero confidence."""
        text = """
        Annual Report FY 2023-24. Revenue increased to $4.2 billion.
        Balance sheet assets total 12.8 billion birr. Fiscal year ended June 30.
        Income statement shows net profit of 840 million. Audit completed.
        Dividends declared at 15% of equity. Cash flow from operations: 2.1B.
        """
        domain, conf = self._make_classifier().classify(text.lower())
        assert domain == DomainHint.FINANCIAL
        assert conf > 0.0

    def test_technical_keywords_detected(self):
        """Text with technical keywords → TECHNICAL."""
        text = """
        Assessment report for the implementation framework. Methodology based on
        technical specifications. System architecture review. API documentation.
        Performance benchmarks. Database schema. Protocol compliance standard.
        """
        domain, conf = self._make_classifier().classify(text.lower())
        assert domain == DomainHint.TECHNICAL
        assert conf > 0.0

    def test_empty_text_returns_general(self):
        """Empty text → GENERAL, confidence 0.0."""
        domain, conf = self._make_classifier().classify("")
        assert domain == DomainHint.GENERAL
        assert conf == 0.0

    def test_no_keyword_hits_returns_general(self):
        """Text with no domain keywords → GENERAL."""
        text = "the quick brown fox jumps over the lazy dog"
        domain, conf = self._make_classifier().classify(text)
        assert domain == DomainHint.GENERAL
        assert conf == 0.0

    def test_ambiguous_text_picks_highest_scoring(self):
        """Text with financial keywords wins over competing domains."""
        text = "balance sheet revenue assets liabilities equity audit fiscal"
        domain, conf = self._make_classifier().classify(text)
        assert domain == DomainHint.FINANCIAL

    def test_config_driven_keywords(self):
        """Keywords loaded from config override defaults for that domain."""
        config = {
            "domains": {
                "financial": {"keywords": ["uniquetoken_xyz"], "weight": 1.0},
                "legal": {"keywords": ["law", "court"], "weight": 1.0},
            }
        }
        classifier = _build_domain_classifier(config)
        domain, conf = classifier.classify("uniquetoken_xyz uniquetoken_xyz")
        assert domain == DomainHint.FINANCIAL

    def test_confidence_between_0_and_1(self):
        """Confidence is always in [0.0, 1.0]."""
        text = "revenue profit balance sheet assets equity fiscal audit tax"
        _, conf = self._make_classifier().classify(text)
        assert 0.0 <= conf <= 1.0

    def test_keyword_domain_strategy_score(self):
        """KeywordDomainStrategy.score() counts hits correctly."""
        strategy = KeywordDomainStrategy(DomainHint.FINANCIAL, ["profit", "loss"], weight=2.0)
        score = strategy.score("profit and loss statement shows a net profit")
        # "profit" appears 2 times, "loss" 1 time → (3 hits) * 2.0 weight = 6.0
        assert score == 6.0


# ---------------------------------------------------------------------------
# Extraction cost estimation tests
# ---------------------------------------------------------------------------

class TestExtractionCostEstimation:

    def test_scanned_always_vision(self):
        """Scanned documents always need vision model."""
        for layout in LayoutComplexity:
            result = _estimate_cost(OriginType.SCANNED_IMAGE, layout)
            assert result == ExtractionCost.NEEDS_VISION_MODEL

    def test_zero_text_always_vision(self):
        """Zero-text documents route to vision model."""
        for layout in LayoutComplexity:
            result = _estimate_cost(OriginType.ZERO_TEXT, layout)
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
        """Mixed origin with single column → still needs layout."""
        result = _estimate_cost(OriginType.MIXED, LayoutComplexity.SINGLE_COLUMN)
        assert result == ExtractionCost.NEEDS_LAYOUT_MODEL

    def test_form_fillable_needs_layout(self):
        """Form-fillable PDF → layout model (structured field parsing)."""
        result = _estimate_cost(OriginType.FORM_FILLABLE, LayoutComplexity.SINGLE_COLUMN)
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
            origin_confidence=0.92,
            layout_confidence=0.85,
            domain_confidence=0.67,
            is_form_fillable=False,
            form_field_count=0,
            zero_text_page_count=0,
        )

        json_str = profile.model_dump_json()
        restored = DocumentProfile.model_validate_json(json_str)

        assert restored.doc_id == profile.doc_id
        assert restored.origin_type == profile.origin_type
        assert restored.layout_complexity == profile.layout_complexity
        assert restored.domain_hint == profile.domain_hint
        assert restored.estimated_extraction_cost == profile.estimated_extraction_cost
        assert restored.avg_char_density == profile.avg_char_density
        assert restored.origin_confidence == profile.origin_confidence
        assert restored.domain_confidence == profile.domain_confidence
        assert restored.is_form_fillable == profile.is_form_fillable

    def test_zero_text_origin_serializes(self):
        """ZERO_TEXT origin type serializes and deserializes correctly."""
        profile = DocumentProfile(
            doc_id="zerotextdoc01234",
            filename="blank.pdf",
            file_path="/data/blank.pdf",
            page_count=5,
            file_size_bytes=4096,
            origin_type=OriginType.ZERO_TEXT,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            domain_hint=DomainHint.GENERAL,
            estimated_extraction_cost=ExtractionCost.NEEDS_VISION_MODEL,
            avg_char_density=0.0,
            avg_image_area_ratio=0.1,
            zero_text_page_count=5,
        )
        restored = DocumentProfile.model_validate_json(profile.model_dump_json())
        assert restored.origin_type == OriginType.ZERO_TEXT
        assert restored.zero_text_page_count == 5
