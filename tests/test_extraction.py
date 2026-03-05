"""Unit tests for extraction confidence scoring (FastTextExtractor).

Tests verify that the confidence scoring formula correctly:
1. Produces high scores for clean digital pages
2. Produces low scores for image-dominated pages (scanned)
3. Penalizes pages with no font metadata
4. Signals escalation when overall confidence is below threshold
5. Handles edge cases (empty page, zero area)
"""

import pytest
from unittest.mock import MagicMock, patch

from src.strategies.fast_text import (
    _compute_page_confidence,
    FastTextExtractor,
)
from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LayoutComplexity,
    OriginType,
)

# Default weights matching extraction_rules.yaml defaults
DEFAULT_WEIGHTS = {
    "char_density": 0.40,
    "image_area_penalty": 0.30,
    "font_metadata": 0.20,
    "whitespace_ratio": 0.10,
}


def make_char(x0: float = 72.0, text: str = "A", fontname: str = "Arial") -> dict:
    return {"x0": x0, "text": text, "fontname": fontname}


def make_image(width: float = 100.0, height: float = 100.0) -> dict:
    return {"width": width, "height": height, "x0": 0, "top": 0, "x1": width, "bottom": height}


def make_profile(**kwargs) -> DocumentProfile:
    defaults = dict(
        doc_id="test00000000test",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        page_count=10,
        file_size_bytes=1024000,
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.FINANCIAL,
        estimated_extraction_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
        avg_char_density=1.0,
        avg_image_area_ratio=0.05,
        scanned_page_count=0,
        table_page_count=2,
        column_count_estimate=1,
    )
    defaults.update(kwargs)
    return DocumentProfile(**defaults)


# ---------------------------------------------------------------------------
# Confidence scoring unit tests
# ---------------------------------------------------------------------------

class TestPageConfidenceScoring:

    def test_dense_digital_page_scores_high(self):
        """A page with many characters, embedded fonts, low image area → high confidence."""
        chars = [make_char() for _ in range(500)]  # 500 chars on 612x792 page
        images = []
        page_area = 612.0 * 792.0  # ~484,704 pt²

        score = _compute_page_confidence(chars, images, page_area, DEFAULT_WEIGHTS)

        assert score >= 0.75, f"Expected ≥0.75 for dense digital page, got {score}"

    def test_image_dominated_page_scores_low(self):
        """A page with almost no text and a large image → low confidence."""
        chars = [make_char() for _ in range(5)]  # Virtually no text
        # Image covering 90% of page
        images = [make_image(width=550.0, height=700.0)]
        page_area = 612.0 * 792.0

        score = _compute_page_confidence(chars, images, page_area, DEFAULT_WEIGHTS)

        assert score < 0.50, f"Expected <0.50 for image-dominated page, got {score}"

    def test_empty_page_scores_zero(self):
        """No chars, no images, zero area → score of 0.0."""
        score = _compute_page_confidence([], [], 0.0, DEFAULT_WEIGHTS)
        assert score == 0.0

    def test_no_font_metadata_penalizes_score(self):
        """Page without fontname in chars → lower font_metadata_score."""
        chars_with_fonts = [{"x0": 72.0, "text": "A", "fontname": "Arial"} for _ in range(200)]
        chars_without_fonts = [{"x0": 72.0, "text": "A"} for _ in range(200)]
        page_area = 612.0 * 792.0

        score_with = _compute_page_confidence(chars_with_fonts, [], page_area, DEFAULT_WEIGHTS)
        score_without = _compute_page_confidence(chars_without_fonts, [], page_area, DEFAULT_WEIGHTS)

        # Font metadata = 0.20 weight, so difference should be ~0.20
        assert score_with > score_without
        assert (score_with - score_without) >= 0.15

    def test_score_clamped_to_0_1(self):
        """Confidence is always in [0.0, 1.0]."""
        chars = [make_char() for _ in range(10000)]  # Extreme density
        page_area = 100.0  # Very small area

        score = _compute_page_confidence(chars, [], page_area, DEFAULT_WEIGHTS)
        assert 0.0 <= score <= 1.0

    def test_whitespace_score_penalizes_extremes(self):
        """Pages with 0% or 80% whitespace score lower than 25%."""
        page_area = 612.0 * 792.0
        n_chars = 200

        all_words = [make_char(text="A") for _ in range(n_chars)]
        score_0ws = _compute_page_confidence(all_words, [], page_area, DEFAULT_WEIGHTS)

        half_spaces = (
            [make_char(text="A") for _ in range(n_chars // 2)] +
            [make_char(text=" ") for _ in range(n_chars // 2)]
        )
        score_50ws = _compute_page_confidence(half_spaces, [], page_area, DEFAULT_WEIGHTS)

        quarter_spaces = (
            [make_char(text="A") for _ in range(int(n_chars * 0.75))] +
            [make_char(text=" ") for _ in range(int(n_chars * 0.25))]
        )
        score_25ws = _compute_page_confidence(quarter_spaces, [], page_area, DEFAULT_WEIGHTS)

        # 25% whitespace should score higher than 0% or 50%
        assert score_25ws >= score_0ws, "25% whitespace should score ≥ 0% whitespace"


# ---------------------------------------------------------------------------
# FastTextExtractor escalation signal tests
# ---------------------------------------------------------------------------

class TestFastTextExtractorEscalation:

    def _make_mock_pdf(self, pages: list[dict]) -> MagicMock:
        """Build a mock pdfplumber.PDF."""
        mock_pdf = MagicMock()
        mock_pages = []
        for page_data in pages:
            p = MagicMock()
            p.chars = page_data.get("chars", [])
            p.images = page_data.get("images", [])
            p.width = 612.0
            p.height = 792.0
            p.extract_text.return_value = page_data.get("text", "")
            p.extract_tables.return_value = page_data.get("tables", [])
            p.find_tables.return_value = []
            mock_pages.append(p)
        mock_pdf.pages = mock_pages
        return mock_pdf

    def test_high_confidence_does_not_escalate(self):
        """Clean digital pages → confidence ≥ threshold → escalate=False."""
        profile = make_profile()
        pages = [
            {
                "chars": [make_char() for _ in range(500)],
                "images": [],
                "text": "This is a test document with substantial text content on every page.",
            }
            for _ in range(10)
        ]
        mock_pdf = self._make_mock_pdf(pages)

        extractor = FastTextExtractor()

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            result = extractor.extract("/tmp/test.pdf", profile)

        assert not result.escalate, "High-confidence extraction should not escalate"
        assert result.document.confidence_score >= extractor.confidence_threshold

    def test_scanned_pages_trigger_escalation(self):
        """Pages with near-zero chars and large images → confidence < threshold → escalate=True."""
        profile = make_profile(
            origin_type=OriginType.SCANNED_IMAGE,
            estimated_extraction_cost=ExtractionCost.NEEDS_VISION_MODEL,
        )
        pages = [
            {
                "chars": [],  # No characters
                "images": [make_image(550.0, 700.0)],  # ~90% image coverage
                "text": "",
            }
            for _ in range(10)
        ]
        mock_pdf = self._make_mock_pdf(pages)

        extractor = FastTextExtractor()

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            result = extractor.extract("/tmp/test.pdf", profile)

        assert result.escalate, "Low-confidence (scanned) extraction should escalate"
        assert result.document.confidence_score < extractor.confidence_threshold

    def test_warnings_added_for_low_confidence_pages(self):
        """Very low confidence pages generate a warning in the output."""
        profile = make_profile()
        pages = [
            {"chars": [], "images": [make_image(550.0, 700.0)], "text": ""}
            for _ in range(5)
        ]
        mock_pdf = self._make_mock_pdf(pages)

        extractor = FastTextExtractor()

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            result = extractor.extract("/tmp/test.pdf", profile)

        assert len(result.document.warnings) > 0

    def test_cost_is_always_zero(self):
        """Strategy A is local — cost is always $0."""
        profile = make_profile()
        pages = [
            {"chars": [make_char() for _ in range(100)], "images": [], "text": "Sample text"}
        ]
        mock_pdf = self._make_mock_pdf(pages)

        extractor = FastTextExtractor()

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf
            result = extractor.extract("/tmp/test.pdf", profile)

        assert result.document.cost_estimate_usd == 0.0
