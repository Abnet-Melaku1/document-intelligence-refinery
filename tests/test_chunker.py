"""Unit tests for ChunkingEngine and ChunkValidator.

Tests verify that:
1. R1 — table header protection is enforced
2. R2 — figure captions stored as metadata, no standalone CAPTION LDU
3. R3 — list items merged into single LIST LDU; split only when over token limit
4. R4 — parent_section propagated to all non-heading LDUs
5. R5 — cross-references detected and stored as LDURelationship objects
6. ChunkValidator raises ChunkingRuleViolation on hard breaches
7. Long paragraphs are split into multiple LDUs within token limits
"""

import pytest
from unittest.mock import patch

from src.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    ExtractionStrategy,
    FigureBlock,
    TableData,
    TextBlock,
)
from src.models.ldu import ChunkType, LDU
from src.agents.chunker import (
    ChunkingEngine,
    ChunkingRuleViolation,
    ChunkValidator,
    _is_list_item,
    _extract_xrefs,
    _split_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox(page: int = 1) -> BoundingBox:
    return BoundingBox(x0=72.0, y0=100.0, x1=540.0, y1=120.0, page=page)


def _heading(text: str, level: int = 1, order: int = 0) -> TextBlock:
    return TextBlock(
        text=text, bbox=_bbox(), reading_order=order,
        is_heading=True, heading_level=level,
    )


def _para(text: str, order: int = 1) -> TextBlock:
    return TextBlock(text=text, bbox=_bbox(), reading_order=order)


def _list_item(text: str, order: int = 1) -> TextBlock:
    return TextBlock(text=f"• {text}", bbox=_bbox(), reading_order=order)


def _table(headers: list[str], rows: list[list[str]], order: int = 2) -> TableData:
    return TableData(
        headers=headers, rows=rows, bbox=_bbox(), page=1, reading_order=order,
    )


def _figure(fig_id: str, caption: str | None = None, alt: str | None = None, order: int = 3) -> FigureBlock:
    return FigureBlock(
        figure_id=fig_id, bbox=_bbox(), page=1,
        caption=caption, alt_text=alt, reading_order=order,
    )


def _doc(
    text_blocks: list[TextBlock] | None = None,
    tables: list[TableData] | None = None,
    figures: list[FigureBlock] | None = None,
) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="testdoc0000000001",
        filename="test.pdf",
        page_count=5,
        text_blocks=text_blocks or [],
        tables=tables or [],
        figures=figures or [],
        strategy_used=ExtractionStrategy.FAST_TEXT,
        confidence_score=0.9,
    )


def _engine() -> ChunkingEngine:
    """ChunkingEngine with default config."""
    return ChunkingEngine()


# ---------------------------------------------------------------------------
# R4 — Section context propagation
# ---------------------------------------------------------------------------

class TestR4SectionContext:

    def test_heading_sets_parent_section_on_following_paragraphs(self):
        """Paragraphs after a heading carry that heading as parent_section."""
        doc = _doc(text_blocks=[
            _heading("Introduction", level=1, order=0),
            _para("This section covers the background.", order=1),
            _para("More background details here.", order=2),
        ])
        ldus = _engine().run(doc)
        paras = [c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH]
        assert all(c.parent_section == "Introduction" for c in paras)

    def test_heading_has_no_parent_section(self):
        """HEADING LDUs do not carry a parent_section."""
        doc = _doc(text_blocks=[_heading("Chapter 1", order=0)])
        ldus = _engine().run(doc)
        headings = [c for c in ldus if c.chunk_type == ChunkType.HEADING]
        assert all(c.parent_section is None for c in headings)

    def test_section_path_updates_on_sub_heading(self):
        """Sub-heading adds to section_path without discarding the parent."""
        doc = _doc(text_blocks=[
            _heading("Chapter 1", level=1, order=0),
            _heading("1.1 Overview", level=2, order=1),
            _para("Overview text.", order=2),
        ])
        ldus = _engine().run(doc)
        para = next(c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH)
        assert "Chapter 1" in para.section_path
        assert "1.1 Overview" in para.section_path

    def test_table_inherits_parent_section(self):
        """Table LDUs carry parent_section from preceding heading."""
        doc = _doc(
            text_blocks=[_heading("Financial Summary", order=0)],
            tables=[_table(["Year", "Revenue"], [["2023", "100M"]], order=1)],
        )
        ldus = _engine().run(doc)
        tbl = next(c for c in ldus if c.chunk_type == ChunkType.TABLE)
        assert tbl.parent_section == "Financial Summary"


# ---------------------------------------------------------------------------
# R1 — Table header protection
# ---------------------------------------------------------------------------

class TestR1TableHeaders:

    def test_table_ldu_includes_headers(self):
        """TABLE LDU carries table_headers from TableData.headers."""
        doc = _doc(tables=[_table(["Col A", "Col B"], [["v1", "v2"]])])
        ldus = _engine().run(doc)
        tbl = next(c for c in ldus if c.chunk_type == ChunkType.TABLE)
        assert tbl.table_headers == ["Col A", "Col B"]

    def test_missing_headers_fall_back_to_first_row(self):
        """When TableData.headers is empty, row[0] is used as headers (R1 satisfied)."""
        td = _table([], [["Col A", "Col B"], ["v1", "v2"]])
        td.headers = []
        doc = _doc(tables=[td])
        ldus = _engine().run(doc)
        tbl = next(c for c in ldus if c.chunk_type == ChunkType.TABLE)
        assert tbl.table_headers == ["Col A", "Col B"]

    def test_table_rows_stored_in_ldu(self):
        """TABLE LDU carries table_rows (data rows, not headers)."""
        doc = _doc(tables=[_table(["Name", "Value"], [["Alpha", "1"], ["Beta", "2"]])])
        ldus = _engine().run(doc)
        tbl = next(c for c in ldus if c.chunk_type == ChunkType.TABLE)
        assert ["Alpha", "1"] in tbl.table_rows

    def test_validator_raises_on_table_without_headers(self):
        """ChunkValidator raises R1 violation if TABLE LDU has no table_headers."""
        ldu = LDU(
            chunk_id="testdoc0000000001-0000",
            doc_id="testdoc0000000001",
            content="Col A | Col B\nv1 | v2",
            chunk_type=ChunkType.TABLE,
            token_count=10,
            page_refs=[1],
            table_headers=[],   # violates R1
            content_hash="",
        )
        validator = ChunkValidator({"rule_r1_table_header_protection": True})
        with pytest.raises(ChunkingRuleViolation) as exc_info:
            validator.validate([ldu])
        assert "R1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R2 — Caption as metadata
# ---------------------------------------------------------------------------

class TestR2CaptionAsMetadata:

    def test_figure_caption_stored_in_ldu_field(self):
        """Figure caption ends up in LDU.figure_caption, not as a separate chunk."""
        doc = _doc(figures=[_figure("fig-1", caption="Figure 1: Revenue trend", alt="bar chart")])
        ldus = _engine().run(doc)
        fig_ldus = [c for c in ldus if c.chunk_type == ChunkType.FIGURE]
        assert len(fig_ldus) == 1
        assert fig_ldus[0].figure_caption == "Figure 1: Revenue trend"

    def test_no_standalone_caption_ldu(self):
        """No LDU with chunk_type == CAPTION is ever emitted."""
        doc = _doc(figures=[_figure("fig-1", caption="Figure 1: Revenue trend")])
        ldus = _engine().run(doc)
        assert not any(c.chunk_type == ChunkType.CAPTION for c in ldus)

    def test_figure_ldu_content_is_alt_text_when_available(self):
        """FIGURE LDU content is the alt_text when provided."""
        doc = _doc(figures=[_figure("fig-1", alt="Bar chart showing quarterly revenue")])
        ldus = _engine().run(doc)
        fig = next(c for c in ldus if c.chunk_type == ChunkType.FIGURE)
        assert "Bar chart" in fig.content

    def test_validator_raises_on_standalone_caption(self):
        """ChunkValidator raises R2 violation on a standalone CAPTION LDU."""
        ldu = LDU(
            chunk_id="testdoc0000000001-0000",
            doc_id="testdoc0000000001",
            content="Figure 1: Revenue trend",
            chunk_type=ChunkType.CAPTION,
            token_count=5,
            page_refs=[1],
            content_hash="",
        )
        validator = ChunkValidator({"rule_r2_caption_as_metadata": True})
        with pytest.raises(ChunkingRuleViolation) as exc_info:
            validator.validate([ldu])
        assert "R2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R3 — List unity
# ---------------------------------------------------------------------------

class TestR3ListUnity:

    def test_consecutive_list_items_merged_into_one_ldu(self):
        """Three consecutive bullet items → one LIST LDU."""
        doc = _doc(text_blocks=[
            _heading("Findings", order=0),
            _list_item("Revenue grew 34%", order=1),
            _list_item("Operating costs fell 12%", order=2),
            _list_item("Net profit up 48%", order=3),
        ])
        ldus = _engine().run(doc)
        list_ldus = [c for c in ldus if c.chunk_type == ChunkType.LIST]
        assert len(list_ldus) == 1
        assert "Revenue grew 34%" in list_ldus[0].content
        assert "Net profit up 48%" in list_ldus[0].content

    def test_list_is_separate_from_paragraph(self):
        """A paragraph between list blocks separates them into different LIST LDUs."""
        doc = _doc(text_blocks=[
            _list_item("Item A", order=0),
            _para("Intervening paragraph.", order=1),
            _list_item("Item B", order=2),
        ])
        ldus = _engine().run(doc)
        list_ldus = [c for c in ldus if c.chunk_type == ChunkType.LIST]
        assert len(list_ldus) == 2

    def test_numbered_list_items_detected(self):
        """Numbered list items (1. 2. 3.) are treated as list blocks."""
        doc = _doc(text_blocks=[
            TextBlock(text="1. First item", bbox=_bbox(), reading_order=0),
            TextBlock(text="2. Second item", bbox=_bbox(), reading_order=1),
        ])
        ldus = _engine().run(doc)
        assert any(c.chunk_type == ChunkType.LIST for c in ldus)

    def test_is_list_item_helper(self):
        """_is_list_item correctly identifies various list markers."""
        assert _is_list_item("• bullet item")
        assert _is_list_item("1. numbered item")
        assert _is_list_item("a. lettered item")
        assert _is_list_item("- dash item")
        assert not _is_list_item("Regular paragraph text.")
        assert not _is_list_item("Revenue grew 34% this year.")


# ---------------------------------------------------------------------------
# R5 — Cross-reference resolution
# ---------------------------------------------------------------------------

class TestR5CrossReferences:

    def test_table_reference_detected(self):
        """'Table 1' in a paragraph → LDURelationship with references_table."""
        doc = _doc(
            text_blocks=[
                _para("As shown in Table 1, revenue increased significantly.", order=1),
            ],
            tables=[_table(["Year", "Revenue"], [["2023", "100M"]], order=0)],
        )
        ldus = _engine().run(doc)
        para = next(c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH)
        ref_types = [r.relationship_type for r in para.relationships]
        assert "references_table" in ref_types

    def test_figure_reference_detected(self):
        """'Figure 1' in a paragraph → LDURelationship with references_figure."""
        doc = _doc(
            text_blocks=[
                _para("Figure 1 shows the distribution of assets.", order=1),
            ],
            figures=[_figure("fig-1", order=0)],
        )
        ldus = _engine().run(doc)
        para = next(c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH)
        ref_types = [r.relationship_type for r in para.relationships]
        assert "references_figure" in ref_types

    def test_section_reference_detected(self):
        """'see Section 3.2' → LDURelationship with see_also."""
        doc = _doc(text_blocks=[
            _para("For details, see Section 3.2 of this report.", order=0),
        ])
        ldus = _engine().run(doc)
        para = next(c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH)
        ref_types = [r.relationship_type for r in para.relationships]
        assert "see_also" in ref_types

    def test_resolved_reference_points_to_chunk_id(self):
        """Table reference is resolved to the actual table chunk_id."""
        doc = _doc(
            text_blocks=[
                _para("Refer to Table 1 for details.", order=1),
            ],
            tables=[_table(["A", "B"], [["x", "y"]], order=0)],
        )
        ldus = _engine().run(doc)
        para = next(c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH)
        table = next(c for c in ldus if c.chunk_type == ChunkType.TABLE)
        table_ref = next(r for r in para.relationships if r.relationship_type == "references_table")
        assert table_ref.target_chunk_id == table.chunk_id

    def test_extract_xrefs_helper(self):
        """_extract_xrefs returns correct relationship types."""
        registry = {"table 1": "doc-0001", "figure 2": "doc-0005"}
        text = "See Table 1 and Figure 2 for details, and Section 4.1."
        refs = _extract_xrefs(text, registry)
        types = {r.relationship_type for r in refs}
        assert "references_table" in types
        assert "references_figure" in types
        assert "see_also" in types

    def test_unresolvable_reference_marked_unresolved(self):
        """References to unknown tables get target_chunk_id='unresolved'."""
        refs = _extract_xrefs("See Table 99 for more.", {})
        assert refs[0].target_chunk_id == "unresolved"


# ---------------------------------------------------------------------------
# Text splitting (long paragraphs)
# ---------------------------------------------------------------------------

class TestTextSplitting:

    def test_short_text_not_split(self):
        """Text within max_tokens limit returns as a single chunk."""
        text = "This is a short paragraph."
        result = _split_text(text, max_tokens=100)
        assert result == [text]

    def test_long_text_split_into_multiple_parts(self):
        """Text exceeding max_tokens is split into multiple parts."""
        # Build text with ~200 tokens
        long_text = ". ".join(["This is a sentence"] * 30) + "."
        parts = _split_text(long_text, max_tokens=50)
        assert len(parts) > 1

    def test_each_part_within_token_limit(self):
        """Each split part has token_count <= max_tokens (with small tolerance)."""
        from src.agents.chunker import _count_tokens
        long_text = ". ".join([f"Sentence number {i} with some extra words here" for i in range(20)]) + "."
        parts = _split_text(long_text, max_tokens=60)
        for part in parts:
            # Allow slight overflow at sentence granularity
            assert _count_tokens(part) <= 80

    def test_long_paragraph_produces_multiple_ldus(self):
        """A text block exceeding max_tokens is split into multiple PARAGRAPH LDUs."""
        from src.agents.chunker import _count_tokens
        # Build ~600 tokens: each sentence ~12 tokens, 50 sentences = ~600 tokens > 512 limit
        long_text = ". ".join(
            [f"This is sentence number {i} with enough words to push token count higher" for i in range(50)]
        ) + "."
        doc = _doc(text_blocks=[_para(long_text, order=0)])

        engine = ChunkingEngine()
        ldus = engine.run(doc)
        paras = [c for c in ldus if c.chunk_type == ChunkType.PARAGRAPH]
        assert len(paras) > 1
        assert all(c.token_count <= engine.max_tokens + 20 for c in paras)


# ---------------------------------------------------------------------------
# ChunkValidator — token limit
# ---------------------------------------------------------------------------

class TestChunkValidatorTokenLimit:

    def test_oversized_chunk_raises(self):
        """ChunkValidator raises when a chunk exceeds max_tokens."""
        ldu = LDU(
            chunk_id="testdoc0000000001-0000",
            doc_id="testdoc0000000001",
            content="word " * 600,
            chunk_type=ChunkType.PARAGRAPH,
            token_count=600,
            page_refs=[1],
            content_hash="",
        )
        validator = ChunkValidator({"max_tokens_per_chunk": 512})
        with pytest.raises(ChunkingRuleViolation) as exc_info:
            validator.validate([ldu])
        assert "TOKEN_LIMIT" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration: full document
# ---------------------------------------------------------------------------

class TestFullDocumentIntegration:

    def test_mixed_document_produces_all_chunk_types(self):
        """A document with headings, text, list, table, and figure → all types present."""
        doc = _doc(
            text_blocks=[
                _heading("1. Introduction", level=1, order=0),
                _para("This report covers annual performance.", order=1),
                _list_item("Revenue grew 34%", order=2),
                _list_item("Costs fell 12%", order=3),
            ],
            tables=[_table(["Metric", "Value"], [["Revenue", "ETB 4.2B"]], order=4)],
            figures=[_figure("fig-01", caption="Figure 1: Growth chart", alt="bar chart", order=5)],
        )
        ldus = _engine().run(doc)
        types = {c.chunk_type for c in ldus}
        assert ChunkType.HEADING in types
        assert ChunkType.PARAGRAPH in types
        assert ChunkType.LIST in types
        assert ChunkType.TABLE in types
        assert ChunkType.FIGURE in types
        assert ChunkType.CAPTION not in types  # R2

    def test_all_chunks_have_doc_id(self):
        """Every LDU carries the correct doc_id."""
        doc = _doc(
            text_blocks=[_heading("H1", order=0), _para("Text.", order=1)],
            tables=[_table(["A"], [["1"]], order=2)],
        )
        ldus = _engine().run(doc)
        assert all(c.doc_id == doc.doc_id for c in ldus)

    def test_chunk_ids_are_sequential(self):
        """Chunk IDs follow the {doc_id}-{sequence:04d} format in order."""
        doc = _doc(text_blocks=[
            _heading("H", order=0),
            _para("P1", order=1),
            _para("P2", order=2),
        ])
        ldus = _engine().run(doc)
        for i, chunk in enumerate(ldus):
            expected_id = f"{doc.doc_id}-{i:04d}"
            assert chunk.chunk_id == expected_id
