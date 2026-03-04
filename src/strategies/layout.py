"""Strategy B: LayoutExtractor — layout-aware extraction via Docling.

Triggers when:
  - layout_complexity in [multi_column, table_heavy, figure_heavy, mixed]
  - OR origin_type == mixed
  - OR Strategy A confidence < threshold_ab

Docling produces a DoclingDocument with:
  - Text blocks with bounding boxes and reading order
  - Tables as proper row/column objects (no merged-cell loss)
  - Figures with captions linked by spatial proximity
  - Section hierarchy reconstructed from heading detection

A DoclingDocumentAdapter normalizes this to our ExtractedDocument schema,
so the rest of the pipeline has a single interface regardless of strategy used.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import (
    BoundingBox,
    EscalationReason,
    ExtractedDocument,
    ExtractionStrategy,
    FigureBlock,
    TableCell,
    TableData,
    TextBlock,
)
from .base import BaseExtractor, ExtractionResult


def _docling_bbox_to_model(bbox_obj, page_num: int) -> BoundingBox:
    """Convert a Docling BoundingBox to our BoundingBox model."""
    # Docling uses (l, t, r, b) in pt, top-left origin
    return BoundingBox(
        x0=bbox_obj.l,
        y0=bbox_obj.t,
        x1=bbox_obj.r,
        y1=bbox_obj.b,
        page=page_num,
    )


def _adapt_docling_document(docling_doc, doc_id: str, filename: str) -> dict:
    """Adapt a DoclingDocument to our internal schema components.

    Returns: dict with keys: text_blocks, tables, figures, warnings
    """
    text_blocks: list[TextBlock] = []
    tables: list[TableData] = []
    figures: list[FigureBlock] = []
    warnings: list[str] = []
    reading_order = 0

    try:
        for item, level in docling_doc.iterate_items():
            item_type = type(item).__name__

            # ---- Text / Paragraph / Heading ----
            if item_type in ("TextItem", "SectionHeaderItem", "ParagraphItem"):
                try:
                    text = item.text if hasattr(item, "text") else str(item)
                    if not text.strip():
                        continue

                    page_num = item.prov[0].page_no if item.prov else 1
                    bbox = _docling_bbox_to_model(item.prov[0].bbox, page_num) if item.prov else BoundingBox(
                        x0=0, y0=0, x1=612, y1=792, page=page_num
                    )

                    is_heading = item_type == "SectionHeaderItem"
                    heading_level: Optional[int] = level if is_heading else None

                    # Build section path from parent items
                    section_path: list[str] = []
                    if hasattr(item, "parent") and item.parent:
                        parent = item.parent
                        while parent and hasattr(parent, "text"):
                            section_path.insert(0, parent.text)
                            parent = getattr(parent, "parent", None)

                    text_blocks.append(TextBlock(
                        text=text.strip(),
                        bbox=bbox,
                        reading_order=reading_order,
                        is_heading=is_heading,
                        heading_level=heading_level,
                        section_path=section_path,
                    ))
                    reading_order += 1
                except Exception as e:
                    warnings.append(f"Text block extraction error: {e}")

            # ---- Table ----
            elif item_type == "TableItem":
                try:
                    page_num = item.prov[0].page_no if item.prov else 1
                    bbox = _docling_bbox_to_model(item.prov[0].bbox, page_num) if item.prov else BoundingBox(
                        x0=0, y0=0, x1=612, y1=792, page=page_num
                    )

                    # Docling table data
                    table_df = item.export_to_dataframe() if hasattr(item, "export_to_dataframe") else None

                    if table_df is not None and not table_df.empty:
                        headers = list(table_df.columns.astype(str))
                        rows = [list(row.astype(str)) for _, row in table_df.iterrows()]
                    else:
                        # Fallback: use raw grid
                        grid = item.data.grid if hasattr(item, "data") and hasattr(item.data, "grid") else []
                        if not grid:
                            continue
                        headers = [str(cell.text or "") for cell in grid[0]] if grid else []
                        rows = [
                            [str(cell.text or "") for cell in row]
                            for row in grid[1:]
                        ]

                    # Build cell objects with span info
                    cells: list[TableCell] = []
                    if hasattr(item, "data") and hasattr(item.data, "grid"):
                        for r_idx, row in enumerate(item.data.grid):
                            for c_idx, cell in enumerate(row):
                                cells.append(TableCell(
                                    row=r_idx,
                                    col=c_idx,
                                    text=str(cell.text or "").strip(),
                                    is_header=(r_idx == 0),
                                    row_span=getattr(cell, "row_span", 1),
                                    col_span=getattr(cell, "col_span", 1),
                                ))

                    caption = getattr(item, "caption_text", None)

                    tables.append(TableData(
                        caption=caption,
                        bbox=bbox,
                        page=page_num,
                        headers=headers,
                        rows=rows,
                        cells=cells,
                        reading_order=reading_order,
                    ))
                    reading_order += 1
                except Exception as e:
                    warnings.append(f"Table extraction error: {e}")

            # ---- Figure ----
            elif item_type == "FigureItem":
                try:
                    page_num = item.prov[0].page_no if item.prov else 1
                    bbox = _docling_bbox_to_model(item.prov[0].bbox, page_num) if item.prov else BoundingBox(
                        x0=0, y0=0, x1=612, y1=792, page=page_num
                    )

                    caption = getattr(item, "caption_text", None)

                    figures.append(FigureBlock(
                        figure_id=f"{doc_id}-fig-{len(figures):04d}",
                        bbox=bbox,
                        page=page_num,
                        caption=caption,
                        reading_order=reading_order,
                    ))
                    reading_order += 1
                except Exception as e:
                    warnings.append(f"Figure extraction error: {e}")

    except Exception as e:
        warnings.append(f"Docling document iteration error: {e}")

    return {
        "text_blocks": text_blocks,
        "tables": tables,
        "figures": figures,
        "warnings": warnings,
    }


def _compute_layout_confidence(
    text_blocks: list[TextBlock],
    tables: list[TableData],
    page_count: int,
) -> float:
    """Estimate confidence for Layout extraction output.

    Heuristic: penalize documents where we got very few text blocks relative
    to page count (may indicate Docling couldn't parse the layout).
    """
    if page_count == 0:
        return 0.0

    # Expect at least 2 text blocks per non-trivial page
    expected_min_blocks = max(page_count * 0.5, 1)
    block_score = min(len(text_blocks) / expected_min_blocks, 1.0)

    # Tables are a bonus signal (we successfully parsed structure)
    table_bonus = min(len(tables) * 0.05, 0.20)

    confidence = min(block_score * 0.80 + table_bonus, 1.0)
    return round(confidence, 4)


class LayoutExtractor(BaseExtractor):
    """Strategy B — layout-aware extraction using Docling (IBM Research).

    Handles multi-column layouts, tables with merged cells, figure captions,
    and reading order reconstruction. More expensive than Strategy A (requires
    Docling's layout model) but still runs locally at zero API cost.
    """

    strategy = ExtractionStrategy.LAYOUT

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml"):
        self.rules_path = rules_path

        # Load escalation threshold
        import yaml
        threshold = 0.60
        path = Path(rules_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                if data:
                    threshold = data.get("extraction", {}).get(
                        "confidence_threshold_bc", 0.60
                    )
        self.confidence_threshold = threshold

    def extract(self, file_path: str, profile: DocumentProfile) -> ExtractionResult:
        """Run Strategy B extraction on the document using Docling."""
        start = time.perf_counter()
        self._validate_file(file_path)

        warnings: list[str] = []

        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False          # We handle scanned in Strategy C
            pipeline_options.do_table_structure = True

            converter = DocumentConverter()
            result = converter.convert(file_path)
            docling_doc = result.document

        except ImportError:
            warnings.append("Docling not installed — falling back to pdfplumber for layout extraction")
            from .fast_text import FastTextExtractor
            fallback = FastTextExtractor(self.rules_path)
            result = fallback.extract(file_path, profile)
            result.document.warnings.extend(warnings)
            result.document.strategy_used = self.strategy
            return result

        except Exception as e:
            warnings.append(f"Docling conversion failed: {e}")
            text_blocks, tables, figures = [], [], []
            confidence = 0.0
        else:
            adapted = _adapt_docling_document(docling_doc, profile.doc_id, profile.filename)
            text_blocks = adapted["text_blocks"]
            tables = adapted["tables"]
            figures = adapted["figures"]
            warnings.extend(adapted["warnings"])
            confidence = _compute_layout_confidence(text_blocks, tables, profile.page_count)

        elapsed = time.perf_counter() - start

        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            page_count=profile.page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            strategy_used=self.strategy,
            confidence_score=confidence,
            cost_estimate_usd=0.0,  # Local — free
            processing_time_seconds=round(elapsed, 3),
            warnings=warnings,
        )

        escalate = confidence < self.confidence_threshold
        detail = (
            f"confidence {confidence:.4f} < threshold {self.confidence_threshold}"
            if escalate else None
        )

        return ExtractionResult(
            document=doc,
            escalate=escalate,
            escalation_reason=EscalationReason.LOW_CONFIDENCE if escalate else None,
            escalation_detail=detail,
        )
