"""Strategy A: FastTextExtractor — low-cost extraction via pdfplumber.

Triggers when: origin_type=native_digital AND layout_complexity=single_column.

Confidence scoring uses four signals weighted by extraction_rules.yaml:
  - char_density_score   (0.40 weight): characters per pt² normalized to [0,1]
  - image_area_penalty   (0.30 weight): penalizes image-dominated pages
  - font_metadata_score  (0.20 weight): presence of embedded font data
  - whitespace_ratio     (0.10 weight): reasonable whitespace = structured text

If the overall confidence < threshold_ab (default 0.75), the ExtractionRouter
will automatically retry with Strategy B (LayoutExtractor).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pdfplumber
import yaml

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    ExtractionStrategy,
    FigureBlock,
    TableCell,
    TableData,
    TextBlock,
)
from .base import BaseExtractor, ExtractionResult

# Normalization cap for char density: above this, score = 1.0
_CHAR_DENSITY_CAP = 2.0  # chars/pt² — typical dense text page


def _load_weights(rules_path: str = "rubric/extraction_rules.yaml") -> dict:
    path = Path(rules_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            if data:
                return data.get("extraction", {}).get("confidence_weights", {})
    return {}


def _compute_page_confidence(
    chars: list,
    images: list,
    page_area: float,
    weights: dict,
) -> float:
    """Compute confidence score for a single page extraction.

    Returns a float in [0.0, 1.0] where 1.0 = highly confident digital text.
    """
    w_density = weights.get("char_density", 0.40)
    w_image = weights.get("image_area_penalty", 0.30)
    w_font = weights.get("font_metadata", 0.20)
    w_ws = weights.get("whitespace_ratio", 0.10)

    # Signal 1: Character density (normalized to cap)
    char_count = len(chars)
    char_density = char_count / page_area if page_area > 0 else 0.0
    char_density_score = min(char_density / _CHAR_DENSITY_CAP, 1.0)

    # Signal 2: Image area (penalty — high image = lower confidence)
    image_area = sum(
        img.get("width", 0) * img.get("height", 0) for img in images
    )
    image_area_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0.0
    image_area_score = 1.0 - image_area_ratio  # Invert: low image = high score

    # Signal 3: Font metadata presence
    fonts_present = any(c.get("fontname") for c in chars[:20])
    font_metadata_score = 1.0 if fonts_present else 0.0

    # Signal 4: Whitespace ratio (space chars / total chars)
    if char_count > 0:
        space_count = sum(1 for c in chars if c.get("text", "") == " ")
        ws_ratio = space_count / char_count
        # Good whitespace range: 10%–40% of chars are spaces
        if 0.10 <= ws_ratio <= 0.40:
            whitespace_score = 1.0
        elif ws_ratio < 0.10:
            whitespace_score = ws_ratio / 0.10
        else:
            whitespace_score = max(0.0, 1.0 - (ws_ratio - 0.40) / 0.60)
    else:
        whitespace_score = 0.0

    confidence = (
        w_density * char_density_score
        + w_image * image_area_score
        + w_font * font_metadata_score
        + w_ws * whitespace_score
    )
    return round(min(max(confidence, 0.0), 1.0), 4)


def _extract_tables_from_page(
    page: pdfplumber.page.Page,
    page_num: int,
    reading_order_offset: int,
) -> list[TableData]:
    """Extract tables from a single page as structured TableData objects."""
    tables = []
    try:
        raw_tables = page.extract_tables()
    except Exception:
        return []

    for table_idx, raw_table in enumerate(raw_tables):
        if not raw_table or len(raw_table) < 2:
            continue

        # First row is headers
        headers = [str(cell or "").strip() for cell in raw_table[0]]
        rows = [
            [str(cell or "").strip() for cell in row]
            for row in raw_table[1:]
        ]

        # Build cell objects
        cells: list[TableCell] = []
        for r_idx, row in enumerate(raw_table):
            for c_idx, cell in enumerate(row):
                cells.append(TableCell(
                    row=r_idx,
                    col=c_idx,
                    text=str(cell or "").strip(),
                    is_header=(r_idx == 0),
                ))

        # Try to get bounding box from pdfplumber's table finder
        try:
            found_tables = page.find_tables()
            bbox_obj = found_tables[table_idx].bbox if table_idx < len(found_tables) else None
            if bbox_obj:
                bbox = BoundingBox(
                    x0=bbox_obj[0], y0=bbox_obj[1],
                    x1=bbox_obj[2], y1=bbox_obj[3],
                    page=page_num,
                )
            else:
                bbox = BoundingBox(x0=0, y0=0, x1=page.width, y1=page.height, page=page_num)
        except Exception:
            bbox = BoundingBox(x0=0, y0=0, x1=page.width, y1=page.height, page=page_num)

        tables.append(TableData(
            bbox=bbox,
            page=page_num,
            headers=headers,
            rows=rows,
            cells=cells,
            reading_order=reading_order_offset + table_idx,
        ))

    return tables


def _extract_text_blocks_from_page(
    page: pdfplumber.page.Page,
    page_num: int,
    reading_order_offset: int,
) -> list[TextBlock]:
    """Extract text blocks from a page, filtering out table regions."""
    text = page.extract_text(x_tolerance=3, y_tolerance=3)
    if not text:
        return []

    # Single block per page for Strategy A (simple, no column splitting)
    # Strategy B (Docling) handles block-level granularity
    page_bbox = BoundingBox(
        x0=0, y0=0, x1=page.width, y1=page.height, page=page_num
    )

    block = TextBlock(
        text=text.strip(),
        bbox=page_bbox,
        reading_order=reading_order_offset,
        font_size=None,
        is_heading=False,
    )
    return [block]


class FastTextExtractor(BaseExtractor):
    """Strategy A — fast, local, zero-cost extraction using pdfplumber.

    Best for: native digital PDFs with single-column layouts.
    Limitation: does not reconstruct reading order for multi-column layouts.
    """

    strategy = ExtractionStrategy.FAST_TEXT

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml"):
        self.rules_path = rules_path
        self.weights = _load_weights(rules_path)

        # Load threshold
        threshold = 0.75
        path = Path(rules_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                if data:
                    threshold = data.get("extraction", {}).get(
                        "confidence_threshold_ab", 0.75
                    )
        self.confidence_threshold = threshold

    def extract(self, file_path: str, profile: DocumentProfile) -> ExtractionResult:
        """Run Strategy A extraction on the document."""
        start = time.perf_counter()
        path = self._validate_file(file_path)

        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        figures: list[FigureBlock] = []
        page_confidences: list[float] = []
        warnings: list[str] = []
        reading_order = 0

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                page_area = page.width * page.height

                chars = page.chars
                images = page.images

                # Confidence for this page
                page_conf = _compute_page_confidence(
                    chars, images, page_area, self.weights
                )
                page_confidences.append(page_conf)

                if page_conf < 0.30:
                    warnings.append(
                        f"Page {page_num}: very low confidence ({page_conf:.2f}) — "
                        f"likely scanned or image-heavy"
                    )
                    reading_order += 1
                    continue

                # Extract text blocks
                page_blocks = _extract_text_blocks_from_page(
                    page, page_num, reading_order
                )
                text_blocks.extend(page_blocks)
                reading_order += len(page_blocks)

                # Extract tables
                page_tables = _extract_tables_from_page(
                    page, page_num, reading_order
                )
                tables.extend(page_tables)
                reading_order += len(page_tables)

                # Detect large image figures
                for img_idx, img in enumerate(images):
                    img_area = img.get("width", 0) * img.get("height", 0)
                    if img_area > 10_000:  # Large enough to be a figure
                        figures.append(FigureBlock(
                            figure_id=f"{profile.doc_id}-fig-p{page_num}-{img_idx}",
                            bbox=BoundingBox(
                                x0=img.get("x0", 0),
                                y0=img.get("top", 0),
                                x1=img.get("x1", page.width),
                                y1=img.get("bottom", page.height),
                                page=page_num,
                            ),
                            page=page_num,
                            reading_order=reading_order,
                        ))
                        reading_order += 1

        # Overall document confidence = mean of page confidences
        overall_confidence = (
            sum(page_confidences) / len(page_confidences)
            if page_confidences else 0.0
        )
        overall_confidence = round(overall_confidence, 4)

        elapsed = time.perf_counter() - start

        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            page_count=profile.page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            strategy_used=self.strategy,
            confidence_score=overall_confidence,
            cost_estimate_usd=0.0,  # Local — free
            processing_time_seconds=round(elapsed, 3),
            warnings=warnings,
        )

        # Signal escalation if confidence is too low
        escalate = overall_confidence < self.confidence_threshold

        return ExtractionResult(document=doc, escalate=escalate)
