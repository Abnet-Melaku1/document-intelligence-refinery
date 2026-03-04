"""Strategy C: VisionExtractor — VLM-based extraction via OpenRouter.

Triggers when:
  - origin_type == scanned_image
  - OR Strategy A/B confidence < their respective thresholds
  - OR handwriting detected (future: handwriting classifier)

Uses Gemini Flash (via OpenRouter) as the primary model — cheapest multimodal
capable of structured document extraction. Falls back to GPT-4o-mini if primary
is unavailable.

BUDGET GUARD: Tracks cumulative token spend per document. If the estimated cost
would exceed budget_cap_usd (default $0.10), the extractor stops and returns
what it has, with a warning. This prevents runaway API spend on large scanned docs.

Cost estimate: Gemini Flash ~$0.075/1M input tokens, $0.30/1M output tokens.
At ~2000 tokens/page (image + prompt), and 3000 output tokens/page:
  - 400-page doc worst case: ~$0.40 input + $0.36 output = $0.76
  - With budget_cap at $0.10: ~50 pages max before cap
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
import yaml

from src.models.document_profile import DocumentProfile, DomainHint
from src.models.extracted_document import (
    BoundingBox,
    EscalationReason,
    ExtractedDocument,
    ExtractionStrategy,
    FigureBlock,
    TableData,
    TextBlock,
)
from .base import BaseExtractor, ExtractionResult

# Cost estimates per 1M tokens (USD) — Gemini Flash 1.5
_COST_INPUT_PER_M = 0.075
_COST_OUTPUT_PER_M = 0.30


def _estimate_page_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * _COST_INPUT_PER_M + \
           (output_tokens / 1_000_000) * _COST_OUTPUT_PER_M


def _pdf_page_to_base64(file_path: str, page_num: int) -> str:
    """Render a PDF page to a base64-encoded PNG for VLM input."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to render page {page_num} to image: {e}") from e


def _build_extraction_prompt(domain: DomainHint, prompts_config: dict) -> str:
    """Select domain-appropriate extraction prompt from config."""
    domain_key = domain.value
    prompt = prompts_config.get(domain_key, prompts_config.get("general", ""))
    if not prompt:
        prompt = (
            "Extract all text, tables, and figures from this document page. "
            "Return structured JSON with: text_blocks (list of {text, is_heading}), "
            "tables (list of {headers, rows, caption}), figures (list of {caption})."
        )
    return prompt.strip()


def _parse_vlm_response(response_text: str, page_num: int, doc_id: str) -> dict:
    """Parse VLM JSON response into text_blocks, tables, figures."""
    text_blocks: list[TextBlock] = []
    tables: list[TableData] = []
    figures: list[FigureBlock] = []
    warnings: list[str] = []

    # Try to extract JSON from response (VLMs sometimes wrap in markdown code blocks)
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat entire response as plain text
        warnings.append(f"Page {page_num}: VLM returned non-JSON, storing as plain text")
        text_blocks.append(TextBlock(
            text=response_text.strip(),
            bbox=BoundingBox(x0=0, y0=0, x1=612, y1=792, page=page_num),
            reading_order=0,
        ))
        return {"text_blocks": text_blocks, "tables": tables, "figures": figures, "warnings": warnings}

    # Parse text blocks
    page_bbox = BoundingBox(x0=0, y0=0, x1=612, y1=792, page=page_num)
    for i, block in enumerate(data.get("text_blocks", [])):
        text_blocks.append(TextBlock(
            text=str(block.get("text", "")).strip(),
            bbox=page_bbox,
            reading_order=i,
            is_heading=bool(block.get("is_heading", False)),
        ))

    # Parse tables
    for t_idx, table in enumerate(data.get("tables", [])):
        raw_headers = table.get("headers", [])
        raw_rows = table.get("rows", [])
        tables.append(TableData(
            caption=table.get("caption"),
            bbox=page_bbox,
            page=page_num,
            headers=[str(h) for h in raw_headers],
            rows=[[str(c) for c in row] for row in raw_rows],
            reading_order=len(text_blocks) + t_idx,
        ))

    # Parse figures
    for f_idx, fig in enumerate(data.get("figures", [])):
        figures.append(FigureBlock(
            figure_id=f"{doc_id}-fig-p{page_num}-{f_idx}",
            bbox=page_bbox,
            page=page_num,
            caption=fig.get("caption"),
            alt_text=fig.get("description"),
            reading_order=len(text_blocks) + len(tables) + f_idx,
        ))

    return {"text_blocks": text_blocks, "tables": tables, "figures": figures, "warnings": warnings}


class VisionExtractor(BaseExtractor):
    """Strategy C — VLM-based extraction for scanned or low-confidence documents.

    Uses OpenRouter API to call Gemini Flash (primary) or GPT-4o-mini (fallback).
    Includes a budget guard that halts extraction if per-document cost would exceed
    the configured cap.
    """

    strategy = ExtractionStrategy.VISION

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml"):
        self.rules_path = rules_path
        self._load_config(rules_path)

    def _load_config(self, rules_path: str) -> None:
        path = Path(rules_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        vision_cfg = data.get("extraction", {}).get("vision", {})
        self.budget_cap = vision_cfg.get("budget_cap_usd", 0.10)
        self.model_primary = vision_cfg.get("model_primary", "google/gemini-flash-1.5")
        self.model_fallback = vision_cfg.get("model_fallback", "openai/gpt-4o-mini")
        self.max_pages_per_call = vision_cfg.get("max_pages_per_call", 5)
        self.prompts_config = data.get("prompts", {})

        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def _call_vlm(
        self,
        image_b64: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> tuple[str, int, int]:
        """Call the VLM API. Returns (response_text, input_tokens, output_tokens)."""
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to your .env file."
            )

        model = model or self.model_primary
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/document-intelligence-refinery",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 3000,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 2000)
        output_tokens = usage.get("completion_tokens", 1500)
        text = data["choices"][0]["message"]["content"]

        return text, input_tokens, output_tokens

    def extract(self, file_path: str, profile: DocumentProfile) -> ExtractionResult:
        """Run Strategy C extraction page by page with budget guard."""
        start = time.perf_counter()
        self._validate_file(file_path)

        all_text_blocks: list[TextBlock] = []
        all_tables: list[TableData] = []
        all_figures: list[FigureBlock] = []
        warnings: list[str] = []
        total_cost = 0.0
        budget_exhausted = False
        budget_exhausted_at_page: Optional[int] = None
        prompt = _build_extraction_prompt(profile.domain_hint, self.prompts_config)

        for page_num in range(1, profile.page_count + 1):
            # Budget guard — check before each page
            if total_cost >= self.budget_cap:
                budget_exhausted = True
                budget_exhausted_at_page = page_num - 1
                warnings.append(
                    f"Budget cap ${self.budget_cap:.3f} reached at page {budget_exhausted_at_page}. "
                    f"Remaining pages ({page_num}–{profile.page_count}) not extracted."
                )
                break

            try:
                image_b64 = _pdf_page_to_base64(file_path, page_num)
            except Exception as e:
                warnings.append(f"Page {page_num}: could not render to image — {e}")
                continue

            # Try primary model, fallback on error
            try:
                response_text, in_tok, out_tok = self._call_vlm(image_b64, prompt)
            except Exception as e:
                warnings.append(f"Page {page_num}: primary model failed ({e}), trying fallback")
                try:
                    response_text, in_tok, out_tok = self._call_vlm(
                        image_b64, prompt, model=self.model_fallback
                    )
                except Exception as e2:
                    warnings.append(f"Page {page_num}: fallback model also failed ({e2}), skipping")
                    continue

            page_cost = _estimate_page_cost(in_tok, out_tok)
            total_cost += page_cost

            parsed = _parse_vlm_response(response_text, page_num, profile.doc_id)
            all_text_blocks.extend(parsed["text_blocks"])
            all_tables.extend(parsed["tables"])
            all_figures.extend(parsed["figures"])
            warnings.extend(parsed["warnings"])

        # Re-index reading_order across all pages
        for i, block in enumerate(all_text_blocks):
            block.reading_order = i

        # Confidence: high if we extracted content, lower if budget was hit early
        pages_processed = sum(
            1 for w in warnings if "Budget cap" not in w or True
        )
        if profile.page_count > 0:
            coverage = min(len(all_text_blocks) / max(profile.page_count, 1), 1.0)
        else:
            coverage = 0.0

        confidence = round(min(0.90 * coverage + 0.10, 1.0), 4)

        elapsed = time.perf_counter() - start

        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            page_count=profile.page_count,
            text_blocks=all_text_blocks,
            tables=all_tables,
            figures=all_figures,
            strategy_used=self.strategy,
            confidence_score=confidence,
            cost_estimate_usd=round(total_cost, 6),
            processing_time_seconds=round(elapsed, 3),
            warnings=warnings,
        )

        # Strategy C never escalates to a higher tier — it's the terminal strategy.
        # However, when the budget was exhausted mid-document the escalation_reason
        # is BUDGET_EXHAUSTED so the router can record it clearly in the ledger.
        return ExtractionResult(
            document=doc,
            escalate=False,
            escalation_reason=EscalationReason.BUDGET_EXHAUSTED if budget_exhausted else None,
            escalation_detail=(
                f"budget ${self.budget_cap:.3f} exhausted at page {budget_exhausted_at_page}"
                if budget_exhausted else None
            ),
        )
