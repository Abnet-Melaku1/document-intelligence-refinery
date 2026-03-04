"""Stage 2: ExtractionRouter — confidence-gated multi-strategy extraction.

The router reads the DocumentProfile produced by Stage 1 (TriageAgent) and
selects the appropriate extraction strategy. If the selected strategy's
confidence falls below the escalation threshold, it automatically retries
with the next more expensive strategy.

Escalation path: Strategy A → Strategy B → Strategy C (terminal)

Every extraction is logged to .refinery/extraction_ledger.jsonl with:
  - strategy_used, confidence_score, cost_estimate_usd, processing_time_seconds
  - escalation_count (how many strategies were tried before success)

This ledger is the audit trail for extraction quality across the corpus.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.document_profile import DocumentProfile, ExtractionCost
from src.models.extracted_document import ExtractedDocument, ExtractionStrategy
from src.strategies.base import ExtractionResult
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor


LEDGER_PATH = Path(".refinery/extraction_ledger.jsonl")


def _write_ledger_entry(
    profile: DocumentProfile,
    document: ExtractedDocument,
    escalation_count: int,
    strategies_tried: list[str],
) -> None:
    """Append one extraction record to the ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "doc_id": profile.doc_id,
        "filename": profile.filename,
        "page_count": profile.page_count,
        "origin_type": profile.origin_type.value,
        "layout_complexity": profile.layout_complexity.value,
        "domain_hint": profile.domain_hint.value,
        "triage_cost_estimate": profile.estimated_extraction_cost.value,
        "strategies_tried": strategies_tried,
        "strategy_used": document.strategy_used.value,
        "confidence_score": document.confidence_score,
        "cost_estimate_usd": document.cost_estimate_usd,
        "processing_time_seconds": document.processing_time_seconds,
        "escalation_count": escalation_count,
        "text_block_count": len(document.text_blocks),
        "table_count": document.table_count,
        "figure_count": document.figure_count,
        "warning_count": len(document.warnings),
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class ExtractionRouter:
    """Routes documents to the appropriate extraction strategy.

    Selection logic:
      1. Start with the strategy recommended by the DocumentProfile (triage output)
      2. If that strategy's confidence < threshold → escalate to next tier
      3. Log every attempt and final result to extraction_ledger.jsonl

    The router implements the Strategy Pattern: each extractor is interchangeable
    behind the BaseExtractor interface.
    """

    def __init__(
        self,
        rules_path: str = "rubric/extraction_rules.yaml",
        ledger_path: Optional[str] = None,
    ):
        self.rules_path = rules_path
        self.fast_text = FastTextExtractor(rules_path)
        self.layout = LayoutExtractor(rules_path)
        self.vision = VisionExtractor(rules_path)

        if ledger_path:
            global LEDGER_PATH
            LEDGER_PATH = Path(ledger_path)

    def _initial_strategy(
        self, profile: DocumentProfile
    ) -> list[ExtractionStrategy]:
        """Return ordered list of strategies to try based on DocumentProfile.

        The first strategy is the cheapest one that should work for this profile.
        The list is the full escalation chain from that point onward.
        """
        cost = profile.estimated_extraction_cost

        if cost == ExtractionCost.FAST_TEXT_SUFFICIENT:
            return [
                ExtractionStrategy.FAST_TEXT,
                ExtractionStrategy.LAYOUT,
                ExtractionStrategy.VISION,
            ]
        elif cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            return [
                ExtractionStrategy.LAYOUT,
                ExtractionStrategy.VISION,
            ]
        else:  # NEEDS_VISION_MODEL
            return [ExtractionStrategy.VISION]

    def _run_strategy(
        self,
        strategy: ExtractionStrategy,
        file_path: str,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        """Dispatch to the correct extractor."""
        if strategy == ExtractionStrategy.FAST_TEXT:
            return self.fast_text.extract(file_path, profile)
        elif strategy == ExtractionStrategy.LAYOUT:
            return self.layout.extract(file_path, profile)
        else:
            return self.vision.extract(file_path, profile)

    def extract(
        self,
        file_path: str,
        profile: DocumentProfile,
        force_strategy: Optional[ExtractionStrategy] = None,
    ) -> ExtractedDocument:
        """Run extraction with automatic confidence-gated escalation.

        Args:
            file_path: Path to the PDF.
            profile: DocumentProfile from TriageAgent.
            force_strategy: Override automatic strategy selection (for testing).

        Returns:
            ExtractedDocument from the best strategy that ran.
        """
        if force_strategy:
            strategy_chain = [force_strategy]
            # Still allow escalation unless it's the terminal strategy
            if force_strategy != ExtractionStrategy.VISION:
                remaining = [
                    s for s in [
                        ExtractionStrategy.FAST_TEXT,
                        ExtractionStrategy.LAYOUT,
                        ExtractionStrategy.VISION,
                    ]
                    if s != force_strategy
                ]
                strategy_chain.extend(remaining)
        else:
            strategy_chain = self._initial_strategy(profile)

        strategies_tried: list[str] = []
        last_result: Optional[ExtractionResult] = None
        escalation_count = 0

        for strategy in strategy_chain:
            strategies_tried.append(strategy.value)

            try:
                result = self._run_strategy(strategy, file_path, profile)
            except Exception as e:
                # Strategy failed entirely — escalate with warning
                if last_result:
                    last_result.document.warnings.append(
                        f"{strategy.value} extraction raised an exception: {e}"
                    )
                escalation_count += 1
                continue

            last_result = result

            if not result.escalate:
                # Confidence is acceptable — use this result
                break

            # Escalate: strategy's confidence was too low
            escalation_count += 1
            strategy_name = strategy.value
            conf = result.document.confidence_score
            next_strategies = strategy_chain[strategy_chain.index(strategy) + 1:]

            if next_strategies:
                result.document.warnings.append(
                    f"Strategy {strategy_name} confidence {conf:.2f} below threshold — "
                    f"escalating to {next_strategies[0].value}"
                )
            # Continue to next strategy in chain

        # Always return something — last result is best we have
        final_doc = last_result.document if last_result else ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            page_count=profile.page_count,
            strategy_used=ExtractionStrategy.FAST_TEXT,
            confidence_score=0.0,
            warnings=["All extraction strategies failed"],
        )

        _write_ledger_entry(profile, final_doc, escalation_count, strategies_tried)

        return final_doc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.table import Table as RichTable

    from src.agents.triage import TriageAgent

    console = Console()

    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.agents.extractor <path/to/document.pdf>[/red]")
        sys.exit(1)

    file_path = sys.argv[1]
    console.print(f"[cyan]Triaging:[/cyan] {file_path}")

    triage = TriageAgent()
    profile = triage.run(file_path)
    profile.save()

    console.print(f"[green]Profile:[/green] {profile.origin_type.value} / "
                  f"{profile.layout_complexity.value} / {profile.domain_hint.value}")
    console.print(f"[cyan]Extracting (recommended: {profile.estimated_extraction_cost.value})...[/cyan]")

    router = ExtractionRouter()
    doc = router.extract(file_path, profile)

    table = RichTable(title=f"Extraction Result — {doc.filename}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("strategy_used", doc.strategy_used.value)
    table.add_row("confidence_score", f"{doc.confidence_score:.4f}")
    table.add_row("cost_estimate_usd", f"${doc.cost_estimate_usd:.6f}")
    table.add_row("processing_time", f"{doc.processing_time_seconds}s")
    table.add_row("text_blocks", str(len(doc.text_blocks)))
    table.add_row("tables", str(doc.table_count))
    table.add_row("figures", str(doc.figure_count))
    table.add_row("warnings", str(len(doc.warnings)))
    table.add_row("ledger", str(LEDGER_PATH))

    console.print(table)

    if doc.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in doc.warnings[:5]:
            console.print(f"  • {w}")
