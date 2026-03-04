"""Stage 2: ExtractionRouter — confidence-gated multi-strategy extraction.

ROUTING LOGIC
─────────────
1. Read DocumentProfile → derive the cheapest strategy that *should* work
2. Build a full strategy chain (ordered A→B→C from the starting point)
3. Execute strategies in order; after each, record a StrategyAttempt
4. If a strategy's confidence falls below its threshold, escalate to the next tier
5. When the chain is exhausted, take the best result we have

HUMAN REVIEW FLAG
─────────────────
If the final result still has confidence < human_review_threshold (default 0.50),
`requires_human_review` is set True on the ExtractedDocument and a clear reason is
written. This surfaces documents that need analyst attention — they are not silently
passed downstream with degraded quality.

AUDIT TRAIL
───────────
Every run appends one entry to .refinery/extraction_ledger.jsonl:
  - routing_decision:       why the initial strategy was chosen + key profile signals
  - strategy_attempts:      per-attempt confidence, escalation reason, cost, timing
  - final confidence:        the accepted result's score
  - requires_human_review:  True/False + reason when True
  - escalation_count:        total number of tiers that escalated
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from src.models.document_profile import DocumentProfile, ExtractionCost
from src.models.extracted_document import (
    EscalationReason,
    ExtractedDocument,
    ExtractionStrategy,
    RoutingDecision,
    StrategyAttempt,
)
from src.strategies.base import ExtractionResult
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRATEGY_ORDER: list[ExtractionStrategy] = [
    ExtractionStrategy.FAST_TEXT,
    ExtractionStrategy.LAYOUT,
    ExtractionStrategy.VISION,
]

LEDGER_PATH = Path(".refinery/extraction_ledger.jsonl")

# ---------------------------------------------------------------------------
# Routing decision builder
# ---------------------------------------------------------------------------

def _build_routing_decision(
    profile: DocumentProfile,
    chain: list[ExtractionStrategy],
) -> RoutingDecision:
    """Construct the RoutingDecision explaining why chain[0] was chosen."""
    initial = chain[0]
    cost = profile.estimated_extraction_cost

    if cost == ExtractionCost.NEEDS_VISION_MODEL:
        reason = (
            f"DocumentProfile indicates scanned/image-heavy document "
            f"(avg_char_density={profile.avg_char_density:.6f}, "
            f"scanned_pages={profile.scanned_page_count}/{profile.page_count}). "
            f"Routing directly to Vision — pdfplumber and Docling produce no output on image pages."
        )
    elif cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
        reason = (
            f"DocumentProfile indicates layout complexity requiring structural analysis "
            f"(layout_complexity={profile.layout_complexity.value}, "
            f"origin_type={profile.origin_type.value}). "
            f"Skipping Strategy A — multi-column or table-heavy layouts exceed pdfplumber's capabilities."
        )
    else:  # FAST_TEXT_SUFFICIENT
        reason = (
            f"DocumentProfile indicates clean single-column native digital document "
            f"(origin_type={profile.origin_type.value}, "
            f"layout_complexity={profile.layout_complexity.value}, "
            f"avg_char_density={profile.avg_char_density:.6f}). "
            f"Starting with Strategy A (pdfplumber) — lowest cost, zero API calls."
        )

    return RoutingDecision(
        initial_strategy=initial,
        selection_reason=reason,
        origin_type=profile.origin_type.value,
        layout_complexity=profile.layout_complexity.value,
        estimated_extraction_cost=profile.estimated_extraction_cost.value,
        avg_char_density=profile.avg_char_density,
        avg_image_area_ratio=profile.avg_image_area_ratio,
        scanned_page_count=profile.scanned_page_count,
        strategy_chain=chain,
    )


# ---------------------------------------------------------------------------
# Ledger writer
# ---------------------------------------------------------------------------

def _write_ledger_entry(
    profile: DocumentProfile,
    document: ExtractedDocument,
) -> None:
    """Append one fully-detailed extraction record to the ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    rd = document.routing_decision
    entry: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_id": profile.doc_id,
        "filename": profile.filename,
        "page_count": profile.page_count,

        # Triage context
        "origin_type": profile.origin_type.value,
        "layout_complexity": profile.layout_complexity.value,
        "domain_hint": profile.domain_hint.value,
        "triage_cost_estimate": profile.estimated_extraction_cost.value,

        # Routing decision
        "routing": {
            "initial_strategy": rd.initial_strategy.value if rd else None,
            "selection_reason": rd.selection_reason if rd else None,
            "strategy_chain": [s.value for s in rd.strategy_chain] if rd else [],
        },

        # Per-attempt breakdown — the core audit trail
        "strategy_attempts": [
            {
                "strategy": a.strategy.value,
                "confidence_score": a.confidence_score,
                "escalated": a.escalated,
                "escalation_reason": a.escalation_reason.value if a.escalation_reason else None,
                "escalation_detail": a.escalation_detail,
                "cost_usd": a.cost_usd,
                "processing_time_seconds": a.processing_time_seconds,
            }
            for a in document.strategy_attempts
        ],

        # Final accepted result
        "strategy_used": document.strategy_used.value,
        "confidence_score": document.confidence_score,
        "escalation_count": document.escalation_count,
        "cost_estimate_usd": document.cost_estimate_usd,
        "processing_time_seconds": document.processing_time_seconds,

        # Content summary
        "text_block_count": len(document.text_blocks),
        "table_count": document.table_count,
        "figure_count": document.figure_count,
        "warning_count": len(document.warnings),

        # Human review flag — the critical signal for downstream triage
        "requires_human_review": document.requires_human_review,
        "human_review_reason": document.human_review_reason,
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ExtractionRouter:
    """Routes documents to the appropriate extraction strategy and orchestrates
    confidence-gated escalation through A → B → C.

    Every routing decision and every strategy attempt is recorded in the
    ExtractedDocument itself and in extraction_ledger.jsonl so the full
    decision trail is auditable without re-running the pipeline.
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
        self.human_review_threshold = self._load_human_review_threshold(rules_path)

        if ledger_path:
            global LEDGER_PATH
            LEDGER_PATH = Path(ledger_path)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @staticmethod
    def _load_human_review_threshold(rules_path: str) -> float:
        """Load the human review confidence floor from config.

        When the final strategy's confidence is below this value, the document
        is flagged for human review. This is the last safety net when even
        Vision underperforms.
        """
        path = Path(rules_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return data.get("extraction", {}).get("human_review_threshold", 0.50)
        return 0.50

    # ------------------------------------------------------------------
    # Strategy chain construction
    # ------------------------------------------------------------------

    def _build_chain(
        self,
        profile: DocumentProfile,
        force_strategy: Optional[ExtractionStrategy],
    ) -> list[ExtractionStrategy]:
        """Return the ordered list of strategies to attempt.

        The chain always runs from a starting point to the terminal (Vision).
        force_strategy overrides the triage recommendation but keeps the tail.
        """
        if force_strategy is not None:
            start_idx = _STRATEGY_ORDER.index(force_strategy)
        else:
            cost = profile.estimated_extraction_cost
            if cost == ExtractionCost.FAST_TEXT_SUFFICIENT:
                start_idx = 0  # A → B → C
            elif cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
                start_idx = 1  # B → C
            else:
                start_idx = 2  # C only

        return _STRATEGY_ORDER[start_idx:]

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _run_strategy(
        self,
        strategy: ExtractionStrategy,
        file_path: str,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        if strategy == ExtractionStrategy.FAST_TEXT:
            return self.fast_text.extract(file_path, profile)
        elif strategy == ExtractionStrategy.LAYOUT:
            return self.layout.extract(file_path, profile)
        else:
            return self.vision.extract(file_path, profile)

    # ------------------------------------------------------------------
    # Human review evaluation
    # ------------------------------------------------------------------

    def _evaluate_human_review(self, document: ExtractedDocument) -> None:
        """Set requires_human_review=True when confidence is below the floor.

        Mutates document in place. Called after the full escalation chain has
        been exhausted so we are evaluating the best result available.
        """
        conf = document.confidence_score
        if conf >= self.human_review_threshold:
            return

        document.requires_human_review = True
        attempts_summary = " → ".join(
            f"{a.strategy.value}({a.confidence_score:.2f})"
            for a in document.strategy_attempts
        )
        document.human_review_reason = (
            f"Final confidence {conf:.4f} is below human_review_threshold "
            f"{self.human_review_threshold} after exhausting all strategies. "
            f"Attempt trace: [{attempts_summary}]. "
            f"Possible causes: extreme scan degradation, unusual layout, "
            f"non-standard encoding, or content requiring specialist interpretation."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        file_path: str,
        profile: DocumentProfile,
        force_strategy: Optional[ExtractionStrategy] = None,
    ) -> ExtractedDocument:
        """Run extraction with confidence-gated escalation.

        Args:
            file_path: Path to the PDF.
            profile: DocumentProfile from TriageAgent.
            force_strategy: Override triage recommendation (useful for testing
                            or manual re-runs). The escalation chain still applies
                            from the forced strategy onward.

        Returns:
            ExtractedDocument with:
              - routing_decision: why the initial strategy was chosen
              - strategy_attempts: every attempt with confidence + escalation trace
              - requires_human_review: True if final confidence < human_review_threshold
              - human_review_reason: detailed explanation when flagged
        """
        chain = self._build_chain(profile, force_strategy)
        routing_decision = _build_routing_decision(profile, chain)

        attempts: list[StrategyAttempt] = []
        last_result: Optional[ExtractionResult] = None

        for strategy in chain:
            t0 = time.perf_counter()

            try:
                result = self._run_strategy(strategy, file_path, profile)
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 3)
                attempts.append(StrategyAttempt(
                    strategy=strategy,
                    confidence_score=0.0,
                    escalated=True,
                    escalation_reason=EscalationReason.EXCEPTION_RAISED,
                    escalation_detail=str(exc),
                    cost_usd=0.0,
                    processing_time_seconds=elapsed,
                ))
                # Keep any prior result; try the next strategy
                continue

            elapsed = round(time.perf_counter() - t0, 3)

            attempts.append(StrategyAttempt(
                strategy=strategy,
                confidence_score=result.document.confidence_score,
                escalated=result.escalate,
                escalation_reason=result.escalation_reason,
                escalation_detail=result.escalation_detail,
                cost_usd=result.document.cost_estimate_usd,
                processing_time_seconds=result.document.processing_time_seconds or elapsed,
            ))
            last_result = result

            if not result.escalate:
                break  # Confidence is acceptable — stop here

        # ------------------------------------------------------------------
        # Assemble the final ExtractedDocument
        # ------------------------------------------------------------------
        if last_result is None:
            final_doc = ExtractedDocument(
                doc_id=profile.doc_id,
                filename=profile.filename,
                page_count=profile.page_count,
                strategy_used=ExtractionStrategy.FAST_TEXT,
                confidence_score=0.0,
                warnings=["All extraction strategies raised exceptions"],
            )
        else:
            final_doc = last_result.document

        # Attach full routing provenance
        final_doc.routing_decision = routing_decision
        final_doc.strategy_attempts = attempts

        # Last-resort safety net
        self._evaluate_human_review(final_doc)

        # Write complete audit entry to ledger
        _write_ledger_entry(profile, final_doc)

        return final_doc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    from src.agents.triage import TriageAgent

    console = Console()

    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.agents.extractor <path/to/document.pdf>[/red]")
        sys.exit(1)

    file_path = sys.argv[1]

    # --- Triage ---
    console.print(f"\n[bold cyan]▶ Stage 1: Triage[/bold cyan]  {file_path}")
    triage = TriageAgent()
    profile = triage.run(file_path)
    profile.save()
    console.print(
        f"  origin=[green]{profile.origin_type.value}[/green]  "
        f"layout=[green]{profile.layout_complexity.value}[/green]  "
        f"domain=[green]{profile.domain_hint.value}[/green]  "
        f"→ cost=[yellow]{profile.estimated_extraction_cost.value}[/yellow]"
    )

    # --- Extract ---
    console.print(f"\n[bold cyan]▶ Stage 2: Extraction[/bold cyan]")
    router = ExtractionRouter()
    doc = router.extract(file_path, profile)

    # Routing decision panel
    if doc.routing_decision:
        rd = doc.routing_decision
        console.print(Panel(
            f"[bold]Initial strategy:[/bold] {rd.initial_strategy.value}\n"
            f"[bold]Chain:[/bold] {' → '.join(s.value for s in rd.strategy_chain)}\n\n"
            f"[bold]Reason:[/bold] {rd.selection_reason}",
            title="Routing Decision",
            border_style="blue",
        ))

    # Per-attempt table
    tbl = RichTable(title="Strategy Attempts", show_header=True, header_style="bold magenta")
    tbl.add_column("#")
    tbl.add_column("Strategy")
    tbl.add_column("Confidence")
    tbl.add_column("Escalated?")
    tbl.add_column("Reason / Detail")
    tbl.add_column("Cost")
    tbl.add_column("Time")

    for i, a in enumerate(doc.strategy_attempts, 1):
        esc_text = "[red]YES[/red]" if a.escalated else "[green]NO ✓[/green]"
        detail = a.escalation_detail or "—"
        tbl.add_row(
            str(i),
            a.strategy.value,
            f"{a.confidence_score:.4f}",
            esc_text,
            detail[:55] + ("…" if len(detail) > 55 else ""),
            f"${a.cost_usd:.6f}",
            f"{a.processing_time_seconds}s",
        )
    console.print(tbl)

    # Final summary
    res = RichTable(title="Final Result", show_header=True)
    res.add_column("Metric", style="cyan")
    res.add_column("Value", style="green")
    res.add_row("strategy_used", doc.strategy_used.value)
    res.add_row("confidence_score", f"{doc.confidence_score:.4f}")
    res.add_row("escalation_count", str(doc.escalation_count))
    res.add_row("cost_estimate_usd", f"${doc.cost_estimate_usd:.6f}")
    res.add_row("text_blocks", str(len(doc.text_blocks)))
    res.add_row("tables", str(doc.table_count))
    res.add_row("figures", str(doc.figure_count))
    res.add_row("warnings", str(len(doc.warnings)))
    res.add_row("ledger", str(LEDGER_PATH))
    console.print(res)

    # Human review flag — impossible to miss
    if doc.requires_human_review:
        console.print(Panel(
            Text(f"⚠  HUMAN REVIEW REQUIRED\n\n{doc.human_review_reason}", style="bold red"),
            border_style="red",
        ))
    else:
        console.print(
            f"\n[green]✓ Accepted — confidence {doc.confidence_score:.4f} ≥ "
            f"human_review_threshold {router.human_review_threshold}[/green]"
        )

    if doc.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in doc.warnings[:5]:
            console.print(f"  • {w}")
        if len(doc.warnings) > 5:
            console.print(f"  … and {len(doc.warnings) - 5} more (see ledger)")
