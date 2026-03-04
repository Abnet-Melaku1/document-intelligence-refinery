"""Base extractor interface — the shared contract all three strategies implement.

Every strategy must:
1. Accept a file path and DocumentProfile
2. Return an ExtractionResult with the document and a structured escalation signal
3. Never raise on a single-page failure — add a warning and continue

This is the Strategy Pattern: ExtractionRouter selects the concrete strategy
based on DocumentProfile, but always calls the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import (
    EscalationReason,
    ExtractedDocument,
    ExtractionStrategy,
)


class ExtractionResult:
    """Structured result returned by every extractor.

    The `escalate` flag tells the router whether to try the next strategy tier.
    When escalate=True, `escalation_reason` and `escalation_detail` explain why —
    this information is recorded in the ledger and attached to the ExtractedDocument
    so the full routing trail is auditable.
    """

    def __init__(
        self,
        document: ExtractedDocument,
        escalate: bool = False,
        escalation_reason: Optional[EscalationReason] = None,
        escalation_detail: Optional[str] = None,
    ):
        self.document = document
        self.escalate = escalate
        # Reason is required when escalating — default to LOW_CONFIDENCE if not provided
        self.escalation_reason: Optional[EscalationReason] = (
            escalation_reason if escalate else None
        )
        if escalate and self.escalation_reason is None:
            self.escalation_reason = EscalationReason.LOW_CONFIDENCE
        self.escalation_detail: Optional[str] = escalation_detail if escalate else None


class BaseExtractor(ABC):
    """Abstract base for all extraction strategies."""

    strategy: ExtractionStrategy  # Must be set by each subclass

    @abstractmethod
    def extract(self, file_path: str, profile: DocumentProfile) -> ExtractionResult:
        """Extract content from the document.

        Args:
            file_path: Path to the PDF file.
            profile: DocumentProfile produced by the Triage Agent.

        Returns:
            ExtractionResult with the document and structured escalation signal.
            escalate=True means confidence was insufficient — router should try next tier.
        """
        ...

    def _validate_file(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")
        return path
