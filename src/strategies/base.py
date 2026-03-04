"""Base extractor interface — the shared contract all three strategies implement.

Every strategy must:
1. Accept a file path and DocumentProfile
2. Return an ExtractedDocument with confidence_score and cost_estimate_usd
3. Never raise on a single-page failure — add a warning and continue

This is the Strategy Pattern: ExtractionRouter selects the concrete strategy
based on DocumentProfile, but always calls the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument, ExtractionStrategy


class ExtractionResult:
    """Thin wrapper returned by all extractors."""
    def __init__(self, document: ExtractedDocument, escalate: bool = False):
        self.document = document
        self.escalate = escalate  # True → router should try the next strategy tier


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
            ExtractionResult with the extracted document and escalation flag.
        """
        ...

    def _validate_file(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")
        return path
