"""ProvenanceChain — audit trail for every fact extracted by the Refinery.

Every answer from the Query Interface Agent must include a ProvenanceChain.
This makes the system auditable: any claim can be traced back to its exact
location in the source document (document + page + bounding box + content hash).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .extracted_document import BoundingBox


class ProvenanceEntry(BaseModel):
    """One source citation — points to a specific location in a specific document."""

    # Document identity
    doc_id: str
    filename: str

    # Location
    page_number: int = Field(description="1-indexed page number in the source document")
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Spatial coordinates of the source text/table/figure"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Title of the section this content belongs to"
    )

    # Content snapshot
    chunk_id: str
    content_hash: str = Field(description="SHA256 of source chunk — enables offline verification")
    excerpt: str = Field(description="Short excerpt (≤200 chars) from the source chunk")

    # Retrieval metadata
    retrieval_score: Optional[float] = Field(
        default=None,
        description="Similarity score from vector search (0-1), or None for exact match"
    )
    retrieval_method: str = Field(
        default="semantic_search",
        description="How this source was found: 'pageindex_navigate', 'semantic_search', 'structured_query'"
    )

    def citation_string(self) -> str:
        """Human-readable citation string."""
        loc = f"p.{self.page_number}"
        if self.section_title:
            loc = f"{self.section_title}, {loc}"
        return f"{self.filename} [{loc}]"


class ProvenanceChain(BaseModel):
    """Complete audit trail for one query answer.

    Contains all source citations used to construct the answer,
    in order of relevance.
    """

    query: str = Field(description="The original user question")
    answer: str = Field(description="The generated answer")

    sources: list[ProvenanceEntry] = Field(
        description="All source citations, ordered by relevance"
    )

    # Verification status
    is_verified: bool = Field(
        default=False,
        description="True if every claim in the answer was grounded in a source"
    )
    unverifiable_claims: list[str] = Field(
        default_factory=list,
        description="Claims that could not be grounded in any source"
    )

    # Audit metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None
    total_tokens_used: Optional[int] = None

    def format_citations(self) -> str:
        """Formatted citation block for display."""
        lines = ["**Sources:**"]
        for i, src in enumerate(self.sources, 1):
            bbox_str = ""
            if src.bounding_box:
                bb = src.bounding_box
                bbox_str = f" [bbox: ({bb.x0:.0f},{bb.y0:.0f})–({bb.x1:.0f},{bb.y1:.0f})]"
            lines.append(
                f"  [{i}] {src.citation_string()}{bbox_str}\n"
                f"       hash: {src.content_hash[:12]}…"
            )
        return "\n".join(lines)

    def to_audit_dict(self) -> dict:
        """Serializable dict for the audit ledger."""
        return {
            "query": self.query,
            "answer_preview": self.answer[:100] + "…" if len(self.answer) > 100 else self.answer,
            "source_count": len(self.sources),
            "sources": [
                {
                    "filename": s.filename,
                    "page": s.page_number,
                    "chunk_id": s.chunk_id,
                    "content_hash": s.content_hash,
                    "method": s.retrieval_method,
                }
                for s in self.sources
            ],
            "is_verified": self.is_verified,
            "unverifiable_claims": self.unverifiable_claims,
            "timestamp": self.timestamp.isoformat(),
        }
