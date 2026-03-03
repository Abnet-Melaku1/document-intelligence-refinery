"""LDU — Logical Document Unit, output of Stage 3 Semantic Chunking Engine.

LDUs are RAG-ready chunks that respect document semantic boundaries.
Each LDU is self-contained and carries full spatial provenance.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .extracted_document import BoundingBox


class ChunkType(str, Enum):
    """The semantic type of content in this LDU."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"           # Numbered or bulleted list kept as single unit
    CAPTION = "caption"     # Figure/table caption (stored as figure metadata, not standalone)
    CODE = "code"
    FOOTNOTE = "footnote"


class LDURelationship(BaseModel):
    """A resolved cross-reference between two LDUs."""
    target_chunk_id: str
    relationship_type: str = Field(description="e.g. 'see_also', 'references_table', 'references_figure'")
    anchor_text: str = Field(description="The text that contained the cross-reference")


class LDU(BaseModel):
    """A Logical Document Unit — one semantically coherent, RAG-ready chunk.

    Every LDU carries:
    - Content with its semantic type
    - Full spatial provenance (page, bounding box)
    - Section hierarchy (parent context)
    - A content hash for deduplication and audit
    - Cross-reference relationships to other LDUs
    """

    chunk_id: str = Field(description="Unique identifier: {doc_id}-{sequence:04d}")
    doc_id: str

    # Content
    content: str
    chunk_type: ChunkType
    token_count: int = Field(description="tiktoken cl100k_base token count")

    # Spatial provenance
    page_refs: list[int] = Field(description="Page numbers this chunk spans (1-indexed)")
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Spatial coordinates. None for multi-page spanning chunks."
    )

    # Structural context
    parent_section: Optional[str] = Field(
        default=None,
        description="Title of the containing section (stored on all child chunks)"
    )
    section_path: list[str] = Field(
        default_factory=list,
        description="Full section hierarchy path e.g. ['3. Results', '3.2 Analysis']"
    )
    heading_level: Optional[int] = None

    # For table chunks: the structured data
    table_headers: Optional[list[str]] = None
    table_rows: Optional[list[list[str]]] = None

    # For figure chunks: caption metadata
    figure_caption: Optional[str] = None
    figure_alt_text: Optional[str] = None

    # Cross-references resolved by the chunker
    relationships: list[LDURelationship] = Field(default_factory=list)

    # Integrity
    content_hash: str = Field(
        description="SHA256 of content — provenance anchor stable across document versions"
    )

    @model_validator(mode="before")
    @classmethod
    def compute_content_hash(cls, values: dict) -> dict:
        if "content_hash" not in values or not values.get("content_hash"):
            content = values.get("content", "")
            values["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
        return values

    @classmethod
    def make_chunk_id(cls, doc_id: str, sequence: int) -> str:
        return f"{doc_id}-{sequence:04d}"
