"""PageIndex — hierarchical navigation tree over a document (Stage 4).

Inspired by VectifyAI's PageIndex. Gives documents a "smart table of contents"
that an LLM can traverse to locate information without reading the entire document.

Use case: When a user asks "What are the capital expenditure projections for Q3?",
the PageIndex allows the retrieval agent to navigate to the relevant section first,
then do vector search only within that section's chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class PageIndexNode(BaseModel):
    """A single node in the PageIndex tree — represents one document section."""

    node_id: str = Field(description="Unique node ID: {doc_id}-node-{sequence:04d}")
    title: str = Field(description="Section title as it appears in the document")
    level: int = Field(description="Heading level (1=top, 2=sub, 3=subsub, etc.)")

    # Page range this section covers
    page_start: int
    page_end: int

    # LLM-generated summary (2-3 sentences, cheap fast model)
    summary: Optional[str] = None

    # Named entities extracted from this section
    key_entities: list[str] = Field(
        default_factory=list,
        description="Named entities: organizations, dates, monetary values, etc."
    )

    # What types of content are present in this section
    data_types_present: list[str] = Field(
        default_factory=list,
        description="Content types: ['tables', 'figures', 'equations', 'lists']"
    )

    # Chunk IDs of LDUs belonging to this section
    chunk_ids: list[str] = Field(default_factory=list)

    # Tree structure
    parent_node_id: Optional[str] = None
    child_node_ids: list[str] = Field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.child_node_ids) == 0

    def page_range_str(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"pp.{self.page_start}–{self.page_end}"


class PageIndex(BaseModel):
    """Complete navigation index for one document.

    Stored as JSON in .refinery/pageindex/{doc_id}.json.
    """

    doc_id: str
    filename: str
    page_count: int

    # All nodes in the tree, keyed by node_id
    nodes: dict[str, PageIndexNode] = Field(default_factory=dict)

    # Root node IDs (top-level sections)
    root_node_ids: list[str] = Field(default_factory=list)

    # Quick lookup: page number → list of node_ids that cover that page
    page_to_nodes: dict[int, list[str]] = Field(default_factory=dict)

    index_version: str = "1.0.0"

    def get_root_nodes(self) -> list[PageIndexNode]:
        return [self.nodes[nid] for nid in self.root_node_ids if nid in self.nodes]

    def get_children(self, node_id: str) -> list[PageIndexNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.child_node_ids if cid in self.nodes]

    def get_nodes_for_page(self, page: int) -> list[PageIndexNode]:
        node_ids = self.page_to_nodes.get(page, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def navigate(self, topic: str) -> list[PageIndexNode]:
        """Return top-3 most relevant nodes for a topic (keyword match on title + summary).

        In production this should use embedding similarity. This implementation uses
        keyword overlap as a fast, free-tier-friendly fallback.
        """
        topic_words = set(topic.lower().split())
        scored: list[tuple[float, PageIndexNode]] = []

        for node in self.nodes.values():
            score = 0.0
            title_words = set(node.title.lower().split())
            score += len(topic_words & title_words) * 2.0

            if node.summary:
                summary_words = set(node.summary.lower().split())
                score += len(topic_words & summary_words) * 1.0

            for entity in node.key_entities:
                if any(w in entity.lower() for w in topic_words):
                    score += 1.5

            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:3]]

    @classmethod
    def index_path(cls, doc_id: str, base_dir: str = ".refinery/pageindex") -> Path:
        return Path(base_dir) / f"{doc_id}.json"

    def save(self, base_dir: str = ".refinery/pageindex") -> Path:
        path = self.index_path(self.doc_id, base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, doc_id: str, base_dir: str = ".refinery/pageindex") -> "PageIndex":
        path = cls.index_path(doc_id, base_dir)
        return cls.model_validate_json(path.read_text())
