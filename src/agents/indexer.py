"""Stage 4: PageIndexBuilder — Hierarchical Navigation Index.

Walks the LDU list produced by ChunkingEngine, detects section boundaries
from HEADING chunks, and builds a PageIndex tree where every node carries:
  - page range (start / end)
  - a 2-3 sentence LLM summary (Gemini Flash via OpenRouter)
  - key entities: monetary values, dates, organisations
  - chunk_ids of all LDUs belonging to that section

Fallback (no API key): extractive summary from first 2 sentences of section text.

Output: PageIndex saved to .refinery/pageindex/{doc_id}.json
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from src.models.ldu import LDU, ChunkType
from src.models.page_index import PageIndex, PageIndexNode

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_index_config(rules_path: str = "rubric/extraction_rules.yaml") -> dict:
    path = Path(rules_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
            return data.get("pageindex", {})
    return {}


# ---------------------------------------------------------------------------
# Entity extraction (no LLM — regex-based)
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(
    r'(?:ETB|USD|EUR|GBP|birr)?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|K)?'
    r'|[\$€£]\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|K))?',
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r'\b(?:FY\s*)?(?:19|20)\d{2}(?:/\d{2,4})?'
    r'|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
      r'\s+\d{4}\b'
    r'|\bQ[1-4]\s+\d{4}\b',
    re.IGNORECASE,
)
_ORG_RE = re.compile(
    r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b'
)


def _extract_entities(text: str, max_each: int = 5) -> list[str]:
    """Pull monetary values, dates, and organisation-like proper nouns from text."""
    entities: list[str] = []
    seen: set[str] = set()

    def _add(matches: list[str]) -> None:
        for m in matches[:max_each]:
            m = m.strip()
            if m and m.lower() not in seen and len(m) > 2:
                seen.add(m.lower())
                entities.append(m)

    _add(_MONEY_RE.findall(text))
    _add(_DATE_RE.findall(text))
    _add(_ORG_RE.findall(text))
    return entities[:15]  # cap total


# ---------------------------------------------------------------------------
# Extractive summary fallback
# ---------------------------------------------------------------------------

def _extractive_summary(text: str, max_sentences: int = 2) -> str:
    """First N sentences of the section text (used when no API key is available)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chosen = [s.strip() for s in sentences[:max_sentences] if s.strip()]
    return " ".join(chosen) if chosen else text[:200]


# ---------------------------------------------------------------------------
# LLM summary via Google Gemini
# ---------------------------------------------------------------------------

def _llm_summary(
    section_text: str,
    section_title: str,
    model: str,
    max_tokens: int,
    api_key: str,
) -> str:
    """Call Google Gemini to generate a 2-3 sentence section summary."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        return _extractive_summary(section_text)

    prompt = (
        f"Summarise the following document section titled '{section_title}' "
        f"in exactly 2-3 sentences. Be factual and specific — include key numbers, "
        f"dates, and named entities. Do not start with 'This section'.\n\n"
        f"{section_text[:3000]}"
    )

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            ),
        )
        return resp.text.strip()
    except Exception as exc:
        print(f"[PageIndexBuilder] LLM summary failed ({exc}), using extractive fallback", file=sys.stderr)
        return _extractive_summary(section_text)


# ---------------------------------------------------------------------------
# PageIndexBuilder
# ---------------------------------------------------------------------------

class PageIndexBuilder:
    """Stage 4 agent — builds a PageIndex from a document's LDU list.

    Usage:
        builder = PageIndexBuilder()
        index = builder.run(doc_id, filename, page_count, ldus)
        index.save()

    Config keys read from extraction_rules.yaml (pageindex section):
        summary_model        LLM model for section summaries
        summary_max_tokens   Max tokens per summary response
        navigate_top_k       Top-k sections returned by navigate()
        min_section_chars    Minimum section text length to include in index
    """

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml"):
        cfg = _load_index_config(rules_path)
        self.summary_model = cfg.get("summary_model", "gemini-2.0-flash")
        self.summary_max_tokens = cfg.get("summary_max_tokens", 150)
        self.min_section_chars = cfg.get("min_section_chars", 100)
        self._api_key = os.environ.get("GEMINI_API_KEY", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        doc_id: str,
        filename: str,
        page_count: int,
        ldus: list[LDU],
    ) -> PageIndex:
        """Build and return a PageIndex for the given LDU list.

        Does NOT save to disk — call index.save() after reviewing.
        """
        index = PageIndex(doc_id=doc_id, filename=filename, page_count=page_count)

        # ------------------------------------------------------------------
        # Step 1: Identify heading LDUs and build node skeleton
        # ------------------------------------------------------------------
        # Stack tracks the current ancestor chain: stack[0] = level-1 ancestor, etc.
        stack: list[PageIndexNode] = []
        node_seq = 0

        # Collect (heading_ldu, node) pairs so we can fill content after
        heading_nodes: list[tuple[LDU, PageIndexNode]] = []

        for ldu in ldus:
            if ldu.chunk_type != ChunkType.HEADING:
                continue

            level = ldu.heading_level or 1
            node_id = f"{doc_id}-node-{node_seq:04d}"
            node_seq += 1

            page = ldu.page_refs[0] if ldu.page_refs else 1

            node = PageIndexNode(
                node_id=node_id,
                title=ldu.content.strip(),
                level=level,
                page_start=page,
                page_end=page,  # will extend as we collect child chunks
            )

            # Wire into tree
            # Trim stack to current level
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                parent = stack[-1]
                parent.child_node_ids.append(node_id)
                node.parent_node_id = parent.node_id
            else:
                index.root_node_ids.append(node_id)

            stack.append(node)
            index.nodes[node_id] = node
            heading_nodes.append((ldu, node))

        # ------------------------------------------------------------------
        # Step 2: Assign non-heading LDUs to their section node
        # ------------------------------------------------------------------
        # For each LDU, find its parent node by section_path matching
        section_title_to_node: dict[str, PageIndexNode] = {
            n.title: n for n in index.nodes.values()
        }

        for ldu in ldus:
            if ldu.chunk_type == ChunkType.HEADING:
                continue

            parent_node: Optional[PageIndexNode] = None
            # Match by parent_section title (most specific)
            if ldu.parent_section and ldu.parent_section in section_title_to_node:
                parent_node = section_title_to_node[ldu.parent_section]
            elif ldu.section_path:
                # Fall back to deepest section_path entry that matches a node
                for title in reversed(ldu.section_path):
                    if title in section_title_to_node:
                        parent_node = section_title_to_node[title]
                        break

            if parent_node:
                parent_node.chunk_ids.append(ldu.chunk_id)
                # Extend page_end
                for pg in ldu.page_refs:
                    if pg > parent_node.page_end:
                        parent_node.page_end = pg
                # Also extend all ancestor nodes' page_end
                self._extend_ancestors(index, parent_node, ldu.page_refs)

        # ------------------------------------------------------------------
        # Step 3: Build page_to_nodes reverse index
        # ------------------------------------------------------------------
        for node in index.nodes.values():
            for pg in range(node.page_start, node.page_end + 1):
                index.page_to_nodes.setdefault(pg, [])
                if node.node_id not in index.page_to_nodes[pg]:
                    index.page_to_nodes[pg].append(node.node_id)

        # ------------------------------------------------------------------
        # Step 4: Generate summaries and extract entities
        # ------------------------------------------------------------------
        ldu_by_id = {ldu.chunk_id: ldu for ldu in ldus}

        for heading_ldu, node in heading_nodes:
            # Collect text from all child LDUs
            section_text = self._collect_section_text(node, ldu_by_id)

            if len(section_text) < self.min_section_chars:
                # Too short — use heading as summary
                node.summary = node.title
            elif self._api_key:
                node.summary = _llm_summary(
                    section_text,
                    node.title,
                    self.summary_model,
                    self.summary_max_tokens,
                    self._api_key,
                )
            else:
                node.summary = _extractive_summary(section_text)

            node.key_entities = _extract_entities(section_text)

            # Detect data types present in this section
            section_ldus = [ldu_by_id[cid] for cid in node.chunk_ids if cid in ldu_by_id]
            data_types: set[str] = set()
            for child in section_ldus:
                if child.chunk_type == ChunkType.TABLE:
                    data_types.add("tables")
                elif child.chunk_type == ChunkType.FIGURE:
                    data_types.add("figures")
                elif child.chunk_type == ChunkType.LIST:
                    data_types.add("lists")
            node.data_types_present = sorted(data_types)

        return index

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extend_ancestors(
        self,
        index: PageIndex,
        node: PageIndexNode,
        page_refs: list[int],
    ) -> None:
        """Walk up the parent chain and extend page_end for all ancestors."""
        current = node
        while current.parent_node_id:
            parent = index.nodes.get(current.parent_node_id)
            if not parent:
                break
            for pg in page_refs:
                if pg > parent.page_end:
                    parent.page_end = pg
            current = parent

    def _collect_section_text(
        self,
        node: PageIndexNode,
        ldu_by_id: dict[str, LDU],
    ) -> str:
        """Concatenate text from a node's direct chunk_ids."""
        parts: list[str] = []
        for cid in node.chunk_ids[:20]:  # cap at 20 chunks for summary
            ldu = ldu_by_id.get(cid)
            if ldu and ldu.chunk_type not in (ChunkType.FIGURE,):
                parts.append(ldu.content)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from rich.console import Console
    from rich.tree import Tree as RichTree

    console = Console()

    if len(sys.argv) < 3:
        console.print(
            "[red]Usage: python -m src.agents.indexer "
            "<extracted_doc.json> <ldus.json>[/red]"
        )
        sys.exit(1)

    from src.models.extracted_document import ExtractedDocument
    doc = ExtractedDocument.model_validate_json(Path(sys.argv[1]).read_text())

    ldus_raw = json.loads(Path(sys.argv[2]).read_text())
    from src.models.ldu import LDU as LDUModel
    ldus = [LDUModel.model_validate(item) for item in ldus_raw]

    builder = PageIndexBuilder()
    index = builder.run(doc.doc_id, doc.filename, doc.page_count, ldus)
    saved = index.save()

    # Pretty print the tree
    def _add_nodes(tree: RichTree, node_ids: list[str]) -> None:
        for nid in node_ids:
            node = index.nodes[nid]
            label = (
                f"[cyan]{node.title}[/cyan] "
                f"[dim]{node.page_range_str()}[/dim] "
                f"[yellow]({len(node.chunk_ids)} chunks)[/yellow]"
            )
            branch = tree.add(label)
            _add_nodes(branch, node.child_node_ids)

    rich_tree = RichTree(f"[bold]{index.filename}[/bold] — {len(index.nodes)} sections")
    _add_nodes(rich_tree, index.root_node_ids)
    console.print(rich_tree)
    console.print(f"\n[bold green]✓ PageIndex saved → {saved}[/bold green]")
