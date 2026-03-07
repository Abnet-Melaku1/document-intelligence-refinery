"""Vector store backed by ChromaDB with sentence-transformers embeddings.

Provides semantic search over all ingested LDUs.  Used by the QueryAgent's
`search_chunks` tool to find relevant document sections for a user query.

Storage layout:
    .refinery/chroma/           ChromaDB persistent directory
    Collection: refinery-chunks (cosine similarity space)

Embedding model:
    Default: sentence-transformers all-MiniLM-L6-v2 (local, free, ~90MB)
    Swap:    Set EMBEDDING_MODEL env var and extend _build_ef() for OpenAI / Cohere.

Usage:
    store = VectorStore()
    store.ingest(ldus, doc_id="abc123")
    results = store.search("net profit 2024", top_k=5)
    for r in results:
        print(r.score, r.chunk_id, r.content[:80])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.models.ldu import LDU, ChunkType

# ---------------------------------------------------------------------------
# ChromaDB availability guard
# ---------------------------------------------------------------------------

try:
    import chromadb
    from chromadb.utils import embedding_functions as _ef_module
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# SearchResult schema
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """One result from VectorStore.search()."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_type: str
    page_refs: list[int]
    parent_section: Optional[str]
    section_path: list[str]
    score: float = 0.0  # cosine similarity 0–1, higher = more relevant


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """Persistent ChromaDB-backed vector store for LDU semantic search.

    Thread-safe for read queries; ingest should be called from one thread.
    """

    COLLECTION_NAME = "refinery-chunks"

    def __init__(self, persist_dir: str = ".refinery/chroma"):
        if not _CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is required: pip install chromadb\n"
                "  The default embedding model (all-MiniLM-L6-v2) will be "
                "downloaded automatically on first use (~90 MB)."
            )

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)

        # Default embedding function: sentence-transformers all-MiniLM-L6-v2
        # Local, free, no API key needed.  Auto-downloaded on first use.
        self._ef = _ef_module.DefaultEmbeddingFunction()

        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, ldus: list[LDU], doc_id: str) -> int:
        """Upsert LDUs into the collection.

        Skips FIGURE LDUs that have no alt_text (nothing meaningful to embed).
        Returns the number of chunks actually ingested.
        """
        if not ldus:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for ldu in ldus:
            # Skip purely visual figures with no text description
            if ldu.chunk_type == ChunkType.FIGURE and not ldu.figure_alt_text:
                continue

            ids.append(ldu.chunk_id)
            documents.append(ldu.content)
            metadatas.append({
                "doc_id": ldu.doc_id,
                "chunk_type": ldu.chunk_type.value,
                "page": ldu.page_refs[0] if ldu.page_refs else 0,
                "page_refs": json.dumps(ldu.page_refs),
                "parent_section": ldu.parent_section or "",
                "section_path": json.dumps(ldu.section_path),
                "content_hash": ldu.content_hash,
            })

        if not ids:
            return 0

        # Upsert in batches to avoid memory pressure on large documents
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._col.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        return len(ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Semantic search over all ingested LDUs.

        Args:
            query:       Natural language query string.
            top_k:       Maximum number of results to return.
            doc_id:      Optional document filter — restricts results to one doc.
            chunk_types: Optional list of chunk_type values to include
                         e.g. ["paragraph", "table"]. None means all types.

        Returns:
            List of SearchResult sorted by score descending (most relevant first).
        """
        total = self._col.count()
        if total == 0:
            return []

        # Build ChromaDB where-filter
        where: Optional[dict] = None
        filters: list[dict] = []
        if doc_id:
            filters.append({"doc_id": {"$eq": doc_id}})
        if chunk_types:
            filters.append({"chunk_type": {"$in": chunk_types}})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        n_results = min(top_k, total)

        raw = self._col.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        meta_list = raw.get("metadatas", [[]])[0]
        dist_list = raw.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, content, meta, distance in zip(
            ids_list, docs_list, meta_list, dist_list
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity in [0, 1]
            score = round(max(0.0, 1.0 - distance / 2.0), 4)

            results.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                content=content,
                chunk_type=meta.get("chunk_type", "paragraph"),
                page_refs=json.loads(meta.get("page_refs", "[1]")),
                parent_section=meta.get("parent_section") or None,
                section_path=json.loads(meta.get("section_path", "[]")),
                score=score,
            ))

        # Sort by score descending (ChromaDB returns by distance ascending,
        # but the conversion may reorder ties)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document from the store."""
        self._col.delete(where={"doc_id": {"$eq": doc_id}})

    def count(self) -> int:
        """Total number of chunks currently in the store."""
        return self._col.count()

    def doc_ids(self) -> list[str]:
        """Return the set of distinct doc_ids in the store."""
        if self._col.count() == 0:
            return []
        result = self._col.get(include=["metadatas"])
        ids: set[str] = set()
        for meta in result.get("metadatas", []):
            if meta and meta.get("doc_id"):
                ids.add(meta["doc_id"])
        return sorted(ids)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore()
    print(f"Collection '{VectorStore.COLLECTION_NAME}' — {store.count()} chunks")
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = store.search(query, top_k=5)
        for r in results:
            print(f"  [{r.score:.3f}] {r.chunk_id} p.{r.page_refs} — {r.content[:80]}")
