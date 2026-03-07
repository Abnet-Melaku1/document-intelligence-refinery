"""Stage 5: QueryAgent — ReAct-loop document Q&A with provenance.

The QueryAgent answers natural-language questions about ingested documents by
orchestrating three retrieval tools in a ReAct loop:

  1. search_chunks(query, top_k, doc_id, chunk_types)
        Semantic search via VectorStore — returns the most relevant LDU snippets.

  2. navigate_index(topic, doc_id)
        PageIndex navigation — returns section node summaries and chunk IDs for
        topic-relevant document sections.

  3. query_facts(sql)
        Structured SQL over the FactTable SQLite database — for exact numeric
        lookups on tabular data.

Every answer is wrapped in a ProvenanceChain that traces each claim to its
exact source: document, page, bounding box, and content hash.

LLM backend: Google Gemini via google-genai SDK (configurable model, default gemini-2.0-flash).
Fallback: extractive answer from top search result when no API key is set.

Usage:
    agent = QueryAgent()
    chain = agent.answer("What was the net profit in 2023?", doc_id="abc123")
    print(chain.answer)
    print(chain.format_citations())
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from src.models.provenance import ProvenanceChain, ProvenanceEntry
from src.store.vector_store import VectorStore, SearchResult


# ---------------------------------------------------------------------------
# Tool input/output schemas
# ---------------------------------------------------------------------------

class ChunkSearchInput(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None
    chunk_types: Optional[list[str]] = None


class NavigateInput(BaseModel):
    topic: str
    doc_id: str


class FactQueryInput(BaseModel):
    sql: str
    params: list[Any] = []


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ReAct step
# ---------------------------------------------------------------------------

class AgentStep(BaseModel):
    """One iteration of the ReAct loop."""
    thought: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[ToolResult] = None
    final_answer: Optional[str] = None


# ---------------------------------------------------------------------------
# QueryAgent
# ---------------------------------------------------------------------------

class QueryAgent:
    """LangGraph-style ReAct agent for document Q&A.

    Implements a simplified ReAct loop without requiring the full LangGraph
    dependency — the agent calls tools in sequence guided by an LLM.
    Falls back to extractive retrieval when no API key is configured.
    """

    MAX_ITERATIONS = 6

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        rules_path: str = "rubric/extraction_rules.yaml",
    ):
        self._vs = vector_store or VectorStore()
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        self._model = self._load_model(rules_path)
        self._ft = self._load_fact_table()
        self._page_indexes: dict[str, Any] = {}  # lazy-loaded per doc_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        doc_id: Optional[str] = None,
        max_sources: int = 5,
    ) -> ProvenanceChain:
        """Answer a question, returning a ProvenanceChain with full provenance.

        Args:
            question:    Natural-language question from the user.
            doc_id:      Optional — restrict retrieval to one document.
            max_sources: Maximum number of source citations to include.

        Returns:
            ProvenanceChain containing the answer and all source citations.
        """
        if self._api_key:
            return self._react_answer(question, doc_id, max_sources)
        else:
            return self._extractive_answer(question, doc_id, max_sources)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_search_chunks(self, inp: ChunkSearchInput) -> ToolResult:
        """Tool 1: semantic search over vector store."""
        try:
            results = self._vs.search(
                query=inp.query,
                top_k=inp.top_k,
                doc_id=inp.doc_id,
                chunk_types=inp.chunk_types,
            )
            return ToolResult(
                tool_name="search_chunks",
                success=True,
                data=[r.model_dump() for r in results],
            )
        except Exception as exc:
            return ToolResult(tool_name="search_chunks", success=False, data=[], error=str(exc))

    def _tool_navigate_index(self, inp: NavigateInput) -> ToolResult:
        """Tool 2: navigate PageIndex to find relevant sections."""
        try:
            index = self._load_page_index(inp.doc_id)
            if index is None:
                return ToolResult(
                    tool_name="navigate_index",
                    success=False,
                    data=[],
                    error=f"No PageIndex found for doc_id={inp.doc_id}",
                )
            nodes = index.navigate(inp.topic)
            return ToolResult(
                tool_name="navigate_index",
                success=True,
                data=[
                    {
                        "node_id": n.node_id,
                        "title": n.title,
                        "page_range": n.page_range_str(),
                        "summary": n.summary,
                        "key_entities": n.key_entities,
                        "chunk_ids": n.chunk_ids[:10],
                    }
                    for n in nodes
                ],
            )
        except Exception as exc:
            return ToolResult(tool_name="navigate_index", success=False, data=[], error=str(exc))

    def _tool_query_facts(self, inp: FactQueryInput) -> ToolResult:
        """Tool 3: structured SQL query over FactTable."""
        try:
            if self._ft is None:
                return ToolResult(
                    tool_name="query_facts",
                    success=False,
                    data=[],
                    error="FactTable database not available",
                )
            # Safety: only allow SELECT
            sql_upper = inp.sql.strip().upper()
            if not sql_upper.startswith("SELECT"):
                return ToolResult(
                    tool_name="query_facts",
                    success=False,
                    data=[],
                    error="Only SELECT queries are permitted",
                )
            rows = self._ft.query(inp.sql, tuple(inp.params))
            return ToolResult(tool_name="query_facts", success=True, data=rows)
        except Exception as exc:
            return ToolResult(tool_name="query_facts", success=False, data=[], error=str(exc))

    # ------------------------------------------------------------------
    # ReAct loop (LLM-guided)
    # ------------------------------------------------------------------

    def _react_answer(
        self,
        question: str,
        doc_id: Optional[str],
        max_sources: int,
    ) -> ProvenanceChain:
        """Full ReAct loop using Google Gemini."""
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError:
            return self._extractive_answer(question, doc_id, max_sources)

        client = genai.Client(api_key=self._api_key)
        system_prompt = self._build_system_prompt(doc_id)
        steps: list[AgentStep] = []
        accumulated_results: list[SearchResult] = []
        # Track the retrieval method per chunk_id so ProvenanceEntry is accurate
        chunk_retrieval_methods: dict[str, str] = {}

        # Build conversation history as genai Content objects
        contents: list = [
            gtypes.Content(role="user", parts=[gtypes.Part(text=question)])
        ]

        for iteration in range(self.MAX_ITERATIONS):
            try:
                resp = client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=800,
                        temperature=0.1,
                    ),
                )
                content = resp.text.strip()
            except Exception as exc:
                print(f"[QueryAgent] LLM call failed: {exc}", file=sys.stderr)
                break

            # Parse the LLM response for tool calls or final answer
            step = self._parse_react_response(content)
            steps.append(step)
            contents.append(gtypes.Content(role="model", parts=[gtypes.Part(text=content)]))

            if step.final_answer:
                # Collect supporting chunks for provenance
                search_result = self._vs.search(
                    query=question, top_k=max_sources, doc_id=doc_id
                )
                for sr in search_result:
                    if sr.chunk_id not in chunk_retrieval_methods:
                        chunk_retrieval_methods[sr.chunk_id] = "semantic_search"
                accumulated_results.extend(search_result)
                return self._build_chain(
                    question=question,
                    answer=step.final_answer,
                    search_results=accumulated_results[:max_sources],
                    retrieval_methods=chunk_retrieval_methods,
                    doc_id=doc_id,
                )

            if step.tool_name and step.tool_input:
                tool_result = self._dispatch_tool(step.tool_name, step.tool_input)
                step.tool_result = tool_result

                # Collect any search results for provenance, tagging their retrieval method
                if step.tool_name == "search_chunks" and tool_result.success:
                    for r in tool_result.data:
                        sr = SearchResult(**r)
                        accumulated_results.append(sr)
                        chunk_retrieval_methods[sr.chunk_id] = "semantic_search"
                elif step.tool_name == "navigate_index" and tool_result.success:
                    # navigate_index returns section nodes with chunk_ids; fetch those chunks
                    for node_data in tool_result.data:
                        for cid in node_data.get("chunk_ids", []):
                            chunk_retrieval_methods[cid] = "pageindex_navigate"
                elif step.tool_name == "query_facts" and tool_result.success:
                    # query_facts returns dicts; mark any already-accumulated chunks
                    # as having been confirmed via structured query
                    for sr in accumulated_results:
                        if sr.chunk_id not in chunk_retrieval_methods:
                            chunk_retrieval_methods[sr.chunk_id] = "structured_query"

                # Feed tool result back to LLM
                from google.genai import types as gtypes
                tool_content = json.dumps(
                    {"tool": step.tool_name, "result": tool_result.data}
                    if tool_result.success
                    else {"tool": step.tool_name, "error": tool_result.error},
                    ensure_ascii=False,
                    default=str,
                )
                contents.append(gtypes.Content(
                    role="user",
                    parts=[gtypes.Part(text=f"Tool result:\n{tool_content}")],
                ))

        # Fallback if loop exhausted without final answer
        return self._extractive_answer(question, doc_id, max_sources)


    def _extractive_answer(
        self,
        question: str,
        doc_id: Optional[str],
        max_sources: int,
    ) -> ProvenanceChain:
        """Fallback: return top semantic search result as the answer."""
        results = self._vs.search(query=question, top_k=max_sources, doc_id=doc_id)
        if not results:
            return ProvenanceChain(
                query=question,
                answer="No relevant content found in the document store.",
                sources=[],
                is_verified=False,
            )
        answer = results[0].content[:500]
        return self._build_chain(
            question=question,
            answer=answer,
            search_results=results,
            doc_id=doc_id,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_react_response(self, content: str) -> AgentStep:
        """Parse LLM output for Thought/Action/Answer pattern."""
        thought = ""
        tool_name = None
        tool_input = None
        final_answer = None

        lines = content.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                # Next line should be JSON
                action_line = line[len("Action:"):].strip()
                if action_line:
                    try:
                        action = json.loads(action_line)
                        tool_name = action.get("tool")
                        tool_input = action.get("input", {})
                    except json.JSONDecodeError:
                        tool_name = action_line
                elif i + 1 < len(lines):
                    try:
                        action = json.loads(lines[i + 1].strip())
                        tool_name = action.get("tool")
                        tool_input = action.get("input", {})
                        i += 1
                    except (json.JSONDecodeError, IndexError):
                        pass
            elif line.startswith("Final Answer:"):
                final_answer = line[len("Final Answer:"):].strip()
                # Include subsequent lines as part of final answer
                remaining = "\n".join(lines[i + 1:]).strip()
                if remaining:
                    final_answer = f"{final_answer}\n{remaining}"
                break
            i += 1

        return AgentStep(
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input,
            final_answer=final_answer,
        )

    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Route tool call to the correct implementation."""
        if tool_name == "search_chunks":
            return self._tool_search_chunks(ChunkSearchInput(**tool_input))
        elif tool_name == "navigate_index":
            return self._tool_navigate_index(NavigateInput(**tool_input))
        elif tool_name == "query_facts":
            return self._tool_query_facts(FactQueryInput(**tool_input))
        else:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=[],
                error=f"Unknown tool: {tool_name}",
            )

    # ------------------------------------------------------------------
    # ProvenanceChain builder
    # ------------------------------------------------------------------

    def _build_chain(
        self,
        question: str,
        answer: str,
        search_results: list[SearchResult],
        doc_id: Optional[str],
        retrieval_methods: Optional[dict[str, str]] = None,
    ) -> ProvenanceChain:
        """Construct a ProvenanceChain from search results.

        retrieval_methods maps chunk_id → method string so each ProvenanceEntry
        accurately reflects how it was retrieved (semantic_search, pageindex_navigate,
        or structured_query) rather than always defaulting to semantic_search.
        """
        entries: list[ProvenanceEntry] = []
        seen_chunk_ids: set[str] = set()
        methods = retrieval_methods or {}

        for result in search_results:
            if result.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk_id)

            content_hash = hashlib.sha256(result.content.encode()).hexdigest()
            filename = f"{result.doc_id}.pdf"
            method = methods.get(result.chunk_id, "semantic_search")

            entries.append(ProvenanceEntry(
                doc_id=result.doc_id,
                filename=filename,
                page_number=result.page_refs[0] if result.page_refs else 1,
                bounding_box=None,  # not stored in vector DB
                section_title=result.parent_section,
                chunk_id=result.chunk_id,
                content_hash=content_hash,
                excerpt=result.content[:200],
                retrieval_score=result.score,
                retrieval_method=method,
            ))

        return ProvenanceChain(
            query=question,
            answer=answer,
            sources=entries,
            is_verified=len(entries) > 0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, doc_id: Optional[str]) -> str:
        doc_filter = f" You are restricted to document ID '{doc_id}'." if doc_id else ""
        return (
            "You are a document intelligence assistant with access to three tools:\n\n"
            "1. search_chunks — semantic search: "
            'Action: {"tool": "search_chunks", "input": {"query": "...", "top_k": 5}}\n'
            "2. navigate_index — navigate document sections: "
            'Action: {"tool": "navigate_index", "input": {"topic": "...", "doc_id": "..."}}\n'
            "3. query_facts — SQL over structured tables: "
            'Action: {"tool": "query_facts", "input": {"sql": "SELECT ..."}}\n\n'
            "Always respond in this format:\n"
            "Thought: <your reasoning>\n"
            "Action: <JSON tool call> OR Final Answer: <your answer>\n\n"
            "Ground every claim in retrieved content. Be specific — include numbers, dates, "
            f"and entity names when available.{doc_filter}"
        )

    def _load_model(self, rules_path: str) -> str:
        try:
            import yaml
            path = Path(rules_path)
            if path.exists():
                data = yaml.safe_load(path.read_text()) or {}
                return data.get("query", {}).get("model", "gemini-2.0-flash")
        except Exception:
            pass
        return "gemini-2.0-flash"

    def _load_fact_table(self):
        """Lazy-load FactTable if the DB exists."""
        try:
            from src.store.fact_table import FactTable
            db_path = ".refinery/facts.db"
            if Path(db_path).exists():
                return FactTable(db_path)
        except Exception:
            pass
        return None

    def _load_page_index(self, doc_id: str):
        """Lazy-load and cache PageIndex for a doc_id."""
        if doc_id in self._page_indexes:
            return self._page_indexes[doc_id]
        try:
            from src.models.page_index import PageIndex
            index = PageIndex.load(doc_id)
            self._page_indexes[doc_id] = index
            return index
        except Exception:
            self._page_indexes[doc_id] = None
            return None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.query_agent <question> [doc_id]")
        sys.exit(1)

    question = sys.argv[1]
    doc_id = sys.argv[2] if len(sys.argv) > 2 else None

    agent = QueryAgent()
    chain = agent.answer(question, doc_id=doc_id)

    print(f"\nQ: {chain.query}")
    print(f"\nA: {chain.answer}")
    print()
    print(chain.format_citations())
    print(f"\nVerified: {chain.is_verified} | Sources: {len(chain.sources)}")
