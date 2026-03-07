"""Stage 5 (Audit Mode): ClaimVerifier — ground claims against source documents.

Given a natural-language claim (e.g. "Net profit was ETB 12.4 billion in FY2023"),
the verifier:
  1. Extracts atomic sub-claims (numbers, dates, entities) via regex heuristics.
  2. Searches the VectorStore for supporting evidence.
  3. Classifies each sub-claim as SUPPORTED / CONTRADICTED / UNVERIFIABLE.
  4. Returns an AuditReport with per-claim verdicts and a ProvenanceChain.

LLM backend: OpenRouter (configurable). Falls back to lexical matching when no
API key is configured.

Usage:
    verifier = ClaimVerifier()
    report = verifier.verify(
        claim="Revenue grew 18% to ETB 42 billion in FY2023/24.",
        doc_id="3f8a2c1d9e4b7051",
    )
    print(report.summary())
"""

from __future__ import annotations

import json
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.models.provenance import ProvenanceChain, ProvenanceEntry
from src.store.vector_store import VectorStore, SearchResult
import hashlib


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    SUPPORTED = "supported"           # Evidence clearly confirms the claim
    CONTRADICTED = "contradicted"     # Evidence clearly contradicts the claim
    PARTIALLY_SUPPORTED = "partially_supported"  # Some evidence aligns
    UNVERIFIABLE = "unverifiable"     # No relevant evidence found


# ---------------------------------------------------------------------------
# Sub-claim and report models
# ---------------------------------------------------------------------------

class SubClaim(BaseModel):
    """One atomic assertion extracted from a compound claim."""
    text: str
    verdict: Verdict = Verdict.UNVERIFIABLE
    evidence: list[str] = Field(default_factory=list, description="Supporting/contradicting excerpts")
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0  # 0–1 confidence in the verdict


class AuditReport(BaseModel):
    """Complete audit result for one claim."""

    original_claim: str
    doc_id: Optional[str]
    overall_verdict: Verdict
    overall_confidence: float = Field(ge=0.0, le=1.0)

    sub_claims: list[SubClaim] = Field(default_factory=list)
    provenance: Optional[ProvenanceChain] = None

    # Counts
    supported_count: int = 0
    contradicted_count: int = 0
    unverifiable_count: int = 0

    def summary(self) -> str:
        """Human-readable verdict summary."""
        lines = [
            f"Claim: {self.original_claim}",
            f"Verdict: {self.overall_verdict.value.upper()} "
            f"(confidence: {self.overall_confidence:.0%})",
            f"Sub-claims: {self.supported_count} supported, "
            f"{self.contradicted_count} contradicted, "
            f"{self.unverifiable_count} unverifiable",
        ]
        if self.provenance and self.provenance.sources:
            lines.append("\nTop evidence:")
            for src in self.provenance.sources[:3]:
                lines.append(f"  • {src.citation_string()}: {src.excerpt[:120]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """Verifies natural-language claims against ingested document content.

    Works in two modes:
    - LLM mode (GEMINI_API_KEY set): uses the LLM to judge evidence.
    - Lexical mode (no API key): uses overlap/numeric matching heuristics.
    """

    # Patterns for extracting sub-claims
    _NUMBER_RE = re.compile(
        r'(?:ETB|USD|EUR|GBP|birr)?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|K|%)?'
        r'|[\$€£]\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|K))?',
        re.IGNORECASE,
    )
    _DATE_RE = re.compile(
        r'\b(?:FY\s*)?(?:19|20)\d{2}(?:/\d{2,4})?'
        r'|\b(?:Q[1-4])\s+\d{4}\b',
        re.IGNORECASE,
    )
    _PCT_RE = re.compile(r'\d+(?:\.\d+)?\s*%')

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        rules_path: str = "rubric/extraction_rules.yaml",
    ):
        self._vs = vector_store or VectorStore()
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        self._model = self._load_model(rules_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        claim: str,
        doc_id: Optional[str] = None,
        top_k: int = 8,
    ) -> AuditReport:
        """Verify a claim against the document store.

        Args:
            claim:  The natural-language assertion to verify.
            doc_id: Optional doc filter — restricts evidence search to one document.
            top_k:  Number of evidence chunks to retrieve.

        Returns:
            AuditReport with per-claim verdicts and provenance.
        """
        # Step 1: retrieve evidence
        evidence_chunks = self._vs.search(query=claim, top_k=top_k, doc_id=doc_id)

        # Step 2: extract atomic sub-claims
        sub_claims = self._extract_sub_claims(claim)

        # Step 3: judge each sub-claim
        if self._api_key and evidence_chunks:
            sub_claims = self._llm_judge(claim, sub_claims, evidence_chunks)
        else:
            sub_claims = self._lexical_judge(sub_claims, evidence_chunks)

        # Step 4: aggregate
        supported = sum(1 for sc in sub_claims if sc.verdict == Verdict.SUPPORTED)
        contradicted = sum(1 for sc in sub_claims if sc.verdict == Verdict.CONTRADICTED)
        partial = sum(1 for sc in sub_claims if sc.verdict == Verdict.PARTIALLY_SUPPORTED)
        unverifiable = sum(1 for sc in sub_claims if sc.verdict == Verdict.UNVERIFIABLE)
        total = len(sub_claims) or 1

        if contradicted > 0:
            overall = Verdict.CONTRADICTED
            conf = contradicted / total
        elif supported == total:
            overall = Verdict.SUPPORTED
            conf = 1.0
        elif supported + partial > 0:
            overall = Verdict.PARTIALLY_SUPPORTED
            conf = (supported + partial * 0.5) / total
        else:
            overall = Verdict.UNVERIFIABLE
            conf = 0.0

        # Build provenance
        provenance = self._build_provenance(claim, evidence_chunks)

        return AuditReport(
            original_claim=claim,
            doc_id=doc_id,
            overall_verdict=overall,
            overall_confidence=round(conf, 3),
            sub_claims=sub_claims,
            provenance=provenance,
            supported_count=supported,
            contradicted_count=contradicted,
            unverifiable_count=unverifiable,
        )

    # ------------------------------------------------------------------
    # Sub-claim extraction
    # ------------------------------------------------------------------

    def _extract_sub_claims(self, claim: str) -> list[SubClaim]:
        """Extract atomic numeric/date/entity assertions from a claim."""
        sub_claims: list[SubClaim] = []
        seen: set[str] = set()

        # Numeric values (including percentages and currency)
        for m in self._NUMBER_RE.finditer(claim):
            text = m.group().strip()
            if text and text not in seen and len(text) >= 2:
                seen.add(text)
                # Get context (±20 chars around match)
                start = max(0, m.start() - 20)
                end = min(len(claim), m.end() + 20)
                context = claim[start:end].strip()
                sub_claims.append(SubClaim(text=context))

        # Date references
        for m in self._DATE_RE.finditer(claim):
            text = m.group().strip()
            if text and text not in seen:
                seen.add(text)
                start = max(0, m.start() - 20)
                end = min(len(claim), m.end() + 20)
                context = claim[start:end].strip()
                sub_claims.append(SubClaim(text=context))

        # Whole claim as fallback if nothing extracted
        if not sub_claims:
            sub_claims.append(SubClaim(text=claim))

        return sub_claims

    # ------------------------------------------------------------------
    # LLM judge
    # ------------------------------------------------------------------

    def _llm_judge(
        self,
        original_claim: str,
        sub_claims: list[SubClaim],
        evidence: list[SearchResult],
    ) -> list[SubClaim]:
        """Use LLM to judge each sub-claim against retrieved evidence."""
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError:
            return self._lexical_judge(sub_claims, evidence)

        client = genai.Client(api_key=self._api_key)
        evidence_text = "\n\n".join(
            f"[{r.chunk_id} p.{r.page_refs[0] if r.page_refs else '?'}] {r.content[:300]}"
            for r in evidence[:5]
        )

        for sc in sub_claims:
            prompt = (
                f"Original claim: \"{original_claim}\"\n"
                f"Sub-claim to verify: \"{sc.text}\"\n\n"
                f"Evidence from the document:\n{evidence_text}\n\n"
                "Respond with a JSON object:\n"
                '{"verdict": "supported"|"contradicted"|"partially_supported"|"unverifiable", '
                '"confidence": 0.0-1.0, '
                '"evidence_excerpt": "<relevant quote or empty string>"}'
            )

            try:
                resp = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(
                        max_output_tokens=200,
                        temperature=0.0,
                    ),
                )
                raw = resp.text.strip()

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    sc.verdict = Verdict(parsed.get("verdict", "unverifiable"))
                    sc.confidence = float(parsed.get("confidence", 0.0))
                    excerpt = parsed.get("evidence_excerpt", "")
                    if excerpt:
                        sc.evidence = [excerpt]

            except Exception as exc:
                print(f"[ClaimVerifier] LLM judge failed: {exc}", file=sys.stderr)
                sc.verdict = Verdict.UNVERIFIABLE
                sc.confidence = 0.0

        return sub_claims

    # ------------------------------------------------------------------
    # Lexical judge (no-LLM fallback)
    # ------------------------------------------------------------------

    def _lexical_judge(
        self,
        sub_claims: list[SubClaim],
        evidence: list[SearchResult],
    ) -> list[SubClaim]:
        """Heuristic: check if key terms from sub-claim appear in evidence."""
        evidence_text = " ".join(r.content for r in evidence).lower()

        for sc in sub_claims:
            # Extract numeric tokens from the sub-claim
            numbers = re.findall(r'[\d,]+(?:\.\d+)?', sc.text)
            keywords = [w.lower() for w in sc.text.split() if len(w) > 3]

            num_hits = sum(1 for n in numbers if n.replace(",", "") in evidence_text.replace(",", ""))
            kw_hits = sum(1 for kw in keywords if kw in evidence_text)

            if not numbers and not keywords:
                sc.verdict = Verdict.UNVERIFIABLE
                sc.confidence = 0.0
            elif num_hits >= len(numbers) and numbers:
                sc.verdict = Verdict.SUPPORTED
                sc.confidence = min(0.7, 0.4 + 0.1 * num_hits)
                # Attach top evidence excerpt
                for r in evidence[:2]:
                    if any(n.replace(",", "") in r.content.replace(",", "") for n in numbers):
                        sc.evidence.append(r.content[:200])
                        sc.evidence_chunk_ids.append(r.chunk_id)
            elif kw_hits >= max(1, len(keywords) // 2):
                sc.verdict = Verdict.PARTIALLY_SUPPORTED
                sc.confidence = 0.3
            else:
                sc.verdict = Verdict.UNVERIFIABLE
                sc.confidence = 0.0

        return sub_claims

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _build_provenance(
        self,
        claim: str,
        evidence: list[SearchResult],
    ) -> ProvenanceChain:
        entries = []
        for r in evidence[:5]:
            content_hash = hashlib.sha256(r.content.encode()).hexdigest()
            entries.append(ProvenanceEntry(
                doc_id=r.doc_id,
                filename=f"{r.doc_id}.pdf",
                page_number=r.page_refs[0] if r.page_refs else 1,
                section_title=r.parent_section,
                chunk_id=r.chunk_id,
                content_hash=content_hash,
                excerpt=r.content[:200],
                retrieval_score=r.score,
                retrieval_method="semantic_search",
            ))

        return ProvenanceChain(
            query=claim,
            answer="[audit mode — no generated answer]",
            sources=entries,
            is_verified=len(entries) > 0,
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.agents.audit "<claim>" [doc_id]')
        sys.exit(1)

    claim = sys.argv[1]
    doc_id = sys.argv[2] if len(sys.argv) > 2 else None

    verifier = ClaimVerifier()
    report = verifier.verify(claim, doc_id=doc_id)
    print(report.summary())
    print(f"\nJSON: {report.model_dump_json(indent=2, exclude={'provenance'})}")
