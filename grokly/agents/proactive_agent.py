"""
grokly/agents/proactive_agent.py — The Proactive Agent.

Surfaces unsolicited insights after every answer. Each role gets a
completely different search strategy — not the same data reformatted:

  developer    → call_graph + raw_code: callers, callees, sibling functions
  business_user → docs + forum: related business processes (no function names)
  manager      → docs + forum: governance, approval rules, audit context
  uat_tester   → commentary + forum: validation functions and failure scenarios
  end_user     → docs + forum: simple next steps in plain language

Cross-cutting behaviours run for every role:
  gap_alert  — low-confidence warning with role-appropriate wording
  staleness  — warns when ingested chunks are > 30 days old
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from dotenv import load_dotenv

from grokly.store.chroma_store import ChromaStore

load_dotenv()
logger = logging.getLogger(__name__)

_LOW_CONFIDENCE = 0.60
_VERY_LOW_CONFIDENCE = 0.35
_STALENESS_DAYS = 30

_STOPWORDS = {
    "what", "does", "how", "when", "where", "which", "about", "with",
    "have", "from", "that", "this", "will", "would", "could", "should",
    "tell", "show", "give", "explain", "list", "find", "make", "into",
    "work", "used", "using", "call", "called", "submit", "create",
}


class ProactiveAgent:
    def __init__(self, store: ChromaStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        question: str,
        answer: str,
        role: str,
        confidence: float,
        sources: list,
        retrieved_chunks: list,
    ) -> dict:
        try:
            if role == "developer":
                suggestions = self._analyse_for_developer(
                    question, answer, confidence, retrieved_chunks)
            elif role == "business_user":
                suggestions = self._analyse_for_business_user(
                    question, answer, confidence)
            elif role == "manager":
                suggestions = self._analyse_for_manager(
                    question, answer, confidence)
            elif role == "uat_tester":
                suggestions = self._analyse_for_uat_tester(
                    question, answer, confidence)
            elif role == "end_user":
                suggestions = self._analyse_for_end_user(
                    question, answer, confidence)
            else:
                suggestions = []
        except Exception as exc:
            logger.warning("Proactive agent error (%s): %s", role, exc)
            suggestions = []

        gap = self._check_knowledge_gap(confidence, question, role)
        stale = self._check_staleness(retrieved_chunks)

        return {
            "gap_alert": gap,
            "related": {
                "triggered": len(suggestions) >= 1,
                "suggestions": suggestions,
            },
            "staleness": stale,
            "has_insights": bool(
                gap.get("triggered")
                or len(suggestions) >= 1
                or stale.get("triggered")
            ),
        }

    # ------------------------------------------------------------------
    # Role-specific analysers
    # ------------------------------------------------------------------

    def _analyse_for_developer(
        self,
        question: str,
        answer: str,
        confidence: float,
        retrieved_chunks: list,
    ) -> list[dict]:
        """Callers, callees, and sibling functions from call_graph + raw_code."""
        fn = _extract_function_name(question)
        if not fn:
            return []

        suggestions: list[dict] = []
        seen: set[str] = {fn}

        # Call-graph neighbours
        try:
            cg_chunks = self.store.query(
                fn, n_results=6,
                where={"chunk_type": {"$eq": "call_graph"}},
            )
            for c in cg_chunks:
                meta = c.get("metadata", {})
                neighbour = meta.get("function_name", "")
                if not neighbour or neighbour in seen:
                    continue
                seen.add(neighbour)
                text = c["text"]
                rel_type = "caller" if fn in text.split("calls")[-1:] else "callee"
                suggestions.append({
                    "type":        rel_type,
                    "label":       "Called by" if rel_type == "caller" else "Also calls",
                    "function":    neighbour,
                    "description": text[:80].replace("\n", " "),
                    "prompt":      f"What does {neighbour} do?",
                })
        except Exception as exc:
            logger.debug("Developer call-graph search failed: %s", exc)

        # Sibling functions in the same module
        module = next(
            (c.get("metadata", {}).get("module") for c in retrieved_chunks
             if c.get("metadata", {}).get("module")),
            None,
        )
        if module and len(suggestions) < 3:
            try:
                mod_chunks = self.store.query(
                    module, n_results=8,
                    where={"chunk_type": {"$eq": "commentary"}},
                )
                for c in mod_chunks:
                    meta = c.get("metadata", {})
                    sibling = meta.get("function_name", "")
                    if not sibling or sibling in seen:
                        continue
                    seen.add(sibling)
                    suggestions.append({
                        "type":        "sibling",
                        "label":       "Same module",
                        "function":    sibling,
                        "description": c["text"][:80].replace("\n", " "),
                        "prompt":      f"What does {sibling} do?",
                    })
            except Exception as exc:
                logger.debug("Developer sibling search failed: %s", exc)

        return suggestions[:3]

    def _analyse_for_business_user(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Related business processes from docs and forum — no function names."""
        topic = _extract_business_topic(question)
        if not topic:
            return []

        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        for chunk_type in ("docs", "forum"):
            try:
                chunks = self.store.query(
                    f"related process {topic}",
                    n_results=5,
                    where={"chunk_type": {"$eq": chunk_type}},
                )
                for c in chunks:
                    meta = c.get("metadata", {})
                    title = _derive_display_title(c, meta)
                    if not title or title in seen_topics:
                        continue
                    seen_topics.add(title)
                    desc = c["text"][:100].replace("\n", " ")
                    suggestions.append({
                        "type":        "related_process",
                        "label":       "Related process",
                        "topic":       title,
                        "description": desc,
                        "prompt":      f"Tell me about {title}",
                    })
            except Exception as exc:
                logger.debug("Business user search failed (%s): %s", chunk_type, exc)

            if len(suggestions) >= 3:
                break

        return suggestions[:3]

    def _analyse_for_manager(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Governance, approval rules, and audit context from docs and forum."""
        topic = _extract_business_topic(question)
        governance_query = f"approval limit delegation audit {topic}" if topic else "approval workflow audit"

        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        for chunk_type in ("docs", "forum"):
            try:
                chunks = self.store.query(
                    governance_query,
                    n_results=5,
                    where={"chunk_type": {"$eq": chunk_type}},
                )
                for c in chunks:
                    meta = c.get("metadata", {})
                    title = _derive_display_title(c, meta)
                    if not title or title in seen_topics:
                        continue
                    seen_topics.add(title)
                    desc = c["text"][:100].replace("\n", " ")
                    suggestions.append({
                        "type":        "governance",
                        "label":       "Governance",
                        "topic":       title,
                        "description": desc,
                        "prompt":      f"What are the approval rules for {title.lower()}?",
                    })
            except Exception as exc:
                logger.debug("Manager search failed (%s): %s", chunk_type, exc)

            if len(suggestions) >= 3:
                break

        return suggestions[:3]

    def _analyse_for_uat_tester(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Validation functions and known failure scenarios from commentary + forum."""
        topic = _extract_business_topic(question) or question[:60]

        suggestions: list[dict] = []
        seen: set[str] = set()

        # Validation functions from commentary
        try:
            val_chunks = self.store.query(
                f"validate {topic} error condition",
                n_results=5,
                where={"chunk_type": {"$eq": "commentary"}},
            )
            for c in val_chunks:
                meta = c.get("metadata", {})
                fn = meta.get("function_name", "")
                if not fn or fn in seen:
                    continue
                if not any(kw in fn.lower() for kw in
                           ("valid", "check", "verif", "assert", "submit", "test")):
                    continue
                seen.add(fn)
                suggestions.append({
                    "type":        "validation",
                    "label":       "Also test",
                    "scenario":    fn.replace("_", " ").title(),
                    "description": c["text"][:100].replace("\n", " "),
                    "prompt":      f"What validation runs in {fn}?",
                })
        except Exception as exc:
            logger.debug("UAT validation search failed: %s", exc)

        # Known failure scenarios from forum
        try:
            forum_chunks = self.store.query(
                f"{topic} error fails rejected",
                n_results=4,
                where={"chunk_type": {"$eq": "forum"}},
            )
            for c in forum_chunks:
                meta = c.get("metadata", {})
                title = meta.get("title") or meta.get("source", "Forum issue")
                if title in seen:
                    continue
                seen.add(title)
                desc = c["text"][:100].replace("\n", " ")
                suggestions.append({
                    "type":        "edge_case",
                    "label":       "Test scenario",
                    "scenario":    title,
                    "description": desc,
                    "prompt":      f"What happens when {topic} fails or is rejected?",
                })
        except Exception as exc:
            logger.debug("UAT forum search failed: %s", exc)

        return suggestions[:3]

    def _analyse_for_end_user(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Simple next steps and related help from docs and forum — plain language only."""
        topic = _extract_business_topic(question) or question[:60]

        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        # Search docs and forum for simple follow-up guidance
        for chunk_type in ("docs", "forum"):
            try:
                chunks = self.store.query(
                    f"how to {topic} next steps",
                    n_results=4,
                    where={"chunk_type": {"$eq": chunk_type}},
                )
                for c in chunks:
                    meta = c.get("metadata", {})
                    title = _derive_display_title(c, meta)
                    if not title or title in seen_topics:
                        continue
                    if re.search(r'[a-z]_[a-z]', title):
                        continue
                    seen_topics.add(title)
                    desc = c["text"][:100].replace("\n", " ")
                    suggestions.append({
                        "type":        "next_step",
                        "label":       "What's next",
                        "topic":       title,
                        "description": desc,
                        "prompt":      f"How do I {topic.lower()}?",
                    })
            except Exception as exc:
                logger.debug("End user search failed (%s): %s", chunk_type, exc)

            if len(suggestions) >= 2:
                break

        return suggestions[:2]

    # ------------------------------------------------------------------
    # Cross-cutting: knowledge gap alert
    # ------------------------------------------------------------------

    def _check_knowledge_gap(
        self, confidence: float, question: str, role: str
    ) -> dict:
        if confidence >= _LOW_CONFIDENCE:
            return {"triggered": False}

        pct = round(confidence * 100)

        if role in ("developer", "system_admin"):
            message = (
                f"Low retrieval confidence ({pct}%). "
                "This function may not be fully indexed in the knowledge base."
            )
        elif role in ("business_user", "manager", "consultant"):
            message = (
                f"I have limited information on this topic ({pct}% confidence). "
                "The suggestions below may help."
            )
        elif role == "uat_tester":
            message = (
                f"Test coverage data is limited for this area ({pct}% confidence). "
                "Consider exploring related scenarios."
            )
        else:
            message = (
                f"I'm not fully confident in this answer ({pct}%). "
                "You may want to check with your administrator."
            )

        return {
            "triggered":  True,
            "confidence": confidence,
            "message":    message,
        }

    # ------------------------------------------------------------------
    # Cross-cutting: staleness detector
    # ------------------------------------------------------------------

    def _check_staleness(self, chunks: list) -> dict:
        oldest_days = 0
        oldest_date = None

        for c in chunks:
            meta = c.get("metadata", {})
            ts_str = meta.get("ingestion_timestamp") or meta.get("generated_at")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str)[:19])
                days = (datetime.utcnow() - ts).days
                if days > oldest_days:
                    oldest_days = days
                    oldest_date = str(ts_str)[:10]
            except Exception:
                continue

        if oldest_days <= _STALENESS_DAYS:
            return {"triggered": False}

        return {
            "triggered":         True,
            "days_old":          oldest_days,
            "oldest_chunk_date": oldest_date,
            "message":           f"Commentary was generated {oldest_days} days ago.",
            "action":            "py ingest.py --source commentary",
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_GENERIC_SOURCE_NAMES = {
    "forum", "docs", "code commentary", "commentary",
    "raw code", "call graph", "unknown",
}


def _derive_display_title(chunk: dict, meta: dict) -> str:
    """Return a human-readable title for a chunk, avoiding generic source names."""
    # 1. Prefer explicit title metadata
    title = meta.get("title") or meta.get("page_title") or meta.get("doc_title")
    if title and title.lower() not in _GENERIC_SOURCE_NAMES:
        return title.strip()

    # 2. Derive from first meaningful line of text
    first_line = chunk.get("text", "").split("\n")[0].strip()
    # Skip lines that look like function headers ("Function: foo_bar")
    if first_line and not re.match(r'^(Function|File|Module|Class):', first_line):
        candidate = first_line[:70]
        if len(candidate) > 8:
            return candidate

    # 3. Fall back to source, but only if it's not a generic name
    source = meta.get("source", "").replace("_", " ").title()
    if source.lower() not in _GENERIC_SOURCE_NAMES:
        return source

    return ""


def _extract_function_name(question: str) -> str:
    """Return the most likely function name (snake_case) from a question."""
    snake = re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', question)
    if snake:
        return max(snake, key=len)
    # PascalCase class names
    pascal = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', question)
    return pascal[0] if pascal else ""


def _extract_business_topic(question: str) -> str:
    """Return a plain-English business topic phrase from a question."""
    # Multi-word business phrases — order matters (longer matches first)
    _PHRASES = [
        "expense claim", "leave application", "purchase order",
        "salary slip", "attendance record", "journal entry",
        "sales order", "purchase invoice", "payment entry",
        "employee transfer", "appraisal cycle", "asset allocation",
        "stock entry", "delivery note", "quality inspection",
    ]
    q_lower = question.lower()
    for phrase in _PHRASES:
        if phrase in q_lower:
            return phrase

    # Fall back to first meaningful multi-word noun phrase
    words = re.sub(r'[^a-zA-Z ]', ' ', question).split()
    candidates = [
        w.lower() for w in words
        if len(w) > 4 and w.lower() not in _STOPWORDS
    ]
    # Prefer two-word pair if available
    if len(candidates) >= 2:
        return f"{candidates[0]} {candidates[1]}"
    return candidates[0] if candidates else ""
