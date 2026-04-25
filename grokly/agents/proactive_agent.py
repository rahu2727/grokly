"""
grokly/agents/proactive_agent.py — The Proactive Agent.

Runs after The Briefer delivers an answer and surfaces three types of
unsolicited insights without the user having to ask:

  1. Knowledge gap alerts  — warns when retrieval confidence is low and
                             suggests better-covered adjacent topics.
  2. Related knowledge     — surfaces callers, callees, and sibling functions
                             from the same module.
  3. Staleness detector    — warns when ingested commentary is > 30 days old.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from dotenv import load_dotenv

from grokly.store.chroma_store import ChromaStore

load_dotenv()
logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.60
_VERY_LOW_CONFIDENCE = 0.35
_STALENESS_DAYS = 30

_STOPWORDS = {
    "what", "does", "how", "when", "where", "which", "about", "with",
    "have", "from", "that", "this", "will", "would", "could", "should",
    "tell", "show", "give", "explain", "list", "find", "make", "into",
    "does", "work", "used", "used", "using", "call", "called",
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
        gap = self._check_knowledge_gap(confidence, question, retrieved_chunks)
        related = self._find_related_knowledge(question, answer, role, retrieved_chunks)
        staleness = self._check_staleness(retrieved_chunks)
        has_insights = (
            gap.get("triggered")
            or related.get("triggered")
            or staleness.get("triggered")
        )
        return {
            "gap_alert": gap,
            "related":   related,
            "staleness": staleness,
            "has_insights": bool(has_insights),
        }

    def format_for_ui(self, insights: dict, role: str) -> dict:
        """Filter and reformat insights to match what each role cares about."""
        gap      = insights.get("gap_alert", {})
        related  = insights.get("related",   {})
        staleness = insights.get("staleness", {})

        out: dict = {"gap_alert": {}, "related": {}, "staleness": {}, "has_insights": False}

        if role in ("developer", "system_admin"):
            out["gap_alert"] = gap
            out["related"]   = related
            out["staleness"] = staleness

        elif role in ("business_user", "manager", "consultant"):
            out["gap_alert"] = gap
            if related.get("triggered"):
                humanised = [
                    {
                        **s,
                        "function": s["function"].replace("_", " ").title(),
                        "prompt":   f"Ask me: Tell me about {s['function'].replace('_', ' ').title()}",
                    }
                    for s in related.get("suggestions", [])
                ]
                out["related"] = {**related, "suggestions": humanised}
            # staleness not surfaced — business users don't care about ingest dates

        elif role == "end_user":
            if gap.get("triggered") and gap.get("confidence", 1.0) < _VERY_LOW_CONFIDENCE:
                out["gap_alert"] = gap
            # related and staleness hidden

        elif role == "uat_tester":
            out["gap_alert"] = gap
            if related.get("triggered"):
                uat_sug = [
                    s for s in related.get("suggestions", [])
                    if any(
                        kw in s.get("function", "").lower()
                        for kw in ("valid", "check", "test", "verif", "assert", "submit")
                    )
                ]
                if uat_sug:
                    out["related"] = {**related, "suggestions": uat_sug}
            # staleness shown for UAT (they care about fresh data)
            out["staleness"] = staleness

        else:
            # doc_generator or unknown: gap + staleness
            out["gap_alert"] = gap
            out["staleness"] = staleness

        out["has_insights"] = bool(
            out["gap_alert"].get("triggered")
            or out["related"].get("triggered")
            or out["staleness"].get("triggered")
        )
        return out

    # ------------------------------------------------------------------
    # Behaviour 1: Knowledge gap alert
    # ------------------------------------------------------------------

    def _check_knowledge_gap(
        self,
        confidence: float,
        question: str,
        chunks: list,
    ) -> dict:
        if confidence >= _LOW_CONFIDENCE_THRESHOLD:
            return {"triggered": False}

        pct = round(confidence * 100)
        suggestions: list[dict] = []

        try:
            candidates = self.store.query(question, n_results=10)
            existing = {c["text"][:120] for c in chunks}

            for c in candidates:
                key = c["text"][:120]
                if key in existing:
                    continue
                dist = c.get("distance", 1.0)
                if dist > 0.45:  # skip low-quality matches
                    continue
                meta = c.get("metadata", {})
                topic = (
                    meta.get("function_name")
                    or meta.get("module")
                    or meta.get("source", "ERPNext")
                )
                suggestions.append({
                    "topic":      topic,
                    "chunk_type": meta.get("chunk_type", "general"),
                    "snippet":    c["text"][:100].replace("\n", " "),
                })
                if len(suggestions) == 3:
                    break
        except Exception as exc:
            logger.debug("Gap check failed: %s", exc)

        return {
            "triggered":   True,
            "confidence":  confidence,
            "message":     f"My confidence on this topic is low ({pct}%).",
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------
    # Behaviour 2: Related knowledge suggestions
    # ------------------------------------------------------------------

    def _find_related_knowledge(
        self,
        question: str,
        answer: str,
        role: str,
        chunks: list,
    ) -> dict:
        entity = _extract_entity(question)
        if not entity:
            return {"triggered": False}

        suggestions: list[dict] = []
        seen_fns: set[str] = {entity}

        try:
            # 1. Call-graph neighbours
            call_chunks = self.store.query(
                entity, n_results=5,
                where={"chunk_type": {"$eq": "call_graph"}},
            )
            for c in call_chunks:
                meta = c.get("metadata", {})
                fn = meta.get("function_name", "")
                if not fn or fn in seen_fns:
                    continue
                seen_fns.add(fn)
                suggestions.append({
                    "type":        "related",
                    "function":    fn,
                    "description": c["text"][:80].replace("\n", " "),
                    "prompt":      f"Ask me: What does {fn} do?",
                })
        except Exception as exc:
            logger.debug("Call-graph search failed: %s", exc)

        try:
            # 2. Sibling functions in same module
            module = next(
                (c.get("metadata", {}).get("module") for c in chunks
                 if c.get("metadata", {}).get("module")),
                None,
            )
            if module:
                mod_chunks = self.store.query(
                    module, n_results=8,
                    where={"chunk_type": {"$eq": "commentary"}},
                )
                for c in mod_chunks:
                    meta = c.get("metadata", {})
                    fn = meta.get("function_name", "")
                    if not fn or fn in seen_fns:
                        continue
                    seen_fns.add(fn)
                    suggestions.append({
                        "type":        "related",
                        "function":    fn,
                        "description": c["text"][:80].replace("\n", " "),
                        "prompt":      f"Ask me: What does {fn} do?",
                    })
        except Exception as exc:
            logger.debug("Module sibling search failed: %s", exc)

        top = suggestions[:3]
        if len(top) < 2:
            return {"triggered": False}

        return {"triggered": True, "suggestions": top}

    # ------------------------------------------------------------------
    # Behaviour 3: Staleness detector
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
            "triggered":          True,
            "days_old":           oldest_days,
            "oldest_chunk_date":  oldest_date,
            "message":            f"Commentary was generated {oldest_days} days ago.",
            "action":             "py ingest.py --source commentary",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_entity(question: str) -> str:
    """Return the most likely technical entity (function/class/module) in a question."""
    # Prefer snake_case identifiers — longest wins
    snake = re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', question)
    if snake:
        return max(snake, key=len)

    # PascalCase class names
    pascal = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', question)
    if pascal:
        return pascal[0]

    # Fall back to longest meaningful word
    words = re.sub(r'[^a-zA-Z ]', ' ', question).split()
    candidates = [w.lower() for w in words if len(w) > 4 and w.lower() not in _STOPWORDS]
    return max(candidates, key=len) if candidates else ""
