"""
grokly/agents/proactive_agent.py — The Proactive Agent.

Surfaces unsolicited insights after every answer. Each role gets a
completely different search strategy — not the same data reformatted:

  developer    → call_graph (chunk_type=call_graph) + commentary (chunk_type=commentary)
  business_user → docs (source=docs) + forum (source=forum) — no function names
  manager      → docs + forum focused on governance/approval
  uat_tester   → commentary (validation fns) + forum (failure scenarios)
  end_user     → docs + forum — plain language next steps

IMPORTANT: ChromaDB schema in this knowledge base
  - source field:     "call_graph" | "code_commentary" | "docs" | "forum"
  - chunk_type field: "call_graph" | "commentary" | "raw_code" | "question" | "answer"
  Filter docs/forum by source; filter commentary/call_graph by chunk_type.

Cross-cutting behaviours:
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
_STALENESS_DAYS = 30

_QUESTION_WORDS = {
    "what", "how", "why", "when", "where", "does", "do",
    "is", "are", "can", "will", "would", "should",
}

_GENERIC_SOURCE_NAMES = {
    "forum", "docs", "code commentary", "commentary",
    "raw code", "call graph", "unknown",
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

        logger.debug(
            "Proactive[%s] q=%r → %d suggestion(s)",
            role, question[:60], len(suggestions),
        )

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
        """Callers, callees, and sibling functions from call_graph + commentary."""
        entity = _extract_entity(question, answer)
        logger.debug("Developer entity: %r", entity)
        if not entity:
            return []

        suggestions: list[dict] = []
        seen: set[str] = {entity}

        # 1. Call-graph neighbours — filter by chunk_type (correct for this DB)
        try:
            cg_chunks = self.store.query(
                f"{entity} function calls",
                n_results=6,
                where={"chunk_type": {"$eq": "call_graph"}},
            )
            logger.debug("Developer call_graph results: %d", len(cg_chunks))
            for c in cg_chunks:
                meta = c.get("metadata", {})
                neighbour = meta.get("function_name", "")
                if not neighbour or neighbour in seen:
                    continue
                seen.add(neighbour)
                text = c["text"]
                # Determine direction: if entity appears after "calls:" it's a callee
                rel_type = "callee" if "calls:" in text and entity in text else "related"
                suggestions.append({
                    "type":        rel_type,
                    "label":       "Also calls" if rel_type == "callee" else "Related function",
                    "function":    neighbour,
                    "description": text[:80].replace("\n", " "),
                    "prompt":      f"What does {neighbour} do?",
                })
        except Exception as exc:
            logger.debug("Developer call-graph search failed: %s", exc)

        # 2. Commentary fallback — query commentary when call_graph is sparse
        if len(suggestions) < 2:
            try:
                com_chunks = self.store.query(
                    entity,
                    n_results=6,
                    where={"chunk_type": {"$eq": "commentary"}},
                )
                logger.debug("Developer commentary results: %d", len(com_chunks))
                for c in com_chunks:
                    meta = c.get("metadata", {})
                    fn = meta.get("function_name", "")
                    if not fn or fn in seen:
                        continue
                    seen.add(fn)
                    suggestions.append({
                        "type":        "related",
                        "label":       "Related function",
                        "function":    fn,
                        "description": c["text"][:80].replace("\n", " "),
                        "prompt":      f"What does {fn} do?",
                    })
            except Exception as exc:
                logger.debug("Developer commentary search failed: %s", exc)

        # 3. Sibling functions in same module (from retrieved context)
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

        logger.debug("Developer suggestions: %d → %s", len(suggestions[:3]),
                     [s["function"] for s in suggestions[:3]])
        return suggestions[:3]

    def _analyse_for_business_user(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Related business processes from docs and forum — no function names ever."""
        topic = _extract_entity(question, answer)
        logger.debug("Business user topic: %r", topic)
        if not topic:
            return []

        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        # Filter by source field (NOT chunk_type — docs chunks have no chunk_type)
        for src in ("docs", "forum"):
            try:
                chunks = self.store.query(
                    f"{topic} process workflow policy",
                    n_results=5,
                    where={"source": {"$eq": src}},
                )
                logger.debug("Business user %s results: %d", src, len(chunks))
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
                logger.debug("Business user search failed (%s): %s", src, exc)

            if len(suggestions) >= 3:
                break

        logger.debug("Business user suggestions: %d", len(suggestions[:3]))
        return suggestions[:3]

    def _analyse_for_manager(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Governance, approval rules, and audit context from docs and forum."""
        topic = _extract_entity(question, answer)
        gov_query = f"approval limit delegation audit {topic}" if topic else "approval workflow audit"

        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        for src in ("docs", "forum"):
            try:
                chunks = self.store.query(
                    gov_query,
                    n_results=5,
                    where={"source": {"$eq": src}},
                )
                logger.debug("Manager %s results: %d", src, len(chunks))
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
                logger.debug("Manager search failed (%s): %s", src, exc)

            if len(suggestions) >= 3:
                break

        logger.debug("Manager suggestions: %d", len(suggestions[:3]))
        return suggestions[:3]

    def _analyse_for_uat_tester(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Validation functions and known failure scenarios."""
        topic = _extract_entity(question, answer) or question[:60]

        suggestions: list[dict] = []
        seen: set[str] = set()

        # Validation functions — chunk_type filter works here (commentary chunks)
        try:
            val_chunks = self.store.query(
                f"validate {topic} error condition",
                n_results=5,
                where={"chunk_type": {"$eq": "commentary"}},
            )
            logger.debug("UAT validation results: %d", len(val_chunks))
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

        # Failure scenarios from forum — filter by source (correct field)
        try:
            forum_chunks = self.store.query(
                f"{topic} error fails rejected",
                n_results=4,
                where={"source": {"$eq": "forum"}},
            )
            logger.debug("UAT forum results: %d", len(forum_chunks))
            for c in forum_chunks:
                title = _derive_display_title(c, c.get("metadata", {}))
                if not title or title in seen:
                    continue
                seen.add(title)
                suggestions.append({
                    "type":        "edge_case",
                    "label":       "Test scenario",
                    "scenario":    title,
                    "description": c["text"][:100].replace("\n", " "),
                    "prompt":      f"What happens when {topic} fails or is rejected?",
                })
        except Exception as exc:
            logger.debug("UAT forum search failed: %s", exc)

        logger.debug("UAT suggestions: %d", len(suggestions[:3]))
        return suggestions[:3]

    def _analyse_for_end_user(
        self,
        question: str,
        answer: str,
        confidence: float,
    ) -> list[dict]:
        """Simple next steps from docs and forum — plain language only."""
        topic = _extract_entity(question, answer) or question[:60]

        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        for src in ("docs", "forum"):
            try:
                chunks = self.store.query(
                    f"how to {topic} next steps",
                    n_results=4,
                    where={"source": {"$eq": src}},
                )
                logger.debug("End user %s results: %d", src, len(chunks))
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
                logger.debug("End user search failed (%s): %s", src, exc)

            if len(suggestions) >= 2:
                break

        logger.debug("End user suggestions: %d", len(suggestions[:2]))
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

def _extract_entity(question: str, answer: str = "") -> str:
    """
    Three-strategy entity extractor that works for both technical and
    business-language questions.

    Strategy 1: snake_case identifiers (function names) — search question + answer
    Strategy 2: word immediately after a question word
    Strategy 3: first meaningful phrase after stripping question words
    """
    combined = question + " " + answer

    # Strategy 1: snake_case (longest wins — most specific)
    snake = re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b', combined)
    if snake:
        return max(snake, key=len)

    # Strategy 2: PascalCase class names
    pascal = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', question)
    if pascal:
        return pascal[0]

    # Strategy 3: first content word after a question word
    words = question.lower().split()
    for i, word in enumerate(words):
        clean_word = word.strip("?,.")
        if clean_word in _QUESTION_WORDS and i + 1 < len(words):
            candidate = words[i + 1].strip("?.,")
            if len(candidate) > 3 and candidate not in _QUESTION_WORDS:
                return candidate

    # Strategy 4: first 3 meaningful words from the question
    content_words = [
        w.strip("?.,") for w in words
        if len(w.strip("?.,")) > 3 and w.strip("?.,") not in _QUESTION_WORDS
    ]
    return " ".join(content_words[:3]) if content_words else ""


def _derive_display_title(chunk: dict, meta: dict) -> str:
    """Return a human-readable title, avoiding generic source names."""
    title = meta.get("title") or meta.get("page_title") or meta.get("doc_title")
    if title and title.lower() not in _GENERIC_SOURCE_NAMES:
        return title.strip()

    # Derive from first meaningful line of text
    first_line = chunk.get("text", "").split("\n")[0].strip()
    if first_line and not re.match(r'^(Function|File|Module|Class):', first_line):
        candidate = first_line[:70]
        if len(candidate) > 8:
            return candidate

    # Source as last resort — but only if non-generic
    source = meta.get("source", "").replace("_", " ").title()
    if source.lower() not in _GENERIC_SOURCE_NAMES:
        return source

    return ""
