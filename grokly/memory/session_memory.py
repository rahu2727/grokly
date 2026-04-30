"""
grokly/memory/session_memory.py — Short-term session memory for Grokly.

Keeps a rolling window of conversation turns. When the window exceeds max_turns,
the oldest turns are compressed into a summary via Claude, keeping context concise
without losing important prior context.
"""

from __future__ import annotations

import logging

import anthropic
from dotenv import load_dotenv

from grokly.model_config import get_agent_config

load_dotenv()
logger = logging.getLogger(__name__)

_REFERENCE_TRIGGERS = {
    "it", "that", "this", "they", "them", "those", "these",
    "tell me more", "what about", "and the", "more about",
    "explain that", "elaborate", "go deeper", "expand on",
    "same thing", "the same", "continue", "also", "as well",
}

_COMPRESS_SYSTEM = (
    "You are a conversation summariser. Summarise the provided conversation turns "
    "into a concise paragraph (3-5 sentences). Preserve: key topics discussed, "
    "important answers given, and any unresolved questions. Be factual."
)

_RESOLVE_SYSTEM = (
    "You are resolving vague references in a question based on recent conversation. "
    "Return ONLY the rewritten, fully-specified question with no explanation. "
    "If the question already stands alone, return it unchanged."
)


class SessionMemory:
    def __init__(self, max_turns: int = 10) -> None:
        self.turns: list[dict] = []
        self.max_turns = max_turns
        self.context_summary: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        question: str,
        answer: str,
        role: str,
        sources: list[str],
        confidence: float,
    ) -> None:
        self.turns.append({
            "question":   question,
            "answer":     answer,
            "role":       role,
            "sources":    sources,
            "confidence": confidence,
        })

        if len(self.turns) > self.max_turns:
            self._compress_oldest()

    def get_context(self) -> str:
        parts: list[str] = []

        if self.context_summary:
            parts.append(f"Earlier in this conversation: {self.context_summary}")

        for t in self.turns:
            answer_preview = t["answer"][:200]
            if len(t["answer"]) > 200:
                answer_preview += "..."
            parts.append(
                f"User asked ({t['role']}): {t['question']}\n"
                f"Grokly answered: {answer_preview}"
            )

        return "\n\n".join(parts) if parts else ""

    def resolve_references(self, question: str) -> str:
        """Rewrite the question to resolve pronouns/vague refs against recent context."""
        if not self.turns:
            return question

        q_lower = question.lower()
        if not any(trigger in q_lower for trigger in _REFERENCE_TRIGGERS):
            return question

        history_text = "\n".join(
            f"Q: {t['question']}\nA: {t['answer'][:300]}"
            for t in self.turns[-3:]
        )

        try:
            _cfg = get_agent_config("memory")
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=_cfg["model"],
                max_tokens=_cfg["max_tokens"],
                temperature=_cfg["temperature"],
                system=_RESOLVE_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Recent conversation:\n{history_text}\n\n"
                            f"New question to resolve: {question}"
                        ),
                    }
                ],
            )
            resolved = resp.content[0].text.strip()
            if resolved and resolved != question:
                logger.debug("Reference resolved: %r → %r", question, resolved)
            return resolved
        except Exception as exc:
            logger.debug("Reference resolution skipped: %s", exc)
            return question

    def get_last_topic(self) -> str:
        if not self.turns:
            return ""
        last_q = self.turns[-1]["question"]
        words = [w for w in last_q.split() if len(w) > 4]
        return words[0] if words else last_q.split()[0] if last_q.split() else ""

    def clear(self) -> None:
        self.turns = []
        self.context_summary = ""

    def to_dict(self) -> dict:
        return {
            "turns":           self.turns,
            "max_turns":       self.max_turns,
            "context_summary": self.context_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionMemory:
        mem = cls(max_turns=data.get("max_turns", 10))
        mem.turns = data.get("turns", [])
        mem.context_summary = data.get("context_summary", "")
        return mem

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compress_oldest(self) -> None:
        """Compress oldest (max_turns - 5) turns into context_summary via Claude."""
        n_to_compress = len(self.turns) - 5
        if n_to_compress <= 0:
            return

        old_turns = self.turns[:n_to_compress]
        self.turns = self.turns[n_to_compress:]

        history_text = "\n".join(
            f"Q ({t['role']}): {t['question']}\nA: {t['answer'][:400]}"
            for t in old_turns
        )

        existing_prefix = (
            f"Previous summary: {self.context_summary}\n\n"
            if self.context_summary
            else ""
        )

        try:
            _cfg = get_agent_config("memory")
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=_cfg["model"],
                max_tokens=_cfg["max_tokens"],
                temperature=_cfg["temperature"],
                system=_COMPRESS_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{existing_prefix}"
                            f"Conversation turns to summarise:\n{history_text}"
                        ),
                    }
                ],
            )
            self.context_summary = resp.content[0].text.strip()
            logger.debug("Session memory compressed %d turns", n_to_compress)
        except Exception as exc:
            logger.debug("Session compression skipped: %s", exc)
            summary_lines = [
                f"- {t['question']}" for t in old_turns
            ]
            fallback = "Topics covered: " + "; ".join(
                t["question"][:60] for t in old_turns
            )
            self.context_summary = (
                (self.context_summary + " " + fallback).strip()
                if self.context_summary
                else fallback
            )
