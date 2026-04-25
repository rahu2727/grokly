"""
grokly/agents/counsel.py — The Counsel: grounded answer generation + self-reflection.

Responsibilities:
  - Generates a factual answer grounded in the retrieved chunks
  - Self-evaluates quality on three dimensions (grounded, complete, role_appropriate)
    each scored 1–5; returns the average as quality_score
  - If quality_score < 3.0 and this is the first attempt, the graph will retry
  - Appends a record to conversation messages for auditability
"""

import json

import anthropic
from dotenv import load_dotenv

from grokly.pipeline.state import GroklyState

load_dotenv()

_ANSWER_SYSTEM = """You are The Counsel, an ERPNext expert providing grounded answers.

Rules:
1. Base your answer ONLY on the provided context chunks — never hallucinate.
2. If the context is insufficient, say so explicitly.
3. Be specific: reference menu paths, field names, and step sequences.
4. Write for the stated user role — match their expected level of detail.
5. Be concise: prefer clear bullet points or numbered steps over long prose."""

_REFLECTION_SYSTEM = """You are a quality auditor reviewing ERPNext assistant answers.
Return ONLY a valid JSON object — no explanation, no markdown fences."""


def counsel_node(state: GroklyState) -> dict:
    """Generate a grounded answer and self-evaluate its quality."""
    question = state["user_question"]
    role = state.get("user_role", "business_user")
    chunks = state.get("retrieved_chunks", [])
    counsel_retries = state.get("counsel_retries", 0)
    messages = list(state.get("messages", []))

    session_context: str = state.get("session_context", "")
    client = anthropic.Anthropic()
    context = _format_context(chunks)

    session_prefix = (
        f"Conversation history for reference:\n{session_context}\n\n"
        if session_context
        else ""
    )

    # Step 1: Generate answer
    answer_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_ANSWER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{session_prefix}"
                    f"Question: {question}\n"
                    f"User role: {role}\n\n"
                    f"Context:\n{context}\n\n"
                    "Provide a clear, grounded answer based only on the context above."
                ),
            }
        ],
    )
    raw_answer = answer_response.content[0].text.strip()

    # Step 2: Self-reflection / quality scoring
    quality_score, _ = _evaluate_quality(client, question, raw_answer, role, context)

    counsel_retries += 1
    messages.append({
        "role":    "counsel",
        "content": raw_answer,
        "quality": quality_score,
        "attempt": counsel_retries,
    })

    return {
        "raw_answer":      raw_answer,
        "quality_score":   quality_score,
        "counsel_retries": counsel_retries,
        "messages":        messages,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluate_quality(
    client: anthropic.Anthropic,
    question: str,
    answer: str,
    role: str,
    context: str,
) -> tuple[float, dict]:
    """Score the answer on three 1–5 dimensions. Returns (avg, details)."""
    prompt = (
        f"Question: {question}\n"
        f"User role: {role}\n"
        f"Context (first 600 chars): {context[:600]}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        "Score this answer on each dimension (integer 1–5):\n"
        "  grounded        — is it based only on the provided context?\n"
        "  complete        — does it fully address the question?\n"
        "  role_appropriate — is the language/detail level right for this role?\n\n"
        'Return ONLY JSON, e.g.: {"grounded": 4, "complete": 3, "role_appropriate": 5}'
    )

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            system=_REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        scores = json.loads(text)
        avg = round(sum(scores.values()) / len(scores), 2)
        return avg, scores
    except Exception:
        return 3.0, {"grounded": 3, "complete": 3, "role_appropriate": 3}


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no context retrieved — answer based on general knowledge only)"
    parts = []
    for i, chunk in enumerate(chunks[:6], 1):
        meta = chunk.get("metadata", {})
        src = meta.get("source", "unknown")
        module = meta.get("module", "")
        tag = f"{src}/{module}" if module else src
        parts.append(f"[{i}: {tag}]\n{chunk['text']}")
    return "\n\n".join(parts)
