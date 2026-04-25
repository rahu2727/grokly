"""
grokly/pipeline/pipeline.py — Public interface for the Grokly agentic pipeline.

Usage:
    from grokly.pipeline.pipeline import run

    result = run("How do I run payroll?", role="business_user")
    print(result["answer"])
    print(result["confidence"])
    print(result["tools_used"])
"""

from grokly.pipeline.graph import get_graph
from grokly.pipeline.state import GroklyState

VALID_ROLES = {
    "end_user", "business_user", "manager", "developer",
    "uat_tester", "doc_generator", "system_admin", "consultant",
}


def run(
    question: str,
    role: str = "business_user",
    session_memory=None,
    user_memory=None,
    user_id: str = None,
) -> dict:
    """
    Run the full Grokly agentic pipeline for a question and role.

    Args:
        question:       Natural-language question about ERPNext.
        role:           One of end_user | business_user | manager | developer |
                        uat_tester | doc_generator | system_admin | consultant.
        session_memory: Optional SessionMemory instance for reference resolution
                        and prior-turn context injection.
        user_memory:    Optional UserMemory instance for long-term profile updates.
        user_id:        User identifier passed to UserMemory (required when
                        user_memory is provided).

    Returns a dict with:
        answer             — final persona-formatted answer (str)
        sources            — list of source tags used (list[str])
        confidence         — retrieval confidence score 0.0–1.0 (float)
        tools_used         — list of tool call log strings (list[str])
        iteration_count    — total reasoning steps (int)
        quality_score      — self-reflection quality score 1.0–5.0 (float)
        resolved_question  — question after reference resolution (str)
    """
    from grokly.memory.session_memory import SessionMemory  # local to avoid circular

    # ── Reference resolution & context extraction ─────────────────────────────
    resolved_question = question
    conversation_context = ""

    if session_memory is not None:
        resolved_question = session_memory.resolve_references(question)
        conversation_context = session_memory.get_context()

    # ── Role normalisation ────────────────────────────────────────────────────
    role_clean = role.lower().replace(" ", "_")
    if role_clean not in VALID_ROLES:
        role_clean = "business_user"

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: GroklyState = {
        "user_question":        resolved_question,
        "user_role":            role_clean,
        "resolved_question":    resolved_question,
        "conversation_context": conversation_context,
        "user_profile_context": "",
        "retrieved_chunks":     [],
        "retrieval_confidence":  0.0,
        "needs_reretrieval":    False,
        "reformulated_query":   resolved_question,
        "raw_answer":           "",
        "final_answer":         "",
        "quality_score":        0.0,
        "counsel_retries":      0,
        "sources":              [],
        "tool_calls_made":      [],
        "iteration_count":      0,
        "tracker_retries":      0,
        "messages":             [],
    }

    graph = get_graph()

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        return {
            "answer":            f"Pipeline error: {exc}",
            "sources":           [],
            "confidence":        0.0,
            "tools_used":        [],
            "iteration_count":   0,
            "quality_score":     0.0,
            "resolved_question": resolved_question,
        }

    answer = final_state.get("final_answer") or final_state.get("raw_answer", "")
    if not answer:
        answer = "No answer could be generated. Check API keys and ChromaDB population."

    result = {
        "answer":            answer,
        "sources":           final_state.get("sources", []),
        "confidence":        round(final_state.get("retrieval_confidence", 0.0), 3),
        "tools_used":        final_state.get("tool_calls_made", []),
        "iteration_count":   final_state.get("iteration_count", 0),
        "quality_score":     round(final_state.get("quality_score", 0.0), 2),
        "resolved_question": resolved_question,
    }

    # ── Post-run memory updates ───────────────────────────────────────────────
    if session_memory is not None:
        session_memory.add_turn(
            question=question,
            answer=answer,
            role=role_clean,
            sources=result["sources"],
            confidence=result["confidence"],
        )

    if user_memory is not None and user_id:
        user_memory.update_profile(
            user_id=user_id,
            question=question,
            role=role_clean,
            confidence=result["confidence"],
        )

    return result
