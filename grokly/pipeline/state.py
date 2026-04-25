"""
grokly/pipeline/state.py — Shared state TypedDict for the Grokly LangGraph pipeline.

All fields use total=False so each agent node can return a partial update dict
and LangGraph will merge it into the running state.
"""

from typing import TypedDict


class GroklyState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    user_question: str       # original question from the user
    user_role: str           # end_user | business_user | manager | developer | uat_tester | doc_generator

    # ── Retrieval (Detective + Tracker) ───────────────────────────────────────
    retrieved_chunks: list[dict]     # list of {text, metadata, distance} dicts
    retrieval_confidence: float      # 0.0–1.0  (1 - avg cosine distance)
    needs_reretrieval: bool          # set by Detective when confidence < 0.6
    reformulated_query: str          # Tracker may rewrite the query before re-searching

    # ── Generation (Counsel) ──────────────────────────────────────────────────
    raw_answer: str          # answer before persona formatting
    quality_score: float     # avg of grounded + complete + role_appropriate (1–5 each)
    counsel_retries: int     # how many times Counsel has run (max 2 total, 1 retry)

    # ── Delivery (Briefer) ────────────────────────────────────────────────────
    final_answer: str        # persona-formatted answer

    # ── Provenance & tracking ─────────────────────────────────────────────────
    sources: list[str]           # unique source tags from retrieved chunks
    tool_calls_made: list[str]   # log of every tool invocation
    iteration_count: int         # total reasoning steps across all nodes
    tracker_retries: int         # how many times Tracker has re-retrieved

    # ── Conversation memory ───────────────────────────────────────────────────
    messages: list[dict]     # running log: {role, content, quality?}
    session_context: str     # compressed prior-turn context injected by SessionMemory
