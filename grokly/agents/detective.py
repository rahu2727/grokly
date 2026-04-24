"""
grokly/agents/detective.py — The Detective: role-aware retrieval from ChromaDB.

Responsibilities:
  - Chooses the best chunk_type to query based on user_role and question keywords
  - Runs primary retrieval, supplements with unfiltered search if results are sparse
  - Scores retrieval confidence from cosine distances (0.0 – 1.0)
  - Sets needs_reretrieval = True when confidence < 0.6
"""

from grokly.store.chroma_store import ChromaStore
from grokly.pipeline.state import GroklyState

# Role → preferred chunk_type (None = no filter, search all)
_ROLE_CHUNK_TYPE: dict[str, str | None] = {
    "developer":     "raw_code",
    "system_admin":  "raw_code",
    "business_user": "commentary",
    "manager":       "commentary",
    "end_user":      "commentary",
    "uat_tester":    None,       # mixed sources — needs both process and code
    "doc_generator": None,       # mixed sources — needs comprehensive context
    "consultant":    None,       # consultants benefit from mixed sources
}

# Keywords that signal an impact / dependency question → call_graph
_IMPACT_KEYWORDS = {
    "impact", "affects", "affected", "calls", "triggers", "cascades",
    "depends", "dependency", "dependencies", "what happens when",
}


def detective_node(state: GroklyState) -> dict:
    """Retrieve initial context chunks with role-aware prioritisation."""
    question = state["user_question"]
    role = state.get("user_role", "business_user").lower().replace(" ", "_")

    store = ChromaStore()

    chunk_type = _pick_chunk_type(question, role)
    chunks = _retrieve(store, question, chunk_type, n=5)

    # Supplement with unfiltered results if primary was thin
    if len(chunks) < 3:
        unfiltered = store.query(question, n_results=5)
        seen = {c["text"] for c in chunks}
        for c in unfiltered:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])

    confidence = _score_confidence(chunks)
    sources = list({c["metadata"].get("source", "unknown") for c in chunks})
    tool_log = f"detective:retrieve(role={role}, chunk_type={chunk_type or 'all'})"

    return {
        "retrieved_chunks":    chunks,
        "retrieval_confidence": confidence,
        "needs_reretrieval":   confidence < 0.6,
        "sources":             sources,
        "tool_calls_made":     [tool_log],
        "iteration_count":     1,
        "tracker_retries":     0,
        "counsel_retries":     0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_chunk_type(question: str, role: str) -> str | None:
    q_lower = question.lower()
    if any(kw in q_lower for kw in _IMPACT_KEYWORDS):
        return "call_graph"
    return _ROLE_CHUNK_TYPE.get(role)


def _retrieve(store: ChromaStore, query: str, chunk_type: str | None, n: int) -> list[dict]:
    if chunk_type is None:
        return store.query(query, n_results=n)
    return store.query(query, n_results=n, where={"chunk_type": {"$eq": chunk_type}})


def _score_confidence(chunks: list[dict]) -> float:
    """Convert cosine distances to a 0–1 confidence score."""
    if not chunks:
        return 0.0
    distances = [c.get("distance", 1.0) for c in chunks[:5]]
    avg_distance = sum(distances) / len(distances)
    return round(max(0.0, min(1.0, 1.0 - avg_distance)), 3)
