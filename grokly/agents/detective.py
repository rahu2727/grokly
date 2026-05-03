"""
grokly/agents/detective.py — The Detective: role-aware retrieval from ChromaDB.

Responsibilities:
  - Chooses the best chunk_type to query based on user_role and question keywords
  - Runs primary retrieval via MCP knowledge_server (falls back to direct ChromaStore)
  - Supplements with unfiltered search if results are sparse
  - Scores retrieval confidence from cosine distances (0.0 – 1.0)
  - Sets needs_reretrieval = True when confidence < 0.6
"""

import json
import logging

from grokly.store.chroma_store import ChromaStore
from grokly.pipeline.state import GroklyState
from grokly.agents.application_router import ApplicationRouter

logger = logging.getLogger(__name__)

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

# Role → number of chunks to retrieve
_ROLE_N_RESULTS: dict[str, int] = {
    "developer":    10,   # technical depth — needs more code/commentary chunks
    "end_user":      5,   # simple how-to questions — fewer is faster and cleaner
}
_DEFAULT_N_RESULTS = 8   # all other roles

# Keywords that signal an impact / dependency question → call_graph
_IMPACT_KEYWORDS = {
    "impact", "affects", "affected", "calls", "triggers", "cascades",
    "depends", "dependency", "dependencies", "what happens when",
}


def detective_node(state: GroklyState) -> dict:
    """Retrieve initial context chunks with role-aware prioritisation."""
    question = state["user_question"]
    role = state.get("user_role", "business_user").lower().replace(" ", "_")
    selected_application = state.get("selected_application", "")

    n = _ROLE_N_RESULTS.get(role, _DEFAULT_N_RESULTS)
    chunk_type = _pick_chunk_type(question, role)

    # Determine application routing
    router = ApplicationRouter()
    app_context = router.route(question, role, selected_application)

    chunks, method = _retrieve_with_mcp_fallback(
        question, chunk_type, role, n=n, app_context=app_context
    )

    # Supplement with unfiltered results if primary was thin
    if len(chunks) < 3:
        store = ChromaStore()
        unfiltered = store.query(question, n_results=n)
        seen = {c["text"] for c in chunks}
        for c in unfiltered:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])

    confidence = _score_confidence(chunks)
    sources = list({c["metadata"].get("source", "unknown") for c in chunks})
    tool_log = (
        f"detective:retrieve(role={role}, chunk_type={chunk_type or 'all'}, "
        f"app={app_context.get('app_label', 'all')}, via={method})"
    )

    return {
        "retrieved_chunks":    chunks,
        "retrieval_confidence": confidence,
        "needs_reretrieval":   confidence < 0.6,
        "sources":             sources,
        "tool_calls_made":     [tool_log],
        "iteration_count":     1,
        "tracker_retries":     0,
        "counsel_retries":     0,
        "application_context": app_context,
    }


def _retrieve_with_mcp_fallback(
    question: str,
    chunk_type: str | None,
    role: str,
    n: int = _DEFAULT_N_RESULTS,
    app_context: dict | None = None,
) -> tuple[list[dict], str]:
    """Try MCP knowledge_server first; fall back to direct ChromaStore."""
    app_filter = _build_app_filter(app_context)

    try:
        from grokly.mcp_servers.server_manager import MCPServerManager
        mgr = MCPServerManager()
        raw = mgr.call_tool("knowledge", "search_knowledge", {
            "query":      question,
            "chunk_type": chunk_type or "all",
            "n_results":  n,
        })
        mcp_results = json.loads(raw)
        chunks = [
            {
                "text":     r["text"],
                "metadata": {
                    "source":        r.get("source", "unknown"),
                    "chunk_type":    r.get("chunk_type", "unknown"),
                    "module":        r.get("module", ""),
                    "function_name": r.get("function_name", ""),
                    "file_path":     r.get("file_path", ""),
                },
                "distance": r.get("distance", 1.0),
            }
            for r in mcp_results
        ]
        logger.debug("Detective used MCP knowledge_server")
        return chunks, "mcp"
    except Exception as exc:
        logger.debug("MCP unavailable, falling back to direct ChromaStore: %s", exc)

    store = ChromaStore()
    chunks = _retrieve(store, question, chunk_type, n, app_filter=app_filter)
    return chunks, "direct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_chunk_type(question: str, role: str) -> str | None:
    q_lower = question.lower()
    if any(kw in q_lower for kw in _IMPACT_KEYWORDS):
        return "call_graph"
    return _ROLE_CHUNK_TYPE.get(role)


def _build_app_filter(app_context: dict | None) -> dict | None:
    if not app_context or app_context.get("search_all"):
        return None
    app_key = app_context.get("application", "")
    return {"application": {"$eq": app_key}} if app_key else None


def _combine_filters(f1: dict | None, f2: dict | None) -> dict | None:
    """Combine two ChromaDB where-clauses with $and, handling None cases."""
    if f1 and f2:
        return {"$and": [f1, f2]}
    return f1 or f2


def _retrieve(
    store: ChromaStore,
    query: str,
    chunk_type: str | None,
    n: int,
    app_filter: dict | None = None,
) -> list[dict]:
    type_filter = {"chunk_type": {"$eq": chunk_type}} if chunk_type else None
    where = _combine_filters(type_filter, app_filter)
    if where:
        return store.query(query, n_results=n, where=where)
    return store.query(query, n_results=n)


def _score_confidence(chunks: list[dict]) -> float:
    """Convert cosine distances to a 0–1 confidence score."""
    if not chunks:
        return 0.0
    distances = [c.get("distance", 1.0) for c in chunks[:5]]
    avg_distance = sum(distances) / len(distances)
    return round(max(0.0, min(1.0, 1.0 - avg_distance)), 3)
