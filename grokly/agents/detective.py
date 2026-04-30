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

    n = _ROLE_N_RESULTS.get(role, _DEFAULT_N_RESULTS)
    chunk_type = _pick_chunk_type(question, role)

    chunks, method = _retrieve_with_mcp_fallback(question, chunk_type, role, n=n)

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
    tool_log = f"detective:retrieve(role={role}, chunk_type={chunk_type or 'all'}, via={method})"

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


def _retrieve_with_mcp_fallback(
    question: str, chunk_type: str | None, role: str, n: int = _DEFAULT_N_RESULTS
) -> tuple[list[dict], str]:
    """Try MCP knowledge_server first; fall back to direct ChromaStore."""
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
    chunks = _retrieve(store, question, chunk_type, n)
    return chunks, "direct"


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
