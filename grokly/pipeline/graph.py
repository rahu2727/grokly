"""
grokly/pipeline/graph.py — LangGraph StateGraph connecting all four Grokly agents.

Pipeline:
    START → detective → tracker → counsel → briefer → END

Conditional edges:
    detective → tracker (always; tracker reads the needs_reretrieval flag)
    counsel   → counsel (retry once if quality_score < 3.0)
              → briefer (when quality is acceptable or max retries reached)

Max retries:
    Tracker:  3 internal iterations (controlled inside tracker_node)
    Counsel:  1 retry via graph conditional edge (2 total Counsel calls max)
"""

from langgraph.graph import END, START, StateGraph

from grokly.agents.briefer import briefer_node
from grokly.agents.counsel import counsel_node
from grokly.agents.detective import detective_node
from grokly.agents.tracker import tracker_node
from grokly.pipeline.state import GroklyState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_detective(state: GroklyState) -> str:
    """Both confidence paths lead to tracker; tracker checks needs_reretrieval."""
    return "tracker"


def _route_after_counsel(state: GroklyState) -> str:
    """Retry counsel once if quality is below threshold; otherwise proceed to briefer."""
    quality = state.get("quality_score", 3.0)
    retries = state.get("counsel_retries", 0)
    if quality < 3.0 and retries <= 1:
        return "counsel"
    return "briefer"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Construct and compile the Grokly StateGraph."""
    builder = StateGraph(GroklyState)

    builder.add_node("detective", detective_node)
    builder.add_node("tracker",   tracker_node)
    builder.add_node("counsel",   counsel_node)
    builder.add_node("briefer",   briefer_node)

    builder.add_edge(START, "detective")

    builder.add_conditional_edges(
        "detective",
        _route_after_detective,
        {"tracker": "tracker"},
    )

    builder.add_edge("tracker", "counsel")

    builder.add_conditional_edges(
        "counsel",
        _route_after_counsel,
        {"counsel": "counsel", "briefer": "briefer"},
    )

    builder.add_edge("briefer", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graph = None


def get_graph():
    """Return the compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
