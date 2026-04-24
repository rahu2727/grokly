"""
grokly/pipeline/tools.py — Tool definitions for The Tracker agent.

Each tool wraps ChromaStore.query() with a chunk_type or source filter.
Falls back to unfiltered search when the filtered collection returns nothing.

Public API:
    TOOL_DEFINITIONS  — Anthropic tool-calling schema list
    execute_tool()    — dispatches tool_name → ChromaStore call
"""

from typing import Any

from grokly.store.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Anthropic tool-calling schema
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "search_commentary",
        "description": (
            "Search business-friendly commentary and explanations of ERPNext features. "
            "Best for business users and managers who need plain-language answers "
            "without code. Use for 'how do I', 'what is', 'why does' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "module": {
                    "type": "string",
                    "description": (
                        "Optional ERPNext module to filter by: "
                        "hr | accounts | stock | buying | projects | payroll | system"
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_raw_code",
        "description": (
            "Search ERPNext source code, Python docstrings, and technical implementations. "
            "Best for developers who need to understand internals, hooks, controllers, "
            "or DocType field definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The technical search query"},
                "module": {
                    "type": "string",
                    "description": "Optional ERPNext module to narrow the search",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_call_graph",
        "description": (
            "Search function call graphs and code-level relationships in ERPNext. "
            "Use for impact analysis, tracing what a function triggers, "
            "or understanding dependencies between methods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function or method to look up",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "search_docs",
        "description": (
            "Search official ERPNext documentation pages. "
            "Best for configuration guides, setup walkthroughs, "
            "and feature overview articles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The documentation search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_forum",
        "description": (
            "Search ERPNext community forum Q&A pairs. "
            "Best for troubleshooting, edge cases, workarounds, "
            "and real-world usage questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The forum search query"},
                "module": {
                    "type": "string",
                    "description": "Optional ERPNext module to filter by",
                },
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    fallback_query: str = "",
) -> list[dict]:
    """
    Execute a named tool and return a list of retrieved chunk dicts.

    Each chunk: {text: str, metadata: dict, distance: float}
    """
    store = ChromaStore()

    if tool_name == "search_commentary":
        return _by_chunk_type(
            store,
            query=tool_input.get("query", fallback_query),
            chunk_type="commentary",
            module=tool_input.get("module"),
        )

    if tool_name == "search_raw_code":
        return _by_chunk_type(
            store,
            query=tool_input.get("query", fallback_query),
            chunk_type="raw_code",
            module=tool_input.get("module"),
        )

    if tool_name == "search_call_graph":
        fn = tool_input.get("function_name", fallback_query)
        return _by_chunk_type(store, query=fn, chunk_type="call_graph", module=None)

    if tool_name == "search_docs":
        return _by_source(store, query=tool_input.get("query", fallback_query), source="docs")

    if tool_name == "search_forum":
        query = tool_input.get("query", fallback_query)
        module = tool_input.get("module")
        if module:
            where = {"$and": [{"source": {"$eq": "forum"}}, {"module": {"$eq": module}}]}
            results = store.query(query, n_results=5, where=where)
            if results:
                return results
        return _by_source(store, query=query, source="forum")

    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_chunk_type(
    store: ChromaStore, query: str, chunk_type: str, module: str | None
) -> list[dict]:
    where: dict = {"chunk_type": {"$eq": chunk_type}}
    if module:
        where = {"$and": [{"chunk_type": {"$eq": chunk_type}}, {"module": {"$eq": module}}]}
    return store.query(query, n_results=5, where=where)


def _by_source(store: ChromaStore, query: str, source: str) -> list[dict]:
    return store.query(query, n_results=5, where={"source": {"$eq": source}})
