"""
grokly/mcp_servers/knowledge_server.py

MCP server wrapping ChromaDB search.

Tools:
  search_knowledge  — semantic search with optional chunk_type / module filter
  get_chunk_stats   — total chunk count plus breakdown by chunk_type and module
"""

import json
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path and is the working directory
# so ChromaStore can find chroma_db/ and Grokly imports resolve.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

import asyncio

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from grokly.store.chroma_store import ChromaStore

load_dotenv()

server = Server("grokly-knowledge")

_CHUNK_TYPE_ALIASES = {"docs", "forum"}  # stored under `source`, not `chunk_type`


def _build_where(chunk_type: str | None, module: str | None) -> dict | None:
    clauses = []

    if chunk_type and chunk_type != "all":
        if chunk_type in _CHUNK_TYPE_ALIASES:
            clauses.append({"source": {"$eq": chunk_type}})
        else:
            clauses.append({"chunk_type": {"$eq": chunk_type}})

    if module:
        clauses.append({"module": {"$eq": module}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge",
            description="Search Grokly knowledge base for relevant information about any topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "chunk_type": {
                        "type": "string",
                        "description": (
                            "Optional filter: commentary | raw_code | call_graph "
                            "| docs | forum | all"
                        ),
                    },
                    "module": {
                        "type": "string",
                        "description": "Optional: hr | payroll | buying | projects",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 20)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_chunk_stats",
            description="Get statistics about what is in the Grokly knowledge base",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_knowledge":
        return await _search_knowledge(arguments)
    if name == "get_chunk_stats":
        return await _get_chunk_stats()
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search_knowledge(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    chunk_type = arguments.get("chunk_type") or None
    module = arguments.get("module") or None
    n_results = min(int(arguments.get("n_results", 5)), 20)

    store = ChromaStore()
    where = _build_where(chunk_type, module)

    try:
        results = store.query(query, n_results=n_results, where=where)
    except Exception as exc:
        return [TextContent(type="text", text=f"Search error: {exc}")]

    output = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        output.append({
            "rank": i,
            "text": r["text"],
            "source": meta.get("source", "unknown"),
            "chunk_type": meta.get("chunk_type", "unknown"),
            "module": meta.get("module", ""),
            "function_name": meta.get("function_name", ""),
            "file_path": meta.get("file_path", ""),
            "distance": round(r.get("distance", 1.0), 4),
        })

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _get_chunk_stats() -> list[TextContent]:
    store = ChromaStore()

    try:
        all_items = store._collection.get(include=["metadatas"])
    except Exception as exc:
        return [TextContent(type="text", text=f"Stats error: {exc}")]

    by_chunk_type: dict[str, int] = {}
    by_module: dict[str, int] = {}

    for meta in all_items["metadatas"]:
        ct = meta.get("chunk_type", meta.get("source", "unknown"))
        by_chunk_type[ct] = by_chunk_type.get(ct, 0) + 1

        mod = meta.get("module", "")
        if mod:
            by_module[mod] = by_module.get(mod, 0) + 1

    stats = {
        "total": store.count(),
        "by_chunk_type": dict(sorted(by_chunk_type.items(), key=lambda x: -x[1])),
        "by_module": dict(sorted(by_module.items(), key=lambda x: -x[1])),
    }

    return [TextContent(type="text", text=json.dumps(stats, indent=2))]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
