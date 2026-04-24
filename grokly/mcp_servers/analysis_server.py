"""
grokly/mcp_servers/analysis_server.py

MCP server for function-level impact analysis using call graph data in ChromaDB.

Tools:
  get_function_callers    — which functions call a given function
  get_function_callees    — which functions a given function calls
  analyse_change_impact   — full impact report for changing a function
"""

import json
import os
import sys
from pathlib import Path

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

server = Server("grokly-analysis")


# ---------------------------------------------------------------------------
# Internal call graph helpers
# ---------------------------------------------------------------------------

def _get_all_call_graph_entries(store: ChromaStore) -> list[dict]:
    """Fetch every call_graph chunk from ChromaDB as a list of metadata dicts."""
    try:
        raw = store._collection.get(
            where={"chunk_type": {"$eq": "call_graph"}},
            include=["metadatas", "documents"],
        )
    except Exception:
        return []

    entries = []
    for meta, doc in zip(raw.get("metadatas", []), raw.get("documents", [])):
        entries.append({
            "function_name": meta.get("function_name", ""),
            "module":        meta.get("module", ""),
            "calls":         meta.get("calls", ""),
            "file_path":     meta.get("file_path", ""),
            "text":          doc,
        })
    return entries


def _parse_calls(calls_str: str) -> list[str]:
    if not calls_str or calls_str == "(none)":
        return []
    return [c.strip() for c in calls_str.split(",") if c.strip()]


def _find_callers(entries: list[dict], function_name: str, module: str | None) -> list[dict]:
    callers = []
    for e in entries:
        called = _parse_calls(e["calls"])
        if function_name in called:
            if module and e["module"] != module:
                continue
            callers.append({
                "function": e["function_name"],
                "module":   e["module"],
                "file":     e["file_path"],
            })
    return callers


def _find_callees(entries: list[dict], function_name: str) -> list[dict]:
    for e in entries:
        if e["function_name"] == function_name:
            called = _parse_calls(e["calls"])
            return [{"function": c} for c in called]
    return []


def _risk_level(n_callers: int, n_modules: int) -> str:
    if n_callers >= 6 or n_modules >= 3:
        return "High"
    if n_callers >= 3 or n_modules >= 2:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_function_callers",
            description="Find all functions that call a specific function — useful for impact analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to look up",
                    },
                    "module": {
                        "type": "string",
                        "description": "Optional: restrict to callers in this module",
                    },
                },
                "required": ["function_name"],
            },
        ),
        Tool(
            name="get_function_callees",
            description="Find all functions called by a specific function",
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to look up",
                    },
                },
                "required": ["function_name"],
            },
        ),
        Tool(
            name="analyse_change_impact",
            description=(
                "Analyse the impact of changing a specific function across the codebase. "
                "Returns direct callers, affected modules, risk level, and testing scope."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Function to analyse",
                    },
                    "module": {
                        "type": "string",
                        "description": "Module where the function lives (optional filter)",
                    },
                },
                "required": ["function_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_function_callers":
        return await _get_callers(arguments)
    if name == "get_function_callees":
        return await _get_callees(arguments)
    if name == "analyse_change_impact":
        return await _analyse_impact(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _get_callers(arguments: dict) -> list[TextContent]:
    fn = arguments.get("function_name", "").strip()
    module = arguments.get("module", "").strip() or None
    if not fn:
        return [TextContent(type="text", text="Error: function_name is required")]

    store = ChromaStore()
    entries = _get_all_call_graph_entries(store)
    callers = _find_callers(entries, fn, module)

    result = {
        "function_name": fn,
        "module_filter": module,
        "caller_count":  len(callers),
        "callers":       callers,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _get_callees(arguments: dict) -> list[TextContent]:
    fn = arguments.get("function_name", "").strip()
    if not fn:
        return [TextContent(type="text", text="Error: function_name is required")]

    store = ChromaStore()
    entries = _get_all_call_graph_entries(store)
    callees = _find_callees(entries, fn)

    result = {
        "function_name": fn,
        "callee_count":  len(callees),
        "callees":       callees,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _analyse_impact(arguments: dict) -> list[TextContent]:
    fn = arguments.get("function_name", "").strip()
    module = arguments.get("module", "").strip() or None
    if not fn:
        return [TextContent(type="text", text="Error: function_name is required")]

    store = ChromaStore()
    entries = _get_all_call_graph_entries(store)

    direct_callers = _find_callers(entries, fn, module=None)
    callees = _find_callees(entries, fn)

    affected_modules = sorted({c["module"] for c in direct_callers if c["module"]})
    n_callers = len(direct_callers)
    n_modules = len(affected_modules)
    risk = _risk_level(n_callers, n_modules)

    testing_scope = []
    if direct_callers:
        testing_scope.append(
            f"Test {n_callers} direct caller(s): "
            + ", ".join(c["function"] for c in direct_callers[:5])
            + (" ..." if n_callers > 5 else "")
        )
    if affected_modules:
        testing_scope.append(f"Regression test modules: {', '.join(affected_modules)}")
    if not testing_scope:
        testing_scope.append("No callers found — isolated function, low risk")

    report = {
        "function_name":   fn,
        "module":          module or "unknown",
        "direct_callers":  direct_callers,
        "callees":         callees,
        "indirect_impact": {
            "affected_modules": affected_modules,
            "module_count":     n_modules,
        },
        "risk_level":      risk,
        "testing_scope":   testing_scope,
        "summary": (
            f"Changing '{fn}' directly impacts {n_callers} function(s) "
            f"across {n_modules} module(s). Risk: {risk}."
        ),
    }
    return [TextContent(type="text", text=json.dumps(report, indent=2))]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
