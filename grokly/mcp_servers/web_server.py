"""
grokly/mcp_servers/web_server.py

MCP server wrapping Tavily web search.
Use only when ChromaDB confidence is low — live web call, costs tokens.

Tools:
  web_search — search the web via Tavily API
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

load_dotenv()

server = Server("grokly-web")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web for current information about ERPNext or enterprise systems. "
                "Use only when knowledge base search returns low confidence results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 3, max 5)",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "web_search":
        return await _web_search(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _web_search(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    max_results = min(int(arguments.get("max_results", 3)), 5)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [TextContent(
            type="text",
            text="Error: TAVILY_API_KEY not set in environment. Web search unavailable.",
        )]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
    except Exception as exc:
        return [TextContent(type="text", text=f"Web search error: {exc}")]

    results = []
    for item in response.get("results", []):
        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "content": item.get("content", "")[:500],
            "score":   round(item.get("score", 0.0), 4),
        })

    output = {"query": query, "results": results}
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
