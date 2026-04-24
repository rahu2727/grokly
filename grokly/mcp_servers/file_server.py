"""
grokly/mcp_servers/file_server.py

MCP server for reading source files from the ingested repositories.

Tools:
  read_source_file   — read a file from erpnext or hrms repo
  list_module_files  — list Python files in a module
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

server = Server("grokly-file")

# Repo name → local directory relative to project root
_REPO_DIRS = {
    "erpnext": Path("data/erpnext_repo"),
    "hrms":    Path("data/hrms_repo"),
}

# Module → (repo_key, path_candidates)
_MODULE_PATHS: dict[str, tuple[str, list[str]]] = {
    "buying":   ("erpnext", ["erpnext/buying"]),
    "projects": ("erpnext", ["erpnext/projects"]),
    "hr":       ("hrms",    ["hrms/hr", "hrms/hrms/hr", "hr"]),
    "payroll":  ("hrms",    ["hrms/payroll", "hrms/hrms/payroll", "payroll"]),
}


def _resolve_module_dir(module: str) -> Path | None:
    if module not in _MODULE_PATHS:
        return None
    repo_key, candidates = _MODULE_PATHS[module]
    repo_dir = _PROJECT_ROOT / _REPO_DIRS[repo_key]
    for candidate in candidates:
        full = repo_dir / candidate
        if full.is_dir():
            return full
    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="read_source_file",
            description="Read a source code file from the ingested repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path from the repo root (e.g. erpnext/buying/doctype/purchase_order/purchase_order.py)",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository: erpnext or hrms",
                    },
                },
                "required": ["file_path", "repo"],
            },
        ),
        Tool(
            name="list_module_files",
            description="List all Python files in a module",
            inputSchema={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Module name: hr | payroll | buying | projects",
                    },
                },
                "required": ["module"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "read_source_file":
        return await _read_source_file(arguments)
    if name == "list_module_files":
        return await _list_module_files(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _read_source_file(arguments: dict) -> list[TextContent]:
    file_path = arguments.get("file_path", "").strip()
    repo = arguments.get("repo", "").strip().lower()

    if not file_path or not repo:
        return [TextContent(type="text", text="Error: file_path and repo are required")]

    if ".." in file_path:
        return [TextContent(type="text", text="Error: path traversal not allowed")]

    if repo not in _REPO_DIRS:
        return [TextContent(type="text", text=f"Error: unknown repo '{repo}'. Use: erpnext or hrms")]

    full_path = _PROJECT_ROOT / _REPO_DIRS[repo] / file_path

    if not full_path.exists():
        return [TextContent(type="text", text=f"File not found: {file_path} in {repo}")]

    if not full_path.is_file():
        return [TextContent(type="text", text=f"Not a file: {file_path}")]

    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [TextContent(type="text", text=f"Error reading file: {exc}")]

    result = {
        "repo": repo,
        "file_path": file_path,
        "size_bytes": len(content.encode("utf-8")),
        "content": content,
    }
    return [TextContent(type="text", text=json.dumps(result))]


async def _list_module_files(arguments: dict) -> list[TextContent]:
    module = arguments.get("module", "").strip().lower()

    if not module:
        return [TextContent(type="text", text="Error: module is required")]

    module_dir = _resolve_module_dir(module)
    if module_dir is None:
        result = {
            "module": module,
            "file_count": 0,
            "files": [],
            "error": (
                f"Module directory not found for '{module}'. "
                "Repositories may not be cloned. "
                f"Configured modules: {', '.join(_MODULE_PATHS)}"
            ),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    repo_key = _MODULE_PATHS[module][0]
    repo_dir = _PROJECT_ROOT / _REPO_DIRS[repo_key]

    py_files = sorted(
        str(f.relative_to(repo_dir)) for f in module_dir.rglob("*.py")
    )

    result = {
        "module": module,
        "repo": repo_key,
        "module_path": str(module_dir.relative_to(_PROJECT_ROOT)),
        "file_count": len(py_files),
        "files": py_files,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
