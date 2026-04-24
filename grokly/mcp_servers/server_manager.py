"""
grokly/mcp_servers/server_manager.py

MCPServerManager — starts MCP servers and calls their tools.

Each call_tool() invocation starts the server subprocess, makes one tool call,
and closes the connection.  This is correct for per-request usage; connection
pooling can be added later for high-frequency use.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent

from mcp import ClientSession, StdioServerParameters, stdio_client


class MCPServerManager:

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        config_path = Path(config_path)

        with open(config_path, encoding="utf-8") as fh:
            raw = json.load(fh)

        self.servers: dict[str, dict] = raw["servers"]
        # Resolve the Python executable from the current process so servers
        # use the same venv.
        self._python = sys.executable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on the named MCP server and return the text result.

        Starts the server subprocess, makes the call, shuts it down.
        Raises RuntimeError if the server is unknown.
        Raises McpError / Exception on tool-level failures.
        """
        if server_name not in self.servers:
            raise RuntimeError(
                f"Unknown server '{server_name}'. "
                f"Available: {list(self.servers)}"
            )

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._call_tool_async(server_name, tool_name, arguments)
            )
        finally:
            loop.close()

    def list_available_tools(self) -> dict[str, list[str]]:
        """Return tool names per server, queried live from each server."""
        result: dict[str, list[str]] = {}
        for server_name in self.servers:
            loop = asyncio.new_event_loop()
            try:
                names = loop.run_until_complete(self._list_tools_async(server_name))
                result[server_name] = names
            except Exception as exc:
                result[server_name] = [f"ERROR: {exc}"]
            finally:
                loop.close()
        return result

    # ------------------------------------------------------------------
    # Internal async helpers
    # ------------------------------------------------------------------

    def _build_params(self, server_name: str) -> StdioServerParameters:
        cfg = self.servers[server_name]
        # Resolve the server script to an absolute path from project root
        raw_args = cfg.get("args", [])
        abs_args = [
            str((_PROJECT_ROOT / a).resolve()) if not Path(a).is_absolute() else a
            for a in raw_args
        ]
        return StdioServerParameters(
            command=self._python,
            args=abs_args,
            env=None,  # inherit parent env (includes ANTHROPIC_API_KEY etc.)
        )

    async def _call_tool_async(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> str:
        params = self._build_params(server_name)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

        if result.isError:
            raise RuntimeError(f"Tool error from {server_name}/{tool_name}")

        texts = [
            block.text
            for block in result.content
            if hasattr(block, "text")
        ]
        return "\n".join(texts)

    async def _list_tools_async(self, server_name: str) -> list[str]:
        params = self._build_params(server_name)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()

        return [t.name for t in tools_result.tools]
