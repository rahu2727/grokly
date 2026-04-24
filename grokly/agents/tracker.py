"""
grokly/agents/tracker.py — The Tracker: ReAct loop for adaptive retrieval.

Responsibilities:
  - Reads retrieved_chunks and retrieval_confidence from the Detective
  - If confidence < 0.6, uses Anthropic tool_use to decide which search to call
  - Executes the chosen tool and merges new chunks into the state
  - Repeats up to MAX_TRACKER_ITERATIONS times or until confidence >= 0.6
  - Logs every tool call to tool_calls_made
"""

import json
import logging

import anthropic
from dotenv import load_dotenv

from grokly.pipeline.state import GroklyState
from grokly.pipeline.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

load_dotenv()

MAX_TRACKER_ITERATIONS = 3

_SYSTEM = """You are The Tracker, an ERPNext retrieval specialist.

Your job: examine the already-retrieved context and decide if it adequately answers
the question. If not, call exactly ONE search tool to find better information.

Tool guide:
  search_commentary  → plain-language explanations  (for business/manager roles)
  search_raw_code    → Python source code/docstrings (for developer/admin roles)
  search_call_graph  → function dependencies        (for impact/dependency questions)
  search_docs        → official ERPNext docs pages
  search_forum       → community Q&A (good fallback for practical questions)

If the current context already adequately answers the question, respond with
plain text only — do NOT call a tool."""


def tracker_node(state: GroklyState) -> dict:
    """ReAct loop: assess retrieval quality, call tools to improve context if needed."""
    question = state["user_question"]
    role = state.get("user_role", "business_user")
    chunks: list[dict] = list(state.get("retrieved_chunks", []))
    confidence: float = state.get("retrieval_confidence", 0.0)
    tool_calls_made: list[str] = list(state.get("tool_calls_made", []))
    iteration_count: int = state.get("iteration_count", 1)
    tracker_retries: int = state.get("tracker_retries", 0)

    # Skip re-retrieval if Detective already found good context
    if confidence >= 0.6 and not state.get("needs_reretrieval", False):
        return {"tracker_retries": tracker_retries}

    client = anthropic.Anthropic()

    while tracker_retries < MAX_TRACKER_ITERATIONS and confidence < 0.6:
        tracker_retries += 1
        iteration_count += 1

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=_SYSTEM,
            tools=TOOL_DEFINITIONS,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"User role: {role}\n"
                        f"Current retrieval confidence: {confidence:.2f} (target ≥ 0.60)\n\n"
                        f"Context retrieved so far:\n{_format_chunks(chunks)}\n\n"
                        "Call the best tool to improve retrieval, or confirm context is sufficient."
                    ),
                }
            ],
        )

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name: str = block.name
                tool_input: dict = block.input
                tool_calls_made.append(
                    f"tracker:{tool_name}({json.dumps(tool_input, separators=(',', ':'))})"
                )

                new_chunks = execute_tool(tool_name, tool_input, question)
                existing_texts = {c["text"] for c in chunks}
                for chunk in new_chunks:
                    if chunk["text"] not in existing_texts:
                        chunks.append(chunk)
                        existing_texts.add(chunk["text"])

                distances = [c.get("distance", 1.0) for c in chunks[:5]]
                if distances:
                    confidence = round(max(0.0, min(1.0, 1.0 - sum(distances) / len(distances))), 3)

                break  # One tool call per iteration (true ReAct step)

        if confidence >= 0.6:
            break

    # Last resort: if confidence is still low, try MCP web search
    if confidence < 0.6:
        chunks, tool_calls_made = _try_web_search_via_mcp(
            question, chunks, tool_calls_made
        )
        distances = [c.get("distance", 1.0) for c in chunks[:5]]
        if distances:
            confidence = round(max(0.0, min(1.0, 1.0 - sum(distances) / len(distances))), 3)

    sources = list({c["metadata"].get("source", "unknown") for c in chunks})

    return {
        "retrieved_chunks":    chunks,
        "retrieval_confidence": confidence,
        "sources":             sources,
        "tool_calls_made":     tool_calls_made,
        "iteration_count":     iteration_count,
        "tracker_retries":     tracker_retries,
        "needs_reretrieval":   False,
    }


def _try_web_search_via_mcp(
    question: str, chunks: list[dict], tool_calls_made: list[str]
) -> tuple[list[dict], list[str]]:
    """Attempt a Tavily web search via MCP web_server; silently skip on failure."""
    try:
        from grokly.mcp_servers.server_manager import MCPServerManager
        mgr = MCPServerManager()
        raw = mgr.call_tool("web", "web_search", {"query": question, "max_results": 3})
        data = json.loads(raw)
        web_results = data.get("results", [])
        tool_calls_made.append(f"tracker:web_search(mcp, query={question[:60]})")

        seen = {c["text"] for c in chunks}
        for r in web_results:
            text = f"{r.get('title', '')}\n{r.get('content', '')}"
            if text not in seen:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": "web",
                        "chunk_type": "web",
                        "url": r.get("url", ""),
                    },
                    "distance": max(0.0, 1.0 - r.get("score", 0.5)),
                })
                seen.add(text)

        logger.debug("Tracker added %d web results via MCP", len(web_results))
    except Exception as exc:
        logger.debug("MCP web search skipped: %s", exc)

    return chunks, tool_calls_made


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "  (none retrieved yet)"
    lines = []
    for i, c in enumerate(chunks[:5], 1):
        meta = c.get("metadata", {})
        dist = c.get("distance", "?")
        dist_str = f"{dist:.3f}" if isinstance(dist, float) else str(dist)
        preview = c["text"][:150].replace("\n", " ")
        lines.append(
            f"  [{i}] src={meta.get('source', '?')} "
            f"type={meta.get('chunk_type', 'general')} "
            f"dist={dist_str}: {preview}"
        )
    return "\n".join(lines)
