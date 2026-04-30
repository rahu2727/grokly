"""
grokly/model_config.py — Model configuration for GroklyAI.

Hierarchy (each level overrides the one above):
  Level 1 — GROKLY_DEFAULT_MODEL in .env
  Level 2 — Agent-specific env var in .env

To change a model:
  Edit .env — takes effect on next app restart.
  No code changes ever needed.

To add a new agent:
  Add its key to AGENT_MODEL_KEYS and AGENT_MAX_TOKENS below.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ── Default model ──────────────────────────────────────────────────────
DEFAULT_MODEL: str = os.getenv(
    "GROKLY_DEFAULT_MODEL",
    "claude-sonnet-4-6",  # fallback if env not set
)

# ── Agent model keys ────────────────────────────────────────────────────
# Maps agent name → its env variable name.
# Add new agents here as the system grows.

AGENT_MODEL_KEYS: dict[str, str] = {
    "commentary": "GROKLY_MODEL_COMMENTARY",
    "counsel":    "GROKLY_MODEL_COUNSEL",
    "tracker":    "GROKLY_MODEL_TRACKER",
    "briefer":    "GROKLY_MODEL_BRIEFER",
    "memory":     "GROKLY_MODEL_MEMORY",
    "proactive":  "GROKLY_MODEL_PROACTIVE",
    "monitor":    "GROKLY_MODEL_MONITOR",
    "detective":  "GROKLY_MODEL_DETECTIVE",
}

# ── Max tokens per agent ────────────────────────────────────────────────
# Tune per agent based on expected output size.

AGENT_MAX_TOKENS: dict[str, int] = {
    "commentary": 400,   # function explanation
    "counsel":    1500,  # full answer with citations
    "tracker":    512,   # tool selection only
    "briefer":    1024,  # role-adapted final answer
    "memory":     256,   # compression / reference resolution
    "proactive":  400,   # suggestions list
    "monitor":    300,   # change analysis
    "detective":  100,   # confidence score only
}


def get_model(agent_name: str) -> str:
    """
    Return the model for *agent_name*.

    Checks the agent-specific env var first; falls back to DEFAULT_MODEL.

        from grokly.model_config import get_model
        model = get_model("counsel")   # "claude-sonnet-4-6" or env override
    """
    env_key = AGENT_MODEL_KEYS.get(agent_name)
    if env_key:
        override = os.getenv(env_key, "").strip()
        if override:
            return override
    return DEFAULT_MODEL


def get_max_tokens(agent_name: str) -> int:
    """Return the max_tokens cap for *agent_name*."""
    return AGENT_MAX_TOKENS.get(agent_name, 1000)


def get_agent_config(agent_name: str) -> dict:
    """
    Return {"model": str, "max_tokens": int, "agent": str} for *agent_name*.

        cfg = get_agent_config("counsel")
        client.messages.create(model=cfg["model"], max_tokens=cfg["max_tokens"], ...)
    """
    return {
        "model":      get_model(agent_name),
        "max_tokens": get_max_tokens(agent_name),
        "agent":      agent_name,
    }


def print_model_summary() -> None:
    """Print a table of every agent's active model assignment."""
    print("\n" + "=" * 58)
    print("GroklyAI — Model Configuration")
    print("=" * 58)
    print(f"  Default model: {DEFAULT_MODEL}")
    print()
    print(f"  {'Agent':<14} {'Model':<36} {'Tokens':>6}")
    print(f"  {'-'*14} {'-'*36} {'-'*6}")

    for agent in AGENT_MODEL_KEYS:
        model   = get_model(agent)
        tokens  = get_max_tokens(agent)
        env_key = AGENT_MODEL_KEYS[agent]
        marker  = "*" if os.getenv(env_key, "").strip() else " "
        print(f"  {marker} {agent:<13} {model:<36} {tokens:>6}")

    print()
    print("  * = agent-specific override in .env")
    print("=" * 58 + "\n")
