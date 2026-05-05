"""
grokly/model_config.py — Model configuration for GroklyAI.

Hierarchy (each level overrides the one above):
  Level 1 — GROKLY_DEFAULT_MODEL in .env
  Level 2 — Model alias resolution via GROKLY_ALIAS_* env vars
  Level 3 — Agent-specific env var (legacy override)

To change models across an environment:
  Edit grokly/environments/<env>.env — no code changes needed.

To add a new agent:
  Add its key to AGENT_MODEL_KEYS, AGENT_MAX_TOKENS, and AGENT_ALIAS_MAP.
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

# ── Agent model keys (legacy per-agent overrides) ──────────────────────
# Maps agent name → its env variable name.
# These take precedence over alias resolution.

AGENT_MODEL_KEYS: dict[str, str] = {
    "commentary": "GROKLY_MODEL_COMMENTARY",
    "counsel":    "GROKLY_MODEL_COUNSEL",
    "tracker":    "GROKLY_MODEL_TRACKER",
    "briefer":    "GROKLY_MODEL_BRIEFER",
    "memory":     "GROKLY_MODEL_MEMORY",
    "proactive":  "GROKLY_MODEL_PROACTIVE",
    "monitor":    "GROKLY_MODEL_MONITOR",
    "detective":  "GROKLY_MODEL_DETECTIVE",
    "screenshot": "GROKLY_MODEL_SCREENSHOT",
    "recording":  "GROKLY_MODEL_RECORDING",
}

# ── Max tokens per agent ────────────────────────────────────────────────

AGENT_MAX_TOKENS: dict[str, int] = {
    "commentary": 400,   # function explanation
    "counsel":    1500,  # full answer with citations
    "tracker":    512,   # tool selection only
    "briefer":    1024,  # role-adapted final answer
    "memory":     256,   # compression / reference resolution
    "proactive":  400,   # suggestions list
    "monitor":    300,   # change analysis
    "detective":  100,   # confidence score only
    "screenshot": 400,   # full screenshot description (6 categories)
    "recording":  300,   # per-frame description (2-3 sentences)
}

# ── Temperature per agent ───────────────────────────────────────────────
# 0.0 = fully deterministic (retrieval, analysis, scoring)
# 0.1 = slight variation (Briefer: more natural language output)

AGENT_TEMPERATURE: dict[str, float] = {
    "commentary": 0.0,
    "counsel":    0.0,
    "tracker":    0.0,
    "briefer":    0.1,
    "memory":     0.0,
    "proactive":  0.0,
    "monitor":    0.0,
    "detective":  0.0,
    "screenshot": 0.0,
    "recording":  0.0,
}

# ── Model aliases ──────────────────────────────────────────────────────
# Agents use aliases not model names.
# Aliases resolve to models via .env
# This decouples code from model choices.

MODEL_ALIAS_KEYS: dict[str, str] = {
    "quality_model":   "GROKLY_ALIAS_QUALITY_MODEL",
    "reasoning_model": "GROKLY_ALIAS_REASONING_MODEL",
    "fast_model":      "GROKLY_ALIAS_FAST_MODEL",
    "balanced_model":  "GROKLY_ALIAS_BALANCED_MODEL",
}

# Default resolution if env var not set
MODEL_ALIAS_DEFAULTS: dict[str, str] = {
    "quality_model":   "claude-sonnet-4-20250514",
    "reasoning_model": "claude-sonnet-4-20250514",
    "fast_model":      "claude-haiku-4-5-20251001",
    "balanced_model":  "claude-sonnet-4-20250514",
}

# ── Agent to alias mapping ──────────────────────────────────────────────
# Maps each agent to the alias it should use.
# Change the alias assignment here to change
# which quality tier an agent uses.

AGENT_ALIAS_MAP: dict[str, str] = {
    "commentary":  "quality_model",
    "counsel":     "reasoning_model",
    "briefer":     "balanced_model",
    "tracker":     "fast_model",
    "detective":   "fast_model",
    "memory":      "fast_model",
    "proactive":   "fast_model",
    "monitor":     "fast_model",
    "screenshot":  "quality_model",
    "recording":   "fast_model",
}


def resolve_alias(alias: str) -> str:
    """
    Resolve a model alias to a concrete model.

        model = resolve_alias("reasoning_model")
        # Returns whatever GROKLY_ALIAS_REASONING_MODEL
        # is set to in the current .env
    """
    env_key = MODEL_ALIAS_KEYS.get(alias)
    if env_key:
        resolved = os.getenv(env_key, "").strip()
        if resolved:
            return resolved
    return MODEL_ALIAS_DEFAULTS.get(alias, DEFAULT_MODEL)


def get_model(agent_name: str) -> str:
    """
    Get model for an agent.
    Priority:
      1. Agent-specific env var (legacy override)
      2. Alias resolution via AGENT_ALIAS_MAP
      3. DEFAULT_MODEL fallback
    """
    # Legacy: agent-specific env var takes highest priority
    env_key = AGENT_MODEL_KEYS.get(agent_name)
    if env_key:
        specific = os.getenv(env_key, "").strip()
        if specific:
            return specific

    # Resolve via alias map
    alias = AGENT_ALIAS_MAP.get(agent_name, "balanced_model")
    return resolve_alias(alias)


def get_max_tokens(agent_name: str) -> int:
    """Return the max_tokens cap for *agent_name*."""
    return AGENT_MAX_TOKENS.get(agent_name, 1000)


def get_temperature(agent_name: str) -> float:
    """Return the temperature setting for *agent_name* (default 0.0)."""
    return AGENT_TEMPERATURE.get(agent_name, 0.0)


def get_agent_config(agent_name: str) -> dict:
    """
    Return {"model", "max_tokens", "temperature", "agent"} for *agent_name*.

        cfg = get_agent_config("counsel")
        client.messages.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            ...)
    """
    return {
        "model":       get_model(agent_name),
        "max_tokens":  get_max_tokens(agent_name),
        "temperature": get_temperature(agent_name),
        "agent":       agent_name,
    }


def get_env_label() -> str:
    """Return current environment label."""
    return os.getenv("GROKLY_ENV_LABEL", "Development")


def _short(model: str) -> str:
    return (
        model
        .replace("claude-", "")
        .replace("-20250514", "")
        .replace("-20251001", "")
    )


def print_model_summary() -> None:
    """Print a table of every agent's active model assignment."""
    env = get_env_label()
    print(f"\n{'='*60}")
    print(f"GroklyAI — Model Configuration")
    print(f"Environment: {env}")
    print(f"{'='*60}")
    print(f"\n  {'Agent':<14} {'Alias':<18} {'Model':<35} {'Tokens':>6}")
    print(f"  {'-'*14} {'-'*18} {'-'*35} {'-'*6}")

    for agent, alias in AGENT_ALIAS_MAP.items():
        model  = get_model(agent)
        tokens = get_max_tokens(agent)
        # Mark agents with legacy per-agent override
        env_key = AGENT_MODEL_KEYS.get(agent, "")
        marker  = "*" if env_key and os.getenv(env_key, "").strip() else " "
        print(f"  {marker}{agent:<13} {alias:<18} {_short(model):<35} {tokens:>6}")

    print(f"\n  Alias resolution:")
    for alias, env_key in MODEL_ALIAS_KEYS.items():
        resolved = resolve_alias(alias)
        print(f"  {alias:<20} -> {_short(resolved)}")

    print(f"\n  * = agent-specific legacy override in .env")
    print(f"{'='*60}\n")
