"""
grokly/env_loader.py — Layered environment configuration loader.

Usage:
    # In any entry point (ingest.py, app/main.py):
    from grokly.env_loader import load_environment
    load_environment()

This loads the base .env first, then overlays
the environment-specific file based on
GROKLY_ENV variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> str:
    """
    Load environment configuration in order:
    1. Base .env (API keys, secrets)
    2. Environment-specific overlay
       (model aliases, feature flags)

    The environment is determined by GROKLY_ENV
    in the base .env file.
    Default: development

    Returns the active environment name.
    """
    # Load base .env first (has API keys) — don't override already-set vars
    load_dotenv(".env", override=False)

    # Determine environment
    env = os.getenv("GROKLY_ENV", "development")

    # Load environment-specific overlay — these override base .env
    env_file = Path(__file__).parent / "environments" / f"{env}.env"

    if env_file.exists():
        load_dotenv(env_file, override=True)
        label = os.getenv("GROKLY_ENV_LABEL", env)
        print(f"[Config] Environment: {label}")
    else:
        print(f"[Config] No overlay found for environment: {env}")
        print(f"[Config] Using base .env only")

    return env
