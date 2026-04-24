"""
grokly/config_loader.py — Loads all source configuration from grokly/config/*.json.

Three files drive the entire ingestion pipeline:
    sources_qna.json   — forum Q&A seed data
    sources_docs.json  — documentation URLs to crawl
    sources_code.json  — source code repositories and modules
"""

from __future__ import annotations

import json
from pathlib import Path


class ConfigLoader:
    """
    Loads and exposes configuration from grokly/config/*.json.

    Usage
    -----
        from grokly.config_loader import ConfigLoader
        cfg = ConfigLoader()
        urls = cfg.get_enabled_doc_urls()
        print(cfg.summary())
    """

    def __init__(self, config_dir: str = "grokly/config") -> None:
        # Resolve relative paths against the project root.
        # Absolute paths are used as-is.
        _project_root = Path(__file__).parent.parent
        p = Path(config_dir)
        self.config_dir: Path = p if p.is_absolute() else (_project_root / config_dir)

        self.qna  = self._load("sources_qna.json")
        self.docs = self._load("sources_docs.json")
        self.code = self._load("sources_code.json")

    def _load(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Expected directory : {self.config_dir}\n"
                f"Run from the project root (grokly/)."
            )
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------------

    def get_enabled_doc_urls(self) -> list[str]:
        """Flat list of all URLs from all enabled documentation sources."""
        urls: list[str] = []
        for source in self.docs.get("sources", []):
            if source.get("enabled", True):
                urls.extend(source.get("urls", []))
        return urls

    def get_crawl_settings(self) -> dict:
        """Crawl settings from sources_docs.json (delay, timeout, user_agent …)."""
        return self.docs.get("crawl_settings", {})

    # ------------------------------------------------------------------
    # Source-code repositories
    # ------------------------------------------------------------------

    def get_enabled_repos(self) -> list[dict]:
        """Enabled repository dicts from sources_code.json."""
        return [
            r for r in self.code.get("repositories", [])
            if r.get("enabled", True)
        ]

    def get_enabled_modules(self, repo_name: str) -> list[dict]:
        """Enabled module dicts for the named repository."""
        for repo in self.code.get("repositories", []):
            if repo["name"] == repo_name:
                return [
                    m for m in repo.get("modules", [])
                    if m.get("enabled", True)
                ]
        return []

    def get_chunking_settings(self) -> dict:
        """Function-level chunking rules from sources_code.json."""
        return self.code.get("chunking", {})

    def get_commentary_settings(self) -> dict:
        """Commentary-agent settings from sources_code.json."""
        return self.code.get("commentary_agent", {})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable configuration overview printed at ingestion startup."""
        from grokly.brand import APP_NAME
        lines: list[str] = [
            f"{APP_NAME} — Source Configuration",
            "=" * 40,
        ]

        total_urls = len(self.get_enabled_doc_urls())
        enabled_doc_sources = sum(
            1 for s in self.docs.get("sources", []) if s.get("enabled", True)
        )
        lines.append(
            f"Docs       : {total_urls} URLs across "
            f"{enabled_doc_sources} enabled source(s)"
        )

        for repo in self.get_enabled_repos():
            modules = self.get_enabled_modules(repo["name"])
            names = ", ".join(m["name"] for m in modules)
            lines.append(
                f"Repo       : {repo['name']}  "
                f"({len(modules)} enabled module(s): {names})"
            )

        ca = self.get_commentary_settings()
        if ca.get("enabled", False):
            lines.append(
                f"Commentary : enabled  "
                f"(model: {ca.get('model', 'unknown')}, "
                f"cost warning at ${ca.get('cost_warning_threshold_usd', 5):.2f})"
            )
        else:
            lines.append("Commentary : disabled")

        lines.append("=" * 40)
        return "\n".join(lines)
