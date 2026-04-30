"""
grokly/agents/change_monitor.py — Git-based change detection for ingested repos.

Tracks the last-ingested commit per repo and returns a list of changed Python
files since that checkpoint. Does NOT call any external API — pure git + filesystem.

Public API
----------
    monitor = ChangeMonitorAgent()
    status  = monitor.check_all_repos()   # dict[repo_name -> per-repo status]
    monitor.save_last_checked_commit("erpnext_main", commit_hash)
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from grokly.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# State file lives beside this module so it is always found regardless of CWD.
_STATE_FILE = Path(__file__).parent / "monitor_state.json"

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class ChangeMonitorAgent:
    """Detects which Python files have changed since the last ingestion checkpoint."""

    def __init__(self, config_loader: ConfigLoader | None = None) -> None:
        self._cfg = config_loader or ConfigLoader()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if _STATE_FILE.exists():
            try:
                return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_state(self, state: dict) -> None:
        _STATE_FILE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_last_checked_commit(self, repo_name: str) -> str | None:
        """Return the last commit hash that was fully ingested for *repo_name*."""
        state = self._load_state()
        return state.get(repo_name, {}).get("commit_hash")

    def save_last_checked_commit(self, repo_name: str, commit_hash: str) -> None:
        """Persist *commit_hash* as the ingestion checkpoint for *repo_name*."""
        state = self._load_state()
        state.setdefault(repo_name, {})
        state[repo_name]["commit_hash"] = commit_hash
        state[repo_name]["updated_at"]  = datetime.now(timezone.utc).isoformat()
        self._save_state(state)
        logger.debug("Saved checkpoint: %s → %s", repo_name, commit_hash[:8])

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def get_current_commit(self, repo_path: Path) -> str | None:
        """Return HEAD commit hash for the repo at *repo_path*, or None on error."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as exc:
            logger.debug("get_current_commit failed for %s: %s", repo_path, exc)
        return None

    def get_changed_files(
        self,
        repo_path: Path,
        since_commit: str,
    ) -> list[str]:
        """
        Return relative paths of .py files changed between *since_commit* and HEAD.

        Excludes test files (test_*.py, *_test.py, tests/ directories).
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{since_commit}..HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            logger.debug("get_changed_files failed: %s", exc)
            return []

        if result.returncode != 0:
            return []

        files = []
        for line in result.stdout.splitlines():
            line = line.strip().replace("\\", "/")
            if not line.endswith(".py"):
                continue
            # Exclude test files
            parts = line.split("/")
            if any(p in ("tests", "test") for p in parts):
                continue
            name = parts[-1]
            if name.startswith("test_") or name.endswith("_test.py"):
                continue
            files.append(line)

        return files

    def get_commit_summary(
        self,
        repo_path: Path,
        since_commit: str,
    ) -> list[str]:
        """Return one-line git log entries between *since_commit* and HEAD."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", f"{since_commit}..HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return []

        if result.returncode != 0:
            return []

        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check_all_repos(self) -> dict[str, dict]:
        """
        Check all enabled repos and return a status dict.

        Return value structure per repo:
            {
                "repo_name":        str,
                "repo_path":        str,
                "current_commit":   str | None,
                "last_commit":      str | None,
                "has_changes":      bool,
                "changed_files":    list[str],
                "commit_summary":   list[str],
                "is_first_run":     bool,
                "error":            str | None,
            }
        """
        results: dict[str, dict] = {}

        for repo in self._cfg.get_enabled_repos():
            name      = repo["name"]
            repo_path = (_PROJECT_ROOT / repo["local_path"]).resolve()

            entry: dict = {
                "repo_name":      name,
                "repo_path":      str(repo_path),
                "current_commit": None,
                "last_commit":    None,
                "has_changes":    False,
                "changed_files":  [],
                "commit_summary": [],
                "is_first_run":   False,
                "error":          None,
            }

            if not repo_path.exists():
                entry["error"] = f"Repo not found at {repo_path}"
                results[name]  = entry
                continue

            current = self.get_current_commit(repo_path)
            if current is None:
                entry["error"] = "Could not read HEAD commit"
                results[name]  = entry
                continue

            entry["current_commit"] = current
            last = self.get_last_checked_commit(name)
            entry["last_commit"] = last

            if last is None:
                entry["is_first_run"] = True
                entry["has_changes"]  = False  # nothing to diff yet
                results[name] = entry
                continue

            if last == current:
                results[name] = entry  # up to date
                continue

            entry["changed_files"]  = self.get_changed_files(repo_path, last)
            entry["commit_summary"] = self.get_commit_summary(repo_path, last)
            entry["has_changes"]    = len(entry["changed_files"]) > 0

            results[name] = entry

        return results
