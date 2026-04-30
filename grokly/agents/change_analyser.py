"""
grokly/agents/change_analyser.py — Impact analysis for changed Python files.

Queries ChromaDB to find which chunks already exist for the changed files,
counts affected functions, and estimates the API cost of a selective update.

No API calls are made here — read-only ChromaDB access only.

Public API
----------
    analyser = ChangeAnalyserAgent(store)
    plan = analyser.analyse_changes(changed_files, repo_path, module_name)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from grokly.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# Approximate cost per function: ~300 input + ~200 output tokens
# claude-sonnet-4-6: ~$0.003/1k input, ~$0.015/1k output
_COST_PER_FUNCTION = 0.006  # conservative estimate


class ChangeAnalyserAgent:
    """Analyses changed files and builds an update plan without calling Claude."""

    def __init__(self, store: ChromaStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_qualifying_functions(self, abs_path: Path) -> int:
        """Count AST-parseable functions in *abs_path* (same rules as ingester)."""
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
            tree    = ast.parse(content)
        except Exception:
            return 0

        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            if name.startswith("test_"):
                continue
            line_count = node.end_lineno - node.lineno + 1
            if line_count >= 5:
                count += 1

        return count

    def _get_existing_chunk_ids(self, rel_path: str) -> list[str]:
        """
        Return IDs of all ChromaDB chunks whose file_path matches *rel_path*.

        Handles both forward-slash and backslash variants.
        """
        variants = [rel_path, rel_path.replace("/", "\\"), rel_path.replace("\\", "/")]
        found_ids: list[str] = []

        for variant in set(variants):
            try:
                result = self._store._collection.get(
                    where={"file_path": {"$eq": variant}},
                    include=[],
                )
                found_ids.extend(result.get("ids", []))
            except Exception:
                pass

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for cid in found_ids:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)

        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_changes(
        self,
        changed_files: list[str],
        repo_path: Path,
        module_name: str = "unknown",
    ) -> dict:
        """
        Build an update plan for *changed_files*.

        Parameters
        ----------
        changed_files : list[str]
            Relative paths (from repo root) of Python files that changed.
        repo_path : Path
            Absolute path to the repo root (used to resolve files for AST).
        module_name : str
            Human-readable label for logging.

        Returns
        -------
        dict with keys:
            files_to_update   — list of dicts with per-file details
            total_functions   — int
            total_existing    — int  (chunks to delete)
            estimated_cost    — float
            skipped_files     — list[str]  (missing / unreadable)
        """
        files_to_update: list[dict] = []
        skipped_files:   list[str]  = []
        total_functions  = 0
        total_existing   = 0

        for rel_path in changed_files:
            abs_path = repo_path / rel_path
            if not abs_path.exists():
                skipped_files.append(rel_path)
                continue

            fn_count  = self._count_qualifying_functions(abs_path)
            chunk_ids = self._get_existing_chunk_ids(rel_path)

            total_functions += fn_count
            total_existing  += len(chunk_ids)

            files_to_update.append({
                "rel_path":       rel_path,
                "abs_path":       str(abs_path),
                "function_count": fn_count,
                "existing_ids":   chunk_ids,
                "existing_count": len(chunk_ids),
                "module":         module_name,
            })

            logger.debug(
                "%s: %d functions, %d existing chunks",
                rel_path, fn_count, len(chunk_ids),
            )

        estimated_cost = total_functions * _COST_PER_FUNCTION

        return {
            "files_to_update":  files_to_update,
            "total_functions":  total_functions,
            "total_existing":   total_existing,
            "estimated_cost":   estimated_cost,
            "skipped_files":    skipped_files,
        }
