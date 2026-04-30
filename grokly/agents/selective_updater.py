"""
grokly/agents/selective_updater.py — Targeted re-ingestion for changed Python files.

Deletes existing ChromaDB chunks for a file, re-runs AST extraction, regenerates
commentary via the Claude API, and stores fresh chunks. Mirrors the patterns in
commentary_ingester.py but operates on individual files rather than whole modules.

Public API
----------
    updater = SelectiveUpdaterAgent(store)
    added   = updater.update_file(file_info, repo_path, dry_run=False)
    added   = updater.update_call_graph(changed_files, repo_path, module_name)
"""

from __future__ import annotations

import ast
import hashlib
import logging
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from grokly.ingestion.router_agent import RouterAgent
from grokly.prompt_loader import PromptLoader
from grokly.store.chroma_store import ChromaStore

load_dotenv()

logger = logging.getLogger(__name__)

_API_DELAY       = 0.5    # seconds between Claude calls
_COST_PER_FN     = 0.006  # rough estimate per function


# ---------------------------------------------------------------------------
# Chunk ID helpers (must match commentary_ingester.py exactly)
# ---------------------------------------------------------------------------


def _commentary_id(file_path: str, function_name: str) -> str:
    return hashlib.md5(
        f"commentary-{file_path}-{function_name}".encode("utf-8")
    ).hexdigest()


def _raw_code_id(file_path: str, function_name: str) -> str:
    return hashlib.md5(
        f"raw_code-{file_path}-{function_name}".encode("utf-8")
    ).hexdigest()


def _callgraph_id(file_path: str, function_name: str) -> str:
    return hashlib.md5(
        f"callgraph-{file_path}-{function_name}".encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _extract_functions(
    file_path: Path,
    min_lines: int = 5,
) -> list[tuple[str, str, int]]:
    """Return (name, source, line_count) for qualifying functions in file_path."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree    = ast.parse(content)
    except Exception:
        return []

    results: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if name.startswith("__") and name.endswith("__"):
            continue
        if name.startswith("test_"):
            continue
        line_count = node.end_lineno - node.lineno + 1
        if line_count < min_lines:
            continue
        source = ast.get_source_segment(content, node)
        if source is None:
            lines  = content.splitlines()
            source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if source and source.strip():
            results.append((name, source.strip(), line_count))

    return results


def _extract_call_graph(
    file_path: Path,
    min_lines: int = 5,
) -> list[tuple[str, list[str], int]]:
    """Return (fn_name, called_functions, line_count) for qualifying functions."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree    = ast.parse(content)
    except Exception:
        return []

    results: list[tuple[str, list[str], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if name.startswith("__") and name.endswith("__"):
            continue
        if name.startswith("test_"):
            continue
        line_count = node.end_lineno - node.lineno + 1
        if line_count < min_lines:
            continue
        called: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        results.append((name, sorted(called), line_count))

    return results


# ---------------------------------------------------------------------------
# Selective Updater
# ---------------------------------------------------------------------------


class SelectiveUpdaterAgent:
    """Deletes stale chunks and re-ingests a single changed Python file."""

    def __init__(self, store: ChromaStore) -> None:
        self._store        = store
        self._prompt_loader = PromptLoader()
        self._router        = RouterAgent()

    # ------------------------------------------------------------------
    # Delete helpers
    # ------------------------------------------------------------------

    def _delete_by_ids(self, ids: list[str]) -> int:
        """Delete chunk IDs from ChromaDB. Returns count deleted."""
        if not ids:
            return 0
        try:
            self._store._collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            logger.warning("Delete failed: %s", exc)
            return 0

    def _delete_all_for_file(self, rel_path: str) -> int:
        """
        Delete every chunk whose file_path matches rel_path (both slash variants).
        """
        variants = {rel_path, rel_path.replace("/", "\\"), rel_path.replace("\\", "/")}
        deleted  = 0
        for variant in variants:
            try:
                result = self._store._collection.get(
                    where={"file_path": {"$eq": variant}},
                    include=[],
                )
                ids = result.get("ids", [])
                deleted += self._delete_by_ids(ids)
            except Exception:
                pass
        return deleted

    # ------------------------------------------------------------------
    # Commentary regeneration
    # ------------------------------------------------------------------

    def _generate_commentary(
        self,
        client: anthropic.Anthropic,
        fn_name: str,
        rel_path: str,
        module_label: str,
        source_code: str,
    ) -> str | None:
        """Call Claude and return commentary text, or None on error."""
        language    = self._router.detect_language(rel_path, source_code)
        prompt_name = self._router.LANGUAGE_RULES[language]["prompt"]

        system_prompt = self._prompt_loader.get_system_prompt(prompt_name)
        user_prompt   = self._prompt_loader.format_user_prompt(
            prompt_name,
            function_name=fn_name,
            file_path=rel_path,
            module_name=module_label,
            function_source_code=source_code,
        )
        settings   = self._prompt_loader.get_settings(prompt_name)
        model      = settings.get("model",      "claude-sonnet-4-6")
        max_tokens = settings.get("max_tokens", 400)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            logger.warning("Commentary failed for %s.%s: %s", rel_path, fn_name, exc)
            return None

    # ------------------------------------------------------------------
    # Public: update one file
    # ------------------------------------------------------------------

    def update_file(
        self,
        file_info: dict,
        dry_run: bool = False,
    ) -> int:
        """
        Delete stale chunks and regenerate commentary + raw_code for one file.

        Parameters
        ----------
        file_info : dict
            A file entry from ChangeAnalyserAgent.analyse_changes() containing
            rel_path, abs_path, module, and existing_ids.
        dry_run : bool
            If True, report planned changes without touching ChromaDB or calling API.

        Returns
        -------
        int  — chunks added (0 in dry_run mode).
        """
        rel_path   = file_info["rel_path"]
        abs_path   = Path(file_info["abs_path"])
        module_lbl = file_info.get("module", "unknown")
        fn_count   = file_info.get("function_count", 0)
        ex_count   = file_info.get("existing_count", 0)

        print(
            f"  update_file: {rel_path} "
            f"({fn_count} functions, {ex_count} existing chunks)"
        )

        if dry_run:
            est = fn_count * _COST_PER_FN
            print(f"    [DRY RUN] would delete {ex_count} chunks, "
                  f"regenerate {fn_count} functions (~${est:.3f})")
            return 0

        # Delete stale chunks
        deleted = self._delete_all_for_file(rel_path)
        print(f"    Deleted {deleted} stale chunks")

        # Extract functions
        functions = _extract_functions(abs_path)
        if not functions:
            print(f"    No qualifying functions found in {rel_path}")
            return 0

        client = anthropic.Anthropic()
        added  = 0

        for fn_name, fn_source, line_count in functions:
            prompt_name    = self._router.LANGUAGE_RULES[
                self._router.detect_language(rel_path, fn_source)
            ]["prompt"]
            prompt_version = self._prompt_loader.get_version(prompt_name)
            expert         = self._router.get_expert_description(
                self._router.detect_language(rel_path, fn_source)
            )

            commentary_raw = self._generate_commentary(
                client, fn_name, rel_path, module_lbl, fn_source
            )

            if commentary_raw is None:
                time.sleep(_API_DELAY)
                continue

            commentary_text = (
                f"Function: {fn_name}\n"
                f"File: {rel_path}\n"
                f"Module: {module_lbl}\n\n"
                f"{commentary_raw}"
            )

            shared_meta = {
                "source":              "code_commentary",
                "file_path":           rel_path,
                "function_name":       fn_name,
                "module":              module_lbl,
                "prompt_name":         prompt_name,
                "prompt_version":      prompt_version,
                "expert_agent":        expert,
                "original_code_lines": line_count,
            }

            added += self._store.upsert(
                texts=[commentary_text],
                metadatas=[{**shared_meta, "chunk_type": "commentary"}],
                ids=[_commentary_id(rel_path, fn_name)],
            )
            added += self._store.upsert(
                texts=[fn_source],
                metadatas=[{**shared_meta, "chunk_type": "raw_code"}],
                ids=[_raw_code_id(rel_path, fn_name)],
            )

            time.sleep(_API_DELAY)

        print(f"    Added {added} new chunks for {len(functions)} functions")
        return added

    # ------------------------------------------------------------------
    # Public: update call graph for changed files
    # ------------------------------------------------------------------

    def update_call_graph(
        self,
        changed_files: list[str],
        repo_path: Path,
        module_name: str = "unknown",
        dry_run: bool    = False,
    ) -> int:
        """
        Rebuild call-graph chunks for *changed_files*. No API calls — AST only.

        Returns chunks added (0 in dry_run mode).
        """
        total_added = 0

        for rel_path in changed_files:
            abs_path = repo_path / rel_path
            if not abs_path.exists():
                continue

            # Delete existing call-graph chunks for this file
            if not dry_run:
                try:
                    result = self._store._collection.get(
                        where={
                            "$and": [
                                {"file_path": {"$eq": rel_path}},
                                {"chunk_type": {"$eq": "call_graph"}},
                            ]
                        },
                        include=[],
                    )
                    ids = result.get("ids", [])
                    if ids:
                        self._store._collection.delete(ids=ids)
                        logger.debug("Deleted %d call_graph chunks for %s", len(ids), rel_path)
                except Exception:
                    pass

            entries = _extract_call_graph(abs_path)
            if not entries:
                continue

            chunks: dict[str, tuple[str, dict]] = {}
            for fn_name, calls, line_count in entries:
                cid       = _callgraph_id(rel_path, fn_name)
                calls_str = ", ".join(calls) if calls else "(none)"
                text = (
                    f"Function {fn_name} in {module_name} calls: "
                    f"{calls_str}. "
                    f"Defined in {rel_path}."
                )
                chunks[cid] = (text, {
                    "source":        "call_graph",
                    "chunk_type":    "call_graph",
                    "function_name": fn_name,
                    "module":        module_name,
                    "calls":         calls_str,
                    "file_path":     rel_path,
                })

            if dry_run:
                print(
                    f"    [DRY RUN] call_graph: {rel_path} "
                    f"— would add {len(chunks)} chunks"
                )
                continue

            ids   = list(chunks.keys())
            texts = [chunks[i][0] for i in ids]
            metas = [chunks[i][1] for i in ids]
            total_added += self._store.upsert(texts=texts, metadatas=metas, ids=ids)

        return total_added
