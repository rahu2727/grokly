"""
grokly/ingestion/call_graph_ingester.py

Builds a function call graph from Python source files using AST only.
No API calls are made — no cost is incurred.

For every qualifying function in each enabled module the ingester records
which other functions it calls, producing a searchable relationship map
inside ChromaDB. This allows queries like "what calls validate_expense?"
or "what does process_payroll depend on?".

Repo and module paths come from sources_code.json via ConfigLoader.

Public API
----------
    from grokly.ingestion.call_graph_ingester import run
    chunks_added = run(store, config_loader)
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

from grokly.store.chroma_store import ChromaStore


# ---------------------------------------------------------------------------
# AST call extraction
# ---------------------------------------------------------------------------


def _extract_call_graph(
    file_path: Path,
    min_lines: int = 5,
) -> list[tuple[str, list[str], int]]:
    """
    Parse one .py file; return (fn_name, called_functions, line_count) for
    every qualifying FunctionDef.

    called_functions is the de-duplicated, sorted list of names the function
    calls. Only Name calls (plain functions) and Attribute calls (methods)
    are captured — nested/dynamic calls are skipped.

    Excluded: under *min_lines* lines, dunder methods, test_ functions.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
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
# Chunk ID helper
# ---------------------------------------------------------------------------


def _chunk_id(file_path: str, function_name: str) -> str:
    return hashlib.md5(
        f"callgraph-{file_path}-{function_name}".encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Module path resolution
# ---------------------------------------------------------------------------


def _locate_module(repo_dir: Path, candidates: list[str], label: str) -> Path | None:
    for candidate in candidates:
        full_path = repo_dir / candidate
        if os.path.isdir(full_path):
            print(f"  [OK] Found {label} module at: {full_path}")
            return full_path
    print(f"  [WARN] Could not find {label} module in {repo_dir.name} — skipping")
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(store: ChromaStore, config_loader) -> int:
    """
    Build and ingest a function call graph for all enabled modules.

    Uses AST parsing only — no API calls, no cost.

    Parameters
    ----------
    store : ChromaStore
        Destination knowledge base.
    config_loader : ConfigLoader
        Loaded configuration; supplies repos and module paths.

    Returns
    -------
    int
        Total call-graph chunks added.
    """
    project_root = Path(__file__).parent.parent.parent

    chunking  = config_loader.get_chunking_settings()
    min_lines = chunking.get("min_function_lines", 5)

    total_chunks_added = 0
    total_files        = 0
    total_functions    = 0

    for repo in config_loader.get_enabled_repos():
        label     = repo["name"]
        clone_dir = (project_root / repo["local_path"]).resolve()

        if not (clone_dir.exists() and any(clone_dir.iterdir())):
            print(f"  [SKIP] Repo not found: {clone_dir.name} — run code ingester first")
            continue

        for module in config_loader.get_enabled_modules(label):
            candidates = module.get("path_candidates", [module.get("path", "")])
            module_dir = _locate_module(clone_dir, candidates, module["name"])
            if module_dir is None:
                continue

            module_chunks    = 0
            module_functions = 0
            module_files     = 0

            for py_file in module_dir.rglob("*.py"):
                rel_path = str(py_file.relative_to(clone_dir))
                entries  = _extract_call_graph(py_file, min_lines=min_lines)

                if not entries:
                    continue

                module_files += 1

                # Use a dict keyed by chunk_id to deduplicate within this file.
                # ast.walk finds nested FunctionDefs (e.g. two classes that both
                # define a method called 'validate'), producing the same ID twice
                # in a single batch — ChromaDB rejects that even with upsert.
                chunks: dict[str, tuple[str, dict]] = {}

                for fn_name, calls, line_count in entries:
                    cid       = _chunk_id(rel_path, fn_name)
                    calls_str = ", ".join(calls) if calls else "(none)"
                    text = (
                        f"Function {fn_name} in {module['name']} calls: "
                        f"{calls_str}. "
                        f"Defined in {rel_path}."
                    )
                    chunks[cid] = (text, {
                        "source":        "call_graph",
                        "chunk_type":    "call_graph",
                        "function_name": fn_name,
                        "module":        module["name"],
                        "calls":         calls_str,
                        "file_path":     rel_path,
                    })
                    module_functions += 1

                ids   = list(chunks.keys())
                texts = [chunks[i][0] for i in ids]
                metas = [chunks[i][1] for i in ids]

                added = store.upsert(texts=texts, metadatas=metas, ids=ids)
                module_chunks += added

            print(
                f"  {module['name']}: {module_files} files, "
                f"{module_functions} functions, {module_chunks} chunks added"
            )
            total_chunks_added += module_chunks
            total_files        += module_files
            total_functions    += module_functions

    print(
        f"\n  Call graph complete: {total_files} files scanned, "
        f"{total_functions} functions mapped, "
        f"{total_chunks_added} chunks added"
    )
    print("  No API calls made — no cost incurred.")
    return total_chunks_added
