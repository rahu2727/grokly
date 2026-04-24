"""
grokly/ingestion/code_ingester.py

Clones configured GitHub repositories and ingests Python source files
and DocType JSON definitions into the ChromaStore knowledge base.

Repositories, module paths, and clone settings are all read from
sources_code.json via ConfigLoader — nothing is hardcoded here.

NOTE: This ingester is superseded by commentary_ingester.py which stores
both raw_code and commentary chunks in a single pass. Only use this ingester
if you need raw code chunks WITHOUT generating commentary — for example:
  - Testing ChromaDB setup before spending API credits
  - Ingesting a large codebase cheaply first, then running commentary selectively
  - Air-gapped environments with no API access

For normal Grokly use: run commentary ingester only.

Public API
----------
    from grokly.ingestion.code_ingester import run
    chunks_added = run(store, config_loader)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from grokly.store.chroma_store import ChromaStore

_MIN_CHUNK_CHARS = 100
_MAX_CHUNKS_FILE = 200


# ---------------------------------------------------------------------------
# Git clone helpers
# ---------------------------------------------------------------------------


def _git_available() -> bool:
    return shutil.which("git") is not None


def _clone_repo(clone_dir: Path, url: str, label: str) -> bool:
    """Shallow-clone *url* into *clone_dir*. Returns True on success."""
    if not _git_available():
        print(
            "  [ERROR] 'git' command not found.\n"
            "          Install Git from https://git-scm.com/download/win\n"
            "          then re-run: python ingest.py --source code"
        )
        return False

    print(f"  Cloning {label} (shallow) into {clone_dir} ...")
    print("  This may take 2-5 minutes depending on your connection.")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--single-branch",
                "--config", "core.protectNTFS=false",
                url,
                str(clone_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(
                f"  [ERROR] git clone failed (exit {result.returncode}).\n"
                f"          {result.stderr.strip()}"
            )
            return False
        print(f"  Clone complete: {label}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] git clone timed out ({label}).")
        return False
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return False


# ---------------------------------------------------------------------------
# Module path resolution
# ---------------------------------------------------------------------------


def _locate_module(repo_dir: Path, candidates: list[str], label: str) -> Path | None:
    """
    Try each candidate path inside *repo_dir* using os.path.isdir().
    Prints [OK] on match or [WARN] if nothing found.
    """
    for candidate in candidates:
        full_path = repo_dir / candidate
        if os.path.isdir(full_path):
            print(f"  [OK] Found {label} module at: {full_path}")
            return full_path
    print(f"  [WARN] Could not find {label} module in {repo_dir.name} — skipping")
    return None


# ---------------------------------------------------------------------------
# Python file chunking
# ---------------------------------------------------------------------------


def _split_by_definitions(content: str) -> list[str]:
    """
    Split Python source *content* into chunks at ``def`` / ``class`` boundaries.

    Any preamble before the first definition is kept as chunk 0 if long enough.
    Each chunk must be >= _MIN_CHUNK_CHARS characters.
    Returns at most _MAX_CHUNKS_FILE chunks.
    """
    pattern   = re.compile(r"^(def |class )", re.MULTILINE)
    positions = [m.start() for m in pattern.finditer(content)]

    if not positions:
        chunk = content.strip()
        return [chunk] if len(chunk) >= _MIN_CHUNK_CHARS else []

    chunks: list[str] = []

    preamble = content[: positions[0]].strip()
    if len(preamble) >= _MIN_CHUNK_CHARS:
        chunks.append(preamble)

    boundaries = positions + [len(content)]
    for i in range(len(positions)):
        chunk = content[positions[i] : boundaries[i + 1]].strip()
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)

    return chunks[:_MAX_CHUNKS_FILE]


# ---------------------------------------------------------------------------
# DocType JSON → readable text
# ---------------------------------------------------------------------------


def _json_to_text(data: dict) -> str:
    """Convert an ERPNext DocType JSON schema to a human-readable text summary."""
    lines: list[str] = []

    name = data.get("name", "Unknown DocType")
    lines.append(f"DocType: {name}")

    module = data.get("module", "")
    if module:
        lines.append(f"Module: {module}")

    description = data.get("description", "")
    if description:
        lines.append(f"Description: {description}")

    if data.get("is_submittable", 0):
        lines.append("Submittable: Yes (has Submit/Cancel/Amend workflow)")

    _SKIP_TYPES = {"Column Break", "Section Break", "HTML", "Fold", "Heading"}
    fields = [
        f for f in data.get("fields", [])
        if f.get("fieldtype") not in _SKIP_TYPES and f.get("label")
    ]
    if fields:
        field_parts = [
            f"{f['label']} ({f.get('fieldtype', '?')})"
            for f in fields[:30]
        ]
        lines.append(f"Fields: {', '.join(field_parts)}")

    permissions = data.get("permissions", [])
    roles = sorted({p.get("role", "") for p in permissions if p.get("role")})
    if roles:
        lines.append(f"Roles with access: {', '.join(roles[:15])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def _chunk_id(file_path: str, index: int) -> str:
    """Deterministic MD5 ID from relative file path + chunk index."""
    return hashlib.md5(f"{file_path}::{index}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module walker
# ---------------------------------------------------------------------------


def _process_module_dir(
    module_dir: Path,
    module_name: str,
    repo_root: Path,
    store: ChromaStore,
) -> int:
    """
    Walk all .py and doctype .json files under *module_dir*.

    Returns total chunks added for this module.
    """
    py_files   = list(module_dir.rglob("*.py"))
    json_files = [
        p for p in module_dir.rglob("*.json")
        if "doctype" in p.parts
    ]

    total_added     = 0
    files_processed = 0

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if not content.strip():
            continue

        chunks = _split_by_definitions(content)
        if not chunks:
            continue

        rel_path = str(py_file.relative_to(repo_root))
        texts, metas, ids = [], [], []

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metas.append(
                {
                    "source":    "code",
                    "file_type": "python",
                    "module":    module_name,
                    "file_path": rel_path,
                }
            )
            ids.append(_chunk_id(rel_path, i))

        added = store.add(texts=texts, metadatas=metas, ids=ids)
        total_added     += added
        files_processed += 1

    for json_file in json_files:
        try:
            data = json.loads(
                json_file.read_text(encoding="utf-8", errors="replace")
            )
        except (json.JSONDecodeError, Exception):
            continue

        if not isinstance(data, dict) or data.get("doctype") != "DocType":
            continue

        text = _json_to_text(data)
        if len(text) < _MIN_CHUNK_CHARS:
            continue

        rel_path = str(json_file.relative_to(repo_root))
        added = store.add(
            texts=[text],
            metadatas=[
                {
                    "source":    "code",
                    "file_type": "json_doctype",
                    "module":    module_name,
                    "file_path": rel_path,
                }
            ],
            ids=[_chunk_id(rel_path, 0)],
        )
        total_added     += added
        files_processed += 1

    print(
        f"    {module_name}: {files_processed} file(s), "
        f"{total_added} chunks added."
    )
    return total_added


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(store: ChromaStore, config_loader) -> int:
    """
    Clone (or reuse) configured repos and ingest code chunks into *store*.

    Parameters
    ----------
    store : ChromaStore
        Destination knowledge base.
    config_loader : ConfigLoader
        Loaded configuration; supplies repos, module paths, and clone settings.

    Returns
    -------
    int
        Total chunks added.
    """
    project_root = Path(__file__).parent.parent.parent
    total_added  = 0

    for repo in config_loader.get_enabled_repos():
        clone_dir = (project_root / repo["local_path"]).resolve()
        label     = repo["name"]

        if clone_dir.exists() and any(clone_dir.iterdir()):
            print(f"  Repo already present: {clone_dir.name}")
        else:
            if not _clone_repo(clone_dir, repo["clone_url"], label):
                continue

        for module in config_loader.get_enabled_modules(label):
            candidates = module.get("path_candidates", [module.get("path", "")])
            module_dir = _locate_module(clone_dir, candidates, module["name"])
            if module_dir is None:
                continue

            print(f"  Processing module: {module['name']}")
            total_added += _process_module_dir(
                module_dir, module["name"], clone_dir, store
            )

    return total_added
