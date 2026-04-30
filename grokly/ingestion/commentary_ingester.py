"""
grokly/ingestion/commentary_ingester.py

Reads Python functions from configured ERPNext/HRMS modules, generates
plain-English commentary via the Claude API, and stores both commentary
and raw code chunks in ChromaDB.

Repositories, module paths, model name, and API settings are all read
from sources_code.json via ConfigLoader. Prompts are loaded from
grokly/prompts/ via PromptLoader. Language routing is handled by
RouterAgent — nothing is hardcoded here.

Public API
----------
    from grokly.ingestion.commentary_ingester import run
    chunks_added = run(store, config_loader, dry_run=False, max_functions=9999)
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from grokly.model_config import get_agent_config
from grokly.prompt_loader import PromptLoader
from grokly.ingestion.router_agent import RouterAgent
from grokly.store.chroma_store import ChromaStore

load_dotenv()

# Fallback cost estimate (input $3/MTok, output $15/MTok).
# ~300 input tokens + ~200 output tokens per function ≈ $0.004 each.
_COST_PER_FUNCTION: float = 0.004


# ---------------------------------------------------------------------------
# Repo clone helpers
# ---------------------------------------------------------------------------


def _git_available() -> bool:
    return shutil.which("git") is not None


def _clone_repo(clone_dir: Path, url: str, label: str) -> bool:
    """Shallow-clone *url* into *clone_dir*. Returns True on success."""
    if not _git_available():
        print("  [ERROR] 'git' command not found. Install Git then re-run.")
        return False

    print(f"  Cloning {label} (shallow) into {clone_dir} ...")
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
    for candidate in candidates:
        full_path = repo_dir / candidate
        if os.path.isdir(full_path):
            print(f"  [OK] Found {label} module at: {full_path}")
            return full_path
    print(f"  [WARN] Could not find {label} module in {repo_dir.name} — skipping")
    return None


# ---------------------------------------------------------------------------
# AST function extraction
# ---------------------------------------------------------------------------


def _extract_functions(
    file_path: Path,
    min_lines: int = 5,
) -> list[tuple[str, str, int]]:
    """
    Parse one .py file; return (name, source_code, line_count) for every
    qualifying FunctionDef.

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

        if not source or not source.strip():
            continue

        results.append((name, source.strip(), line_count))

    return results


# ---------------------------------------------------------------------------
# Chunk ID helpers
# ---------------------------------------------------------------------------


def _commentary_id(file_path: str, function_name: str) -> str:
    return hashlib.md5(
        f"commentary-{file_path}-{function_name}".encode("utf-8")
    ).hexdigest()


def _raw_code_id(file_path: str, function_name: str) -> str:
    return hashlib.md5(
        f"raw_code-{file_path}-{function_name}".encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Claude API call — prompts loaded from YAML registry
# ---------------------------------------------------------------------------


def _generate_commentary(
    client: anthropic.Anthropic,
    function_name: str,
    file_path: str,
    module_label: str,
    source_code: str,
    prompt_loader: PromptLoader,
    router_agent: RouterAgent,
) -> tuple[str | None, str, str, str]:
    """
    Call Claude and return (commentary_text, prompt_name, prompt_version, expert).

    Returns (None, prompt_name, prompt_version, expert) on API failure.
    """
    language    = router_agent.detect_language(file_path, source_code)
    prompt_name = router_agent.LANGUAGE_RULES[language]["prompt"]
    expert      = router_agent.get_expert_description(language)

    print(f"    [{expert}] generating commentary for {function_name}")

    system_prompt  = prompt_loader.get_system_prompt(prompt_name)
    user_prompt    = prompt_loader.format_user_prompt(
        prompt_name,
        function_name=function_name,
        file_path=file_path,
        module_name=module_label,
        function_source_code=source_code,
    )
    prompt_version = prompt_loader.get_version(prompt_name)
    _cfg           = get_agent_config("commentary")

    try:
        response = client.messages.create(
            model=_cfg["model"],
            max_tokens=_cfg["max_tokens"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text, prompt_name, prompt_version, expert
    except Exception as exc:
        print(f"    [API ERROR] {function_name}: {exc}")
        return None, prompt_name, prompt_version, expert


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    store: ChromaStore,
    config_loader,
    dry_run: bool = False,
    max_functions: int = 9999,
) -> int:
    """
    Generate AI commentary for Python functions from configured repos.

    Parameters
    ----------
    store : ChromaStore
        Destination knowledge base.
    config_loader : ConfigLoader
        Loaded configuration; supplies repos, module paths, and agent settings.
    dry_run : bool
        If True, scan and count without calling the API or storing anything.
    max_functions : int
        Cap on functions to process. Use 20 for a test run.

    Returns
    -------
    int
        Total chunks added (0 in dry_run mode).
    """
    project_root = Path(__file__).parent.parent.parent

    prompt_loader = PromptLoader()
    router_agent  = RouterAgent()

    if not dry_run:
        print(prompt_loader.summary())
        print()

    commentary   = config_loader.get_commentary_settings()
    api_delay    = commentary.get("api_delay_seconds",          0.5)
    cost_warning = commentary.get("cost_warning_threshold_usd", 5.0)

    chunking  = config_loader.get_chunking_settings()
    min_lines = chunking.get("min_function_lines", 5)

    if dry_run:
        print("  DRY RUN — no API calls will be made, nothing stored.\n")

    print("  Checking repositories ...")
    ready_repos: set[str] = set()

    for repo in config_loader.get_enabled_repos():
        clone_dir = (project_root / repo["local_path"]).resolve()
        label     = repo["name"]

        if clone_dir.exists() and any(clone_dir.iterdir()):
            print(f"  Repo already present: {clone_dir.name}")
            ready_repos.add(label)
        elif dry_run:
            print(f"  [DRY RUN] Repo not found: {clone_dir.name}")
        else:
            if _clone_repo(clone_dir, repo["clone_url"], label):
                ready_repos.add(label)

    print()

    all_functions: list[tuple[str, str, str, str, int]] = []

    for repo in config_loader.get_enabled_repos():
        label     = repo["name"]
        clone_dir = (project_root / repo["local_path"]).resolve()

        if label not in ready_repos:
            print(f"  [SKIP] {label} — repo unavailable")
            continue

        for module in config_loader.get_enabled_modules(label):
            candidates = module.get("path_candidates", [module.get("path", "")])
            module_dir = _locate_module(clone_dir, candidates, module["name"])
            if module_dir is None:
                continue

            module_count = 0
            for py_file in module_dir.rglob("*.py"):
                rel_path = str(py_file.relative_to(clone_dir)).replace("\\", "/")
                for fn_name, fn_source, fn_lines in _extract_functions(
                    py_file, min_lines=min_lines
                ):
                    all_functions.append(
                        (fn_name, fn_source, rel_path, module["name"], fn_lines)
                    )
                    module_count += 1

            print(f"  {module['name']}: {module_count} qualifying functions found")

    total_found = len(all_functions)

    all_ids_fwd  = [_commentary_id(f[2],                    f[0]) for f in all_functions]
    all_ids_back = [_commentary_id(f[2].replace("/", "\\"), f[0]) for f in all_functions]
    all_ids_both = list(set(all_ids_fwd + all_ids_back))
    already_done = 0
    try:
        existing = store._collection.get(ids=all_ids_both, include=[])
        fwd_set  = set(all_ids_fwd)
        back_set = set(all_ids_back)
        hit_set  = set(existing["ids"])
        already_done = len((hit_set & fwd_set) | (hit_set & back_set))
    except Exception:
        already_done = 0

    remaining  = total_found - already_done
    to_process = min(remaining, max_functions)

    print(f"\n  Total qualifying functions : {total_found}")
    print(f"  Already processed         : {already_done}")
    print(f"  Remaining                 : {remaining}")
    print(f"  Will process              : {to_process}")

    if not dry_run:
        estimated = to_process * _COST_PER_FUNCTION
        print(f"  Estimated cost (approx)   : ${estimated:.2f}")
        if estimated >= cost_warning:
            print(
                f"  [COST WARNING] Estimated cost ${estimated:.2f} exceeds "
                f"threshold ${cost_warning:.2f} — proceeding."
            )

    if dry_run:
        print("\n  DRY RUN complete — no changes made.")
        return 0

    client             = anthropic.Anthropic()
    total_chunks_added = 0
    processed          = 0
    skip_count         = 0
    cost_so_far        = 0.0

    for fn_name, fn_source, rel_path, module_label, line_count in all_functions:
        if processed >= to_process:
            break

        existing_id      = _commentary_id(rel_path,                    fn_name)
        existing_id_back = _commentary_id(rel_path.replace("/", "\\"), fn_name)
        try:
            existing = store._collection.get(ids=[existing_id, existing_id_back], include=[])
            if existing and len(existing["ids"]) > 0:
                skip_count += 1
                continue
        except Exception:
            pass

        commentary_text_raw, prompt_name, prompt_version, expert = _generate_commentary(
            client, fn_name, rel_path, module_label,
            fn_source, prompt_loader, router_agent,
        )

        if commentary_text_raw is None:
            time.sleep(api_delay)
            continue

        commentary_text = (
            f"Function: {fn_name}\n"
            f"File: {rel_path}\n"
            f"Module: {module_label}\n\n"
            f"{commentary_text_raw}"
        )

        added_c = store.upsert(
            texts=[commentary_text],
            metadatas=[{
                "source":              "code_commentary",
                "file_path":           rel_path,
                "function_name":       fn_name,
                "module":              module_label,
                "chunk_type":          "commentary",
                "prompt_name":         prompt_name,
                "prompt_version":      prompt_version,
                "expert_agent":        expert,
                "original_code_lines": line_count,
            }],
            ids=[_commentary_id(rel_path, fn_name)],
        )
        total_chunks_added += added_c

        added_r = store.upsert(
            texts=[fn_source],
            metadatas=[{
                "source":              "code_commentary",
                "file_path":           rel_path,
                "function_name":       fn_name,
                "module":              module_label,
                "chunk_type":          "raw_code",
                "prompt_name":         prompt_name,
                "prompt_version":      prompt_version,
                "expert_agent":        expert,
                "original_code_lines": line_count,
            }],
            ids=[_raw_code_id(rel_path, fn_name)],
        )
        total_chunks_added += added_r

        processed   += 1
        cost_so_far  = processed * _COST_PER_FUNCTION

        if processed % 10 == 0:
            print(
                f"  Processed {processed}/{to_process} functions"
                f" — estimated cost so far: ${cost_so_far:.2f}"
            )

        time.sleep(api_delay)

    print("\n  Commentary complete:")
    print(f"    Skipped (already processed): {skip_count}")
    print(f"    New functions processed    : {processed}")
    print(f"    Chunks added this run      : {total_chunks_added}")
    print(f"    Estimated cost this run    : ${cost_so_far:.2f}")

    return total_chunks_added
