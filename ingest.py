"""
ingest.py — CLI entry point for building the Grokly knowledge base.

Usage
-----
    python ingest.py                       # run forum + docs + code
    python ingest.py --source forum        # forum Q&A only
    python ingest.py --source docs         # crawl documentation
    python ingest.py --source code         # raw code (no commentary)
    python ingest.py --source commentary   # AI function commentary
    python ingest.py --source call_graph   # AST call graph (no API cost)
    python ingest.py --reset               # wipe DB, then run all
    python ingest.py --stats               # show stats and exit
    python ingest.py --view-source         # show source config and exit
    python ingest.py --source commentary --dry-run
    python ingest.py --source commentary --max-functions 20
    python ingest.py --source monitor                    # detect + update changes
    python ingest.py --source monitor --dry-run          # simulate, no API calls
    python ingest.py --source monitor --auto-approve     # skip y/N prompt
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import time

from grokly.brand import APP_NAME, APP_VERSION
from grokly.config_loader import ConfigLoader
from grokly.model_config import print_model_summary
from grokly.store.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCE_REGISTRY: dict[str, str] = {
    "forum":       "grokly.ingestion.forum_ingester",
    "docs":        "grokly.ingestion.docs_ingester",
    "code":        "grokly.ingestion.code_ingester",
    "commentary":  "grokly.ingestion.commentary_ingester",
    "call_graph":  "grokly.ingestion.call_graph_ingester",
}

# Sources handled specially (not via _SOURCE_REGISTRY)
_SPECIAL_SOURCES = {"monitor"}

# Default run excludes commentary — it calls the Claude API and incurs cost.
_SOURCES_ALL = ["forum", "docs", "code"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> ChromaStore:
    return ChromaStore()


def _ingest_source(
    source: str,
    store: ChromaStore,
    config_loader: ConfigLoader,
    dry_run: bool = False,
    max_functions: int = 999,
) -> int:
    """
    Import and run one ingester. Returns chunks added (0 on skip/error).
    config_loader, dry_run, and max_functions are forwarded to ingesters
    that declare those parameters.
    """
    module_path = _SOURCE_REGISTRY.get(source)
    if module_path is None:
        print(f"  [SKIP] Unknown source '{source}'.")
        return 0

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        print(f"  [SKIP] '{source}' ingester not built yet ({module_path}.py).")
        return 0
    except ImportError as exc:
        print(f"  [SKIP] '{source}' ingester failed to import: {exc}")
        return 0

    if not hasattr(mod, "run"):
        print(f"  [SKIP] '{module_path}' has no run() function.")
        return 0

    t0 = time.perf_counter()
    try:
        sig    = inspect.signature(mod.run)
        params = sig.parameters

        extra: dict = {}
        if "dry_run" in params:
            extra["dry_run"] = dry_run
        if "max_functions" in params:
            extra["max_functions"] = max_functions

        if len(params) >= 2:
            added = mod.run(store, config_loader, **extra)
        else:
            added = mod.run(store, **extra)

    except Exception as exc:
        print(f"  [ERROR] '{source}' ingester raised: {exc}")
        return 0

    elapsed = time.perf_counter() - t0
    print(f"  [{source}] {added} chunks added in {elapsed:.1f}s")
    return added


def _print_stats(store: ChromaStore) -> None:
    stats = store.stats()
    print("\nCollection statistics")
    print(f"  Total chunks : {stats['total']}")
    if stats["by_source"]:
        print("  By source    :")
        for src, count in sorted(stats["by_source"].items()):
            print(f"    {src:<15s}  {count:4d}")
    else:
        print("  (collection is empty)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{APP_NAME} knowledge-base ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python ingest.py                          # run forum + docs + code
  python ingest.py --source commentary      # AI commentary (costs ~$7)
  python ingest.py --source commentary --dry-run
  python ingest.py --source commentary --max-functions 20
  python ingest.py --source call_graph      # AST call graph (no API cost)
  python ingest.py --reset                  # wipe then run all
  python ingest.py --stats                  # show stats and exit
  python ingest.py --view-source            # show source config and exit
""",
    )
    p.add_argument(
        "--source",
        choices=list(_SOURCE_REGISTRY.keys()) + list(_SPECIAL_SOURCES),
        default=None,
        help="Data source to ingest. Omit to run forum + docs + code.",
    )
    p.add_argument(
        "--auto-approve",
        action="store_true",
        dest="auto_approve",
        help="(monitor) Skip y/N confirmation prompt and approve all changes.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the collection before ingesting.",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics (after any ingest/reset) and exit.",
    )
    p.add_argument(
        "--view-source",
        action="store_true",
        dest="view_source",
        help="Print source configuration summary and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="(commentary) Scan and report without calling the API.",
    )
    p.add_argument(
        "--max-functions",
        type=int,
        default=999,
        dest="max_functions",
        metavar="N",
        help="(commentary) Process at most N functions (default: 999).",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    print(f"{APP_NAME} v{APP_VERSION} — Ingestion Pipeline")
    print("=" * 45)
    print_model_summary()

    config_loader = ConfigLoader()

    if args.view_source:
        print(config_loader.summary())
        return

    # Monitor — handled separately from normal ingestion sources
    if args.source == "monitor":
        from grokly.agents.update_orchestrator import UpdateOrchestrator
        orch = UpdateOrchestrator()
        orch.run(dry_run=args.dry_run, auto_approve=args.auto_approve)
        return

    if args.source == "code":
        print("[WARN] code ingester is superseded by commentary.")
        print("       Run: python ingest.py --source commentary")
        print("       Continue anyway? (y/n): ", end="")
        response = input()
        if response.lower() != "y":
            sys.exit(0)

    # Stats-only shortcut
    if args.stats and not args.source and not args.reset:
        print(config_loader.summary())
        print()
        store = _make_store()
        _print_stats(store)
        return

    print(config_loader.summary())
    print()

    store = _make_store()

    if args.reset:
        print("Resetting collection...")
        store.reset()
        print(f"  Done. Chunks remaining: {store.count()}")

    sources_to_run = [args.source] if args.source else _SOURCES_ALL

    total_added = 0
    for source in sources_to_run:
        print(f"\nIngesting source: {source}")
        total_added += _ingest_source(
            source,
            store,
            config_loader,
            dry_run=args.dry_run,
            max_functions=args.max_functions,
        )

    print(f"\nIngestion complete. Total chunks added this run: {total_added}")
    print(f"Collection total: {store.count()} chunks")

    if args.stats:
        _print_stats(store)


if __name__ == "__main__":
    main()
