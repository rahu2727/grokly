"""
grokly/scripts/tag_existing_chunks.py — One-time migration: tag existing chunks with application key.

Reads all chunks from ChromaDB that lack an `application` metadata field and adds
`"application": "<default_app>"` to each. Safe to run multiple times (idempotent).

Usage:
    python -m grokly.scripts.tag_existing_chunks
    python -m grokly.scripts.tag_existing_chunks --dry-run
    python -m grokly.scripts.tag_existing_chunks --application erpnext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from grokly.store.chroma_store import ChromaStore  # noqa: E402


_BATCH_SIZE = 500


def tag_chunks(application: str, dry_run: bool = False) -> int:
    """
    Add ``application`` metadata field to every chunk that lacks it.

    Returns the number of chunks updated.
    """
    store = ChromaStore()
    collection = store._collection

    offset = 0
    total_updated = 0

    print(f"Tagging existing chunks with application='{application}'")
    print(f"{'[DRY RUN] ' if dry_run else ''}Scanning ChromaDB collection …\n")

    while True:
        result = collection.get(
            limit=_BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        ids: list[str] = result.get("ids", [])
        metadatas: list[dict] = result.get("metadatas", [])

        if not ids:
            break

        to_update_ids: list[str] = []
        to_update_metas: list[dict] = []

        for chunk_id, meta in zip(ids, metadatas):
            if meta is None:
                meta = {}
            if "application" not in meta:
                updated_meta = {**meta, "application": application}
                to_update_ids.append(chunk_id)
                to_update_metas.append(updated_meta)

        if to_update_ids:
            if not dry_run:
                collection.update(ids=to_update_ids, metadatas=to_update_metas)
            total_updated += len(to_update_ids)
            print(
                f"  offset={offset}: "
                f"{'would update' if dry_run else 'updated'} "
                f"{len(to_update_ids)}/{len(ids)} chunks"
            )
        else:
            print(f"  offset={offset}: {len(ids)} chunks already tagged — skipping")

        offset += len(ids)
        if len(ids) < _BATCH_SIZE:
            break

    print(
        f"\n{'[DRY RUN] Would tag' if dry_run else 'Tagged'} "
        f"{total_updated} chunks with application='{application}'"
    )
    return total_updated


def _default_application() -> str:
    config_path = Path(__file__).parent.parent / "config" / "applications.json"
    try:
        with config_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("routing", {}).get("default_application", "erpnext")
    except Exception:
        return "erpnext"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag existing ChromaDB chunks with an application key."
    )
    parser.add_argument(
        "--application",
        default=_default_application(),
        help="Application key to tag chunks with (default: from applications.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without modifying ChromaDB",
    )
    args = parser.parse_args()

    updated = tag_chunks(application=args.application, dry_run=args.dry_run)
    sys.exit(0 if updated >= 0 else 1)


if __name__ == "__main__":
    main()
