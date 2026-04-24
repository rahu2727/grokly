"""
query_test.py — Quick smoke-test for the Grokly knowledge base.

Usage
-----
    python query_test.py
    python query_test.py --query "how do I submit an expense claim?"
    python query_test.py --n 10
    python query_test.py --source forum
"""

from __future__ import annotations

import argparse

from grokly.brand import APP_NAME, APP_VERSION
from grokly.store.chroma_store import ChromaStore

_DEFAULT_QUERIES = [
    "How do I submit a leave application?",
    "What is the difference between a Purchase Order and a Material Request?",
    "How is overtime calculated in payroll?",
    "How do I transfer stock between warehouses?",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{APP_NAME} query smoke-test")
    p.add_argument("--query", "-q", default=None, help="Custom query string.")
    p.add_argument("--n",          type=int, default=3,    help="Number of results (default: 3).")
    p.add_argument("--source",     default=None,           help="Filter by source (forum, docs, code_commentary, call_graph).")
    p.add_argument("--stats",      action="store_true",    help="Show collection stats and exit.")
    return p


def main() -> None:
    args  = _build_parser().parse_args()
    store = ChromaStore()

    print(f"{APP_NAME} v{APP_VERSION} — Query Test")
    print("=" * 45)

    stats = store.stats()
    print(f"Collection: {stats['total']} total chunks")
    for src, cnt in sorted(stats.get("by_source", {}).items()):
        print(f"  {src:<18s} {cnt:4d}")
    print()

    if args.stats:
        return

    queries = [args.query] if args.query else _DEFAULT_QUERIES
    where   = {"source": {"$eq": args.source}} if args.source else None

    for query in queries:
        print(f"Query: {query}")
        results = store.query(query, n_results=args.n, where=where)
        if not results:
            print("  (no results)\n")
            continue
        for i, r in enumerate(results, 1):
            meta  = r["metadata"]
            score = max(0, round((1 - r["distance"]) * 100, 1))
            src   = meta.get("source", "?")
            cat   = meta.get("category", meta.get("module", meta.get("function_name", "")))
            print(f"  [{i}] {score:5.1f}%  [{src}] {cat}")
            print(f"       {r['text'][:120].strip()}...")
        print()


if __name__ == "__main__":
    main()
