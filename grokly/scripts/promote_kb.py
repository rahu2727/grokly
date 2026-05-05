"""
Knowledge Base Promotion Script

Packages the current ChromaDB as a versioned
artefact ready for deployment to test or
production environments.

Usage:
    python grokly/scripts/promote_kb.py \\
        --version 1.3 \\
        --label "Sprint 4 — HR and Payroll"

Output:
    releases/chroma_db_v1.3_2026-05-10.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import date, datetime
from pathlib import Path


def promote_kb(
    version:     str,
    label:       str,
    source_path: str = "chroma_db",
    output_dir:  str = "releases",
) -> str:
    today    = date.today().isoformat()
    filename = f"chroma_db_v{version}_{today}.tar.gz"
    output_path = Path(output_dir) / filename

    Path(output_dir).mkdir(exist_ok=True)

    print(f"GroklyAI Knowledge Base Promotion")
    print(f"{'='*50}")
    print(f"Version:  {version}")
    print(f"Label:    {label}")
    print(f"Source:   {source_path}")
    print(f"Output:   {output_path}")
    print()

    if not Path(source_path).exists():
        print(f"[ERROR] Source path not found: {source_path}")
        raise SystemExit(1)

    # Get chunk count before packaging
    total_chunks = 0
    try:
        import chromadb
        client       = chromadb.PersistentClient(path=source_path)
        collections  = client.list_collections()
        total_chunks = sum(c.count() for c in collections)
        print(f"Chunks:   {total_chunks:,}")
    except Exception as exc:
        print(f"Could not count chunks: {exc}")

    # Write manifest into source before packaging
    manifest = {
        "version":                   version,
        "label":                     label,
        "promoted_date":             today,
        "promoted_at":               datetime.now().isoformat(),
        "chunk_count":               total_chunks,
        "source_path":               source_path,
        "filename":                  filename,
        "compatible_grokly_version": "0.1.0",
    }
    manifest_path = Path(source_path) / "MANIFEST.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nPackaging knowledge base...")

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_path, arcname="chroma_db")

    # Calculate checksum
    sha256 = hashlib.sha256()
    with open(output_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()

    checksum_path = Path(output_dir) / f"{filename}.sha256"
    with open(checksum_path, "w") as fh:
        fh.write(f"{checksum}  {filename}\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\nPromotion complete:")
    print(f"  File:     {output_path}")
    print(f"  Size:     {size_mb:.1f} MB")
    print(f"  Checksum: {checksum[:16]}...")
    print(f"  Chunks:   {total_chunks:,}")
    print()
    print(f"To deploy to test or production:")
    print(f"  python grokly/scripts/deploy_kb.py \\")
    print(f"    --artefact {output_path} \\")
    print(f"    --target ./chroma_db_promoted")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package the ChromaDB knowledge base as a versioned artefact."
    )
    parser.add_argument("--version", required=True, help="Version string, e.g. 1.3")
    parser.add_argument("--label",   required=True, help="Human-readable release label")
    parser.add_argument("--source",  default="chroma_db",  dest="source", help="Source ChromaDB path")
    parser.add_argument("--output",  default="releases",   dest="output", help="Output directory")
    args = parser.parse_args()

    promote_kb(args.version, args.label, args.source, args.output)
