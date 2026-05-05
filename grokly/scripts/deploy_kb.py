"""
Deploy a promoted knowledge base artefact.

Usage:
    python grokly/scripts/deploy_kb.py \\
        --artefact releases/chroma_db_v1.3_2026-05-10.tar.gz \\
        --target ./chroma_db_promoted \\
        --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def deploy_kb(
    artefact_path: str,
    target_path:   str,
    verify:        bool = True,
) -> bool:
    artefact = Path(artefact_path)
    target   = Path(target_path)

    print(f"GroklyAI Knowledge Base Deployment")
    print(f"{'='*50}")
    print(f"Artefact: {artefact}")
    print(f"Target:   {target}")

    if not artefact.exists():
        print(f"[ERROR] Artefact not found: {artefact}")
        return False

    # Verify checksum
    if verify:
        checksum_file = Path(f"{artefact_path}.sha256")
        if checksum_file.exists():
            with open(checksum_file) as fh:
                expected = fh.read().split()[0]

            sha256 = hashlib.sha256()
            with open(artefact, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    sha256.update(chunk)
            actual = sha256.hexdigest()

            if actual == expected:
                print(f"[OK] Checksum verified")
            else:
                print(f"[FAIL] Checksum mismatch!")
                print(f"   Expected: {expected}")
                print(f"   Actual:   {actual}")
                return False
        else:
            print(f"[WARN] No checksum file found — skipping verification")

    # Backup existing if present
    if target.exists():
        backup = Path(f"{target_path}_backup")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(target, backup)
        print(f"[OK] Existing KB backed up to {backup}")
        shutil.rmtree(target)

    # Extract artefact
    print(f"\nDeploying...")
    with tarfile.open(artefact, "r:gz") as tar:
        tar.extractall(path=target.parent, filter="data")

    # Rename extracted folder to target name
    extracted = target.parent / "chroma_db"
    if extracted.exists() and extracted != target:
        extracted.rename(target)

    # Read and display manifest
    manifest_path = target / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        print(f"\nDeployment complete:")
        print(f"  Version:  {manifest['version']}")
        print(f"  Label:    {manifest['label']}")
        print(f"  Promoted: {manifest['promoted_date']}")
        print(f"  Chunks:   {manifest['chunk_count']:,}")
    else:
        print(f"\nDeployment complete.")

    print(f"\n[OK] Knowledge base ready at {target}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy a promoted ChromaDB knowledge base artefact."
    )
    parser.add_argument("--artefact", required=True, help="Path to .tar.gz artefact")
    parser.add_argument("--target",   default="chroma_db_promoted", help="Deployment target path")
    parser.add_argument("--verify",   action="store_true", default=True,
                        help="Verify SHA-256 checksum before deploying (default: on)")
    args = parser.parse_args()

    success = deploy_kb(args.artefact, args.target, args.verify)
    raise SystemExit(0 if success else 1)
