"""
test_monitor.py — Smoke tests for Sprint 4D Change Monitor.

Creates a minimal temporary git repo to demonstrate all 4 test scenarios:
  TEST 1: First run — no checkpoint (shows first-run message)
  TEST 2: Save initial checkpoint
  TEST 3: Run again — up to date
  TEST 4: Simulate a change — detects modified Python file
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# We test ChangeMonitorAgent directly (bypassing sources_code.json)
from grokly.agents.change_monitor import ChangeMonitorAgent, _STATE_FILE


def _git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True)
    return r.stdout.strip()


def _setup_test_repo(tmp: Path) -> Path:
    """Create a minimal git repo with one Python file and an initial commit."""
    repo = tmp / "test_repo"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "test@test.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    py_file = repo / "module" / "utils.py"
    py_file.parent.mkdir()
    py_file.write_text(
        "def process_invoice(doc):\n"
        "    '''Process an invoice document.'''\n"
        "    doc.status = 'Submitted'\n"
        "    doc.save()\n"
        "    return True\n\n"
        "def validate_supplier(supplier_id):\n"
        "    '''Validate a supplier exists.'''\n"
        "    if not supplier_id:\n"
        "        raise ValueError('supplier_id required')\n"
        "    return True\n",
        encoding="utf-8",
    )
    _git(["add", "."], repo)
    _git(["commit", "-m", "Initial commit"], repo)
    return repo


def _hr(title: str) -> None:
    print(f"\n{'='*55}")
    print(f" {title}")
    print("=" * 55)


# ------------------------------------------------------------------
# Clean up any leftover state from previous runs
# ------------------------------------------------------------------

if _STATE_FILE.exists():
    _STATE_FILE.unlink()
    print(f"[setup] Removed leftover state: {_STATE_FILE.name}")

monitor = ChangeMonitorAgent()

with tempfile.TemporaryDirectory() as _td:
    tmp       = Path(_td)
    test_repo = _setup_test_repo(tmp)
    repo_name = "test_erpnext"

    # ================================================================
    # TEST 1 — First run: no checkpoint saved yet
    # ================================================================
    _hr("TEST 1 — First run (no checkpoint)")
    current = monitor.get_current_commit(test_repo)
    last    = monitor.get_last_checked_commit(repo_name)

    print(f"  current commit : {current}")
    print(f"  last checkpoint: {last}")
    assert last is None, "Expected no checkpoint on first run"
    print("  Result: PASS — no checkpoint, is_first_run scenario confirmed")

    # ================================================================
    # TEST 2 — Save initial checkpoint
    # ================================================================
    _hr("TEST 2 — Save initial checkpoint")
    monitor.save_last_checked_commit(repo_name, current)
    saved = monitor.get_last_checked_commit(repo_name)

    print(f"  saved: {saved}")
    assert saved == current, "Checkpoint not saved correctly"
    state_content = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    print(f"  state file: {json.dumps(state_content, indent=2)}")
    print("  Result: PASS — checkpoint saved to monitor_state.json")

    # ================================================================
    # TEST 3 — Up to date (HEAD == checkpoint)
    # ================================================================
    _hr("TEST 3 — No changes (up to date)")
    current2 = monitor.get_current_commit(test_repo)
    last2    = monitor.get_last_checked_commit(repo_name)
    changed  = monitor.get_changed_files(test_repo, last2) if last2 != current2 else []

    print(f"  HEAD matches checkpoint: {current2 == last2}")
    print(f"  changed files: {changed}")
    assert current2 == last2
    assert changed == []
    print("  Result: PASS — HEAD == checkpoint, no changed files")

    # ================================================================
    # TEST 4 — Simulate a change and detect it
    # ================================================================
    _hr("TEST 4 — Simulate change, verify detection")

    # Modify the file and commit
    py_file = test_repo / "module" / "utils.py"
    py_file.write_text(
        py_file.read_text(encoding="utf-8") +
        "\ndef cancel_invoice(doc):\n"
        "    '''Cancel a submitted invoice.'''\n"
        "    doc.status = 'Cancelled'\n"
        "    doc.save()\n"
        "    return True\n",
        encoding="utf-8",
    )
    _git(["add", "."], test_repo)
    _git(["commit", "-m", "Add cancel_invoice function"], test_repo)

    new_commit  = monitor.get_current_commit(test_repo)
    old_commit  = monitor.get_last_checked_commit(repo_name)
    files       = monitor.get_changed_files(test_repo, old_commit)
    summary     = monitor.get_commit_summary(test_repo, old_commit)

    print(f"  old checkpoint: {old_commit[:8]}")
    print(f"  new HEAD      : {new_commit[:8]}")
    print(f"  changed files : {files}")
    print(f"  commit summary: {summary}")

    assert new_commit != old_commit
    assert len(files) == 1
    assert "utils.py" in files[0]
    assert len(summary) == 1
    print("  Result: PASS — change detected correctly")

    # ================================================================
    # Cleanup
    # ================================================================
    _hr("All tests passed")
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()
    print("  monitor_state.json cleaned up")

print()
