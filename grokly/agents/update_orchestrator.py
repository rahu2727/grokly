"""
grokly/agents/update_orchestrator.py — Master agentic loop for selective updates.

Orchestrates the full Monitor → Analyse → Decide → Act → Validate pipeline:
  1. Monitor   — detect changed Python files since last checkpoint
  2. Analyse   — count affected functions, estimate API cost
  3. Decide    — prompt for approval (unless --auto-approve)
  4. Act       — delete stale chunks, regenerate commentary + call graph
  5. Validate  — confirm new chunk counts and save commit checkpoint

Public API
----------
    orch = UpdateOrchestrator()
    orch.run(dry_run=False, auto_approve=False)
"""

from __future__ import annotations

import logging
from pathlib import Path

from grokly.agents.change_analyser import ChangeAnalyserAgent
from grokly.agents.change_monitor import ChangeMonitorAgent
from grokly.agents.selective_updater import SelectiveUpdaterAgent
from grokly.config_loader import ConfigLoader
from grokly.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class UpdateOrchestrator:
    """
    Ties together ChangeMonitorAgent, ChangeAnalyserAgent, and
    SelectiveUpdaterAgent into an end-to-end agentic loop.
    """

    def __init__(
        self,
        store:          ChromaStore | None  = None,
        config_loader:  ConfigLoader | None = None,
    ) -> None:
        self._store   = store         or ChromaStore()
        self._cfg     = config_loader or ConfigLoader()
        self._monitor  = ChangeMonitorAgent(self._cfg)
        self._analyser = ChangeAnalyserAgent(self._store)
        self._updater  = SelectiveUpdaterAgent(self._store)

    # ------------------------------------------------------------------
    # Step 1: Monitor
    # ------------------------------------------------------------------

    def _step_monitor(self) -> dict[str, dict]:
        print("\n[Monitor] Checking repos for changes ...")
        return self._monitor.check_all_repos()

    # ------------------------------------------------------------------
    # Step 2: Analyse
    # ------------------------------------------------------------------

    def _step_analyse(
        self,
        repo_status: dict[str, dict],
    ) -> dict[str, dict]:
        """For each repo that has changes, build an update plan."""
        plans: dict[str, dict] = {}

        for repo_name, status in repo_status.items():
            if status.get("error"):
                print(f"  [SKIP] {repo_name}: {status['error']}")
                continue

            if status.get("is_first_run"):
                print(
                    f"  [INFO] {repo_name}: first run — no checkpoint yet. "
                    f"Run `python ingest.py --source commentary` to build the initial "
                    f"knowledge base, then save a checkpoint."
                )
                continue

            if not status["has_changes"]:
                print(f"  [OK] {repo_name}: up to date (HEAD={status['current_commit'][:8]})")
                continue

            print(
                f"\n  {repo_name}: {len(status['changed_files'])} changed file(s) "
                f"since {status['last_commit'][:8]}"
            )
            if status["commit_summary"]:
                for line in status["commit_summary"][:5]:
                    print(f"    {line}")
                if len(status["commit_summary"]) > 5:
                    print(f"    ... and {len(status['commit_summary']) - 5} more")

            # Map changed files to their enabled module
            module_map = self._build_module_map(repo_name)
            repo_path  = Path(status["repo_path"])

            files_by_module: dict[str, list[str]] = {}
            unmatched: list[str] = []
            for rel_path in status["changed_files"]:
                module = _match_module(rel_path, module_map)
                if module:
                    files_by_module.setdefault(module, []).append(rel_path)
                else:
                    unmatched.append(rel_path)

            if unmatched:
                print(f"  [SKIP] {len(unmatched)} file(s) not in any enabled module")

            all_files_to_update: list[dict] = []
            total_fns  = 0
            total_cost = 0.0

            for module_name, files in files_by_module.items():
                plan = self._analyser.analyse_changes(files, repo_path, module_name)
                all_files_to_update.extend(plan["files_to_update"])
                total_fns  += plan["total_functions"]
                total_cost += plan["estimated_cost"]
                if plan["skipped_files"]:
                    print(f"  [WARN] {len(plan['skipped_files'])} file(s) not found on disk")

            plans[repo_name] = {
                "status":           status,
                "files_to_update":  all_files_to_update,
                "total_functions":  total_fns,
                "estimated_cost":   total_cost,
                "module_map":       module_map,
            }

        return plans

    # ------------------------------------------------------------------
    # Step 3: Decide
    # ------------------------------------------------------------------

    def _step_decide(
        self,
        plans: dict[str, dict],
        auto_approve: bool,
        dry_run:      bool,
    ) -> dict[str, dict]:
        """Return only the repos approved for update."""
        approved: dict[str, dict] = {}

        for repo_name, plan in plans.items():
            fn_count = plan["total_functions"]
            cost     = plan["estimated_cost"]
            n_files  = len(plan["files_to_update"])

            print(
                f"\n[Decide] {repo_name}: {n_files} file(s), "
                f"{fn_count} function(s), estimated cost ${cost:.3f}"
            )

            if dry_run:
                print("  --dry-run: skipping approval, will simulate update")
                approved[repo_name] = plan
                continue

            if auto_approve:
                print("  --auto-approve: proceeding without prompt")
                approved[repo_name] = plan
                continue

            try:
                answer = input(f"  Update {repo_name}? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer == "y":
                approved[repo_name] = plan
            else:
                print(f"  Skipping {repo_name}.")

        return approved

    # ------------------------------------------------------------------
    # Step 4: Act
    # ------------------------------------------------------------------

    def _step_act(
        self,
        approved: dict[str, dict],
        dry_run:  bool,
    ) -> dict[str, int]:
        """Run selective update for each approved repo. Returns chunks_added per repo."""
        totals: dict[str, int] = {}

        for repo_name, plan in approved.items():
            print(f"\n[Act] Updating {repo_name} ...")
            repo_path  = Path(plan["status"]["repo_path"])
            added      = 0
            all_rel    = [f["rel_path"] for f in plan["files_to_update"]]

            for file_info in plan["files_to_update"]:
                added += self._updater.update_file(file_info, dry_run=dry_run)

            # Rebuild call graph for all changed files
            cg_added = self._updater.update_call_graph(
                all_rel,
                repo_path,
                module_name=repo_name,
                dry_run=dry_run,
            )
            if not dry_run and cg_added:
                print(f"  Call graph: {cg_added} chunks updated")
                added += cg_added

            totals[repo_name] = added
            print(f"  [Act] {repo_name}: {added} total chunks {'(dry run)' if dry_run else 'added'}")

        return totals

    # ------------------------------------------------------------------
    # Step 5: Validate
    # ------------------------------------------------------------------

    def _step_validate(
        self,
        approved: dict[str, dict],
        totals:   dict[str, int],
        dry_run:  bool,
    ) -> None:
        """Verify chunk counts and save commit checkpoints."""
        print("\n[Validate]")

        for repo_name, plan in approved.items():
            added = totals.get(repo_name, 0)

            if dry_run:
                print(f"  {repo_name}: dry run complete — no checkpoint saved")
                continue

            if added == 0:
                print(
                    f"  [WARN] {repo_name}: 0 chunks added — "
                    f"checkpoint NOT saved (re-run to retry)"
                )
                continue

            total = self._store.count()
            print(f"  {repo_name}: {added} chunks added. Collection total: {total:,}")

            commit = plan["status"]["current_commit"]
            self._monitor.save_last_checked_commit(repo_name, commit)
            print(f"  Checkpoint saved: {commit[:8]}")

    # ------------------------------------------------------------------
    # Module map helper
    # ------------------------------------------------------------------

    def _build_module_map(self, repo_name: str) -> dict[str, str]:
        """Return {module_path_prefix -> module_name} for enabled modules in repo."""
        mapping: dict[str, str] = {}
        for repo in self._cfg.get_enabled_repos():
            if repo["name"] != repo_name:
                continue
            for module in self._cfg.get_enabled_modules(repo_name):
                candidates = module.get("path_candidates", [module.get("path", "")])
                for c in candidates:
                    mapping[c.replace("\\", "/")] = module["name"]
        return mapping

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False, auto_approve: bool = False) -> None:
        """
        Execute the full Monitor → Analyse → Decide → Act → Validate loop.

        Parameters
        ----------
        dry_run : bool
            Simulate the update without calling the Claude API or modifying ChromaDB.
        auto_approve : bool
            Skip the interactive y/N prompt and approve all detected changes.
        """
        print("=" * 55)
        print("GroklyAI Change Monitor")
        print("=" * 55)
        if dry_run:
            print("DRY RUN — no API calls, no DB changes.\n")

        # 1. Monitor
        repo_status = self._step_monitor()

        any_changes = any(
            s["has_changes"] for s in repo_status.values()
            if not s.get("error") and not s.get("is_first_run")
        )

        first_runs = [
            name for name, s in repo_status.items()
            if s.get("is_first_run")
        ]

        if first_runs and not any_changes:
            print(
                "\nNo checkpoint found — this appears to be your first monitor run."
            )
            self._offer_checkpoint_save(repo_status, auto_approve, dry_run)
            return

        if not any_changes and not any(s.get("error") for s in repo_status.values()):
            print("\nAll repos are up to date. Nothing to do.")
            return

        # 2. Analyse
        plans = self._step_analyse(repo_status)
        if not plans:
            print("\nNo actionable changes detected.")
            return

        # 3. Decide
        approved = self._step_decide(plans, auto_approve=auto_approve, dry_run=dry_run)
        if not approved:
            print("\nNo repos approved for update. Exiting.")
            return

        # 4. Act
        totals = self._step_act(approved, dry_run=dry_run)

        # 5. Validate
        self._step_validate(approved, totals, dry_run=dry_run)

        print("\nDone.")

    def _offer_checkpoint_save(
        self,
        repo_status: dict[str, dict],
        auto_approve: bool,
        dry_run: bool,
    ) -> None:
        """On first run, offer to save current HEAD as the baseline checkpoint."""
        for repo_name, status in repo_status.items():
            if not status.get("is_first_run"):
                continue
            commit = status.get("current_commit")
            if not commit:
                continue

            print(
                f"\n  {repo_name}: HEAD is {commit[:8]}. "
                f"Save as baseline checkpoint? "
                f"(Future runs will detect changes from this point.)"
            )

            if dry_run:
                print("  [DRY RUN] Would save checkpoint.")
                continue

            if auto_approve:
                self._monitor.save_last_checked_commit(repo_name, commit)
                print(f"  Checkpoint saved: {commit[:8]}")
                continue

            try:
                answer = input("  Save? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer == "y":
                self._monitor.save_last_checked_commit(repo_name, commit)
                print(f"  Checkpoint saved: {commit[:8]}")
            else:
                print(f"  Skipped checkpoint for {repo_name}.")


# ---------------------------------------------------------------------------
# Module matching helper
# ---------------------------------------------------------------------------


def _match_module(rel_path: str, module_map: dict[str, str]) -> str | None:
    """Return the module name whose path prefix matches rel_path, or None."""
    rel = rel_path.replace("\\", "/")
    # Longest prefix wins (more specific module)
    best: tuple[int, str] | None = None
    for prefix, module_name in module_map.items():
        if rel.startswith(prefix.rstrip("/") + "/") or rel.startswith(prefix):
            length = len(prefix)
            if best is None or length > best[0]:
                best = (length, module_name)
    return best[1] if best else None
