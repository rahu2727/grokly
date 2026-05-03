"""
grokly/identity/admin.py — Admin functions for managing the user master file.

Command-line tools for Phase 2. Phase 3 will add a web admin panel.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path

_MASTER_FILE = Path(__file__).parent.parent / "config" / "users_master.json"


class UserAdmin:
    """Write operations on the user master file."""

    def __init__(self, master_file: str | Path = _MASTER_FILE) -> None:
        self._path = Path(master_file)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            with self._path.open(encoding="utf-8") as fh:
                return json.load(fh)
        return {"metadata": {}, "users": []}

    def _save(self, data: dict) -> None:
        data.setdefault("metadata", {})["generated_date"] = str(date.today())
        data["metadata"]["last_modified"] = datetime.now().isoformat()
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"Saved to {self._path}")

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def add_user(
        self,
        user_id:            str,
        display_name:       str,
        employee_id:        str,
        department:         str,
        grokly_role:        str,
        application_access: list[str],
        granted_by:         str,
        start_date:         str | None = None,
        end_date:           str | None = None,
        reason:             str = "",
    ) -> None:
        """
        Add a new user or append an assignment to an existing user.
        Permanent assignment when end_date is None; temporary otherwise.
        """
        data  = self._load()
        by_id = {u["user_id"]: u for u in data["users"]}

        if user_id not in by_id:
            by_id[user_id] = {
                "user_id":          user_id,
                "display_name":     display_name,
                "employee_id":      employee_id,
                "department":       department,
                "account_status":   "active",
                "role_assignments": [],
                "temporary_access": [],
            }
        else:
            # Update display fields in case they changed
            by_id[user_id]["display_name"] = display_name
            by_id[user_id]["department"]   = department

        assignment = {
            "assignment_id":      f"ASSIGN-{uuid.uuid4().hex[:8].upper()}",
            "grokly_role":        grokly_role,
            "application_access": application_access,
            "granted_by":         granted_by,
            "start_date":         start_date or str(date.today()),
            "end_date":           end_date,
            "reason":             reason,
            "status":             "active",
        }

        if end_date:
            by_id[user_id]["temporary_access"].append(assignment)
            print(f"Added temporary access: {user_id} -> {grokly_role} until {end_date}")
        else:
            by_id[user_id]["role_assignments"].append(assignment)
            print(f"Added permanent role: {user_id} -> {grokly_role}")

        data["users"] = list(by_id.values())
        self._save(data)

    def deactivate_user(self, user_id: str, deactivated_by: str) -> None:
        """Deactivate a user — they can no longer access GroklyAI."""
        data    = self._load()
        matched = False
        for user in data["users"]:
            if user["user_id"].lower() == user_id.lower():
                user["account_status"] = "inactive"
                user.setdefault("audit", {}).update({
                    "deactivated_by":   deactivated_by,
                    "deactivated_date": str(date.today()),
                })
                matched = True
                print(f"Deactivated: {user_id}")
        if not matched:
            print(f"User not found: {user_id}")
        self._save(data)

    def expire_temporary_access(self) -> int:
        """
        Mark expired temporary assignments as 'expired'.
        Run daily — ideally via Windows Task Scheduler or cron.
        """
        data          = self._load()
        today         = date.today()
        expired_count = 0

        for user in data["users"]:
            for temp in user.get("temporary_access", []):
                if temp.get("status") != "active":
                    continue
                end_str = temp.get("end_date")
                if not end_str:
                    continue
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                if end < today:
                    temp["status"] = "expired"
                    expired_count  += 1
                    print(f"Expired: {user['user_id']} -> {temp['grokly_role']}")

        self._save(data)
        print(f"Total expired: {expired_count}")
        return expired_count

    def list_expiring(self, days: int = 7) -> None:
        """Print a table of access expiring within the next *days* days."""
        from grokly.identity.user_manager import UserManager
        expiring = UserManager().get_expiring_access(days)

        if not expiring:
            print(f"No access expiring in next {days} days")
            return

        print(f"\nAccess expiring in next {days} days:")
        print(f"{'User':<30} {'Role':<20} {'Expires':<12} {'Days':>4}")
        print("-" * 70)
        for e in expiring:
            print(
                f"{e['display_name']:<30} "
                f"{e['role']:<20} "
                f"{e['end_date']:<12} "
                f"{e['days_left']:>4}"
            )
