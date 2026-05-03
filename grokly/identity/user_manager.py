"""
grokly/identity/user_manager.py — User identity and role management.

Reads from users_master.json to determine who the user is, what roles
they hold, which applications they can access, and whether their access
is valid on a given date.

SAP equivalent: reading PA0001 infotype with validity-date check at runtime.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MASTER_FILE = Path(__file__).parent.parent / "config" / "users_master.json"
_RBAC_FILE   = Path(__file__).parent.parent / "config" / "role_permissions.json"

# Highest → lowest privilege ordering used when a user holds multiple roles
_ROLE_PRIORITY = [
    "it_developer",
    "manager",
    "business_analyst",
    "uat_tester",
    "support",
    "end_user",
]


class UserManager:
    """Read-only view of the user master file with date-aware role resolution."""

    def __init__(
        self,
        master_file: str | Path = _MASTER_FILE,
        rbac_file:   str | Path = _RBAC_FILE,
    ) -> None:
        self._master_path = Path(master_file)
        self._rbac_path   = Path(rbac_file)
        self._users:       dict[str, dict] | None = None
        self._rbac:        dict | None = None
        self._loaded_date: date | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_if_needed(self) -> None:
        """Reload on date change — supports daily file-swap without restart."""
        today = date.today()
        if self._users is None or self._loaded_date != today:
            self._load_master_file()
            self._loaded_date = today

    def _load_master_file(self) -> None:
        if not self._master_path.exists():
            self._users = {}
            logger.warning("users_master.json not found at %s", self._master_path)
            return

        with self._master_path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        self._users = {
            u["user_id"].lower(): u
            for u in data.get("users", [])
        }

        with self._rbac_path.open(encoding="utf-8") as fh:
            self._rbac = json.load(fh)

        print(f"[UserManager] Loaded {len(self._users)} users from master file")

    # ------------------------------------------------------------------
    # Public — lookup
    # ------------------------------------------------------------------

    def get_user(self, user_id: str) -> Optional[dict]:
        """Return the raw user record for *user_id*, or None if not found."""
        self._load_if_needed()
        return self._users.get(user_id.lower())

    def get_effective_role(
        self,
        user_id:     str,
        application: str,
        as_of_date:  date | None = None,
    ) -> str | None:
        """
        Determine the effective organisational role for a user on a date.

        Checks both permanent role_assignments and temporary_access.
        Temporary access overrides permanent when both are active on the
        same date — the highest-priority active role is returned.

        Returns None when the account is inactive (access denied).
        Returns "end_user" as the safe default when the user is unknown
        or has no valid assignments.
        """
        if as_of_date is None:
            as_of_date = date.today()

        user = self.get_user(user_id)
        if not user:
            return "end_user"

        if user.get("account_status") != "active":
            return None

        effective: list[str] = []

        def _parse(d: str | None) -> date | None:
            if d is None:
                return None
            if isinstance(d, date):
                return d
            return datetime.strptime(d, "%Y-%m-%d").date()

        def _is_valid(assignment: dict) -> bool:
            if assignment.get("status") != "active":
                return False
            start = _parse(assignment.get("start_date"))
            end   = _parse(assignment.get("end_date"))
            apps  = assignment.get("application_access", [])

            date_ok = (
                start is not None
                and start <= as_of_date
                and (end is None or end >= as_of_date)
            )
            app_ok = not apps or application in apps or "*" in apps
            return date_ok and app_ok

        for assignment in user.get("role_assignments", []):
            if _is_valid(assignment):
                effective.append(assignment["grokly_role"])

        for temp in user.get("temporary_access", []):
            if _is_valid(temp):
                effective.append(temp["grokly_role"])

        if not effective:
            return "end_user"

        for role in _ROLE_PRIORITY:
            if role in effective:
                return role

        return effective[0]

    def get_expiring_access(self, days_ahead: int = 7) -> list[dict]:
        """
        Return temporary assignments expiring within the next *days_ahead* days.
        Useful for admin dashboards and automated reminders.
        """
        self._load_if_needed()
        today   = date.today()
        results = []

        for user_id, user in self._users.items():
            for temp in user.get("temporary_access", []):
                if temp.get("status") != "active":
                    continue
                end_str = temp.get("end_date")
                if not end_str:
                    continue
                end       = datetime.strptime(end_str, "%Y-%m-%d").date()
                days_left = (end - today).days
                if 0 <= days_left <= days_ahead:
                    results.append({
                        "user_id":      user_id,
                        "display_name": user.get("display_name", user_id),
                        "role":         temp["grokly_role"],
                        "end_date":     end_str,
                        "days_left":    days_left,
                        "reason":       temp.get("reason", ""),
                    })

        return sorted(results, key=lambda x: x["days_left"])

    def get_access_summary(self, user_id: str) -> dict:
        """Return a full access summary for a user (used in profile display and audit)."""
        user = self.get_user(user_id)
        if not user:
            return {"found": False, "user_id": user_id}

        today          = date.today()
        effective_role = self.get_effective_role(user_id, "*", today)

        from grokly.rbac import RBACManager
        allowed_personas = RBACManager().get_allowed_personas(effective_role or "end_user")

        return {
            "found":                True,
            "user_id":              user_id,
            "display_name":         user.get("display_name", ""),
            "department":           user.get("department", ""),
            "effective_role":       effective_role,
            "allowed_personas":     allowed_personas,
            "account_status":       user.get("account_status", ""),
            "has_temporary_access": bool(user.get("temporary_access")),
        }

    # ------------------------------------------------------------------
    # Public — authentication (Phase 2 — email lookup only)
    # ------------------------------------------------------------------

    def authenticate_simple(self, email: str) -> dict:
        """
        Phase 2 authentication: email-lookup only, no password.
        Assumes network-level security (internal deployment).
        Phase 3 will replace this with Azure AD OAuth token validation.
        """
        user = self.get_user(email)

        if not user:
            return {
                "authenticated": False,
                "reason":        "User not found in GroklyAI directory",
            }

        if user.get("account_status") != "active":
            return {
                "authenticated": False,
                "reason":        "Account is not active",
            }

        return {
            "authenticated": True,
            "user_id":       email,
            "display_name":  user.get("display_name", email),
            "department":    user.get("department", ""),
        }
