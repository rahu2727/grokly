"""
grokly/rbac.py — Role-Based Access Control for persona selection.

Reads grokly/config/role_permissions.json to enforce which personas
each organisational role may access. No code changes are needed when
org roles or persona mappings change — edit the JSON file instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config" / "role_permissions.json"


class RBACManager:
    """Enforces persona access by org role."""

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        with config_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        self._perms: dict[str, dict] = data["role_permissions"]
        self._default_org_role: str = data.get("default_org_role", "end_user")
        self._allow_self_selection: bool = data.get("allow_self_selection", True)

    # ------------------------------------------------------------------
    # Org roles
    # ------------------------------------------------------------------

    def get_org_roles(self) -> list[str]:
        """Return all configured org role keys."""
        return list(self._perms.keys())

    def get_org_role_labels(self) -> dict[str, str]:
        """Return {role_key: label} for all org roles."""
        return {k: v.get("label", k) for k, v in self._perms.items()}

    def get_default_org_role(self) -> str:
        return self._default_org_role

    def is_self_selection_allowed(self) -> bool:
        return self._allow_self_selection

    # ------------------------------------------------------------------
    # Persona access
    # ------------------------------------------------------------------

    def get_allowed_personas(self, org_role: str) -> list[str]:
        """Return the persona keys this org role may use."""
        entry = self._perms.get(org_role, self._perms.get(self._default_org_role, {}))
        return list(entry.get("allowed_personas", ["end_user"]))

    def get_default_persona(self, org_role: str) -> str:
        entry = self._perms.get(org_role, self._perms.get(self._default_org_role, {}))
        return entry.get("default_persona", "end_user")

    def can_switch_freely(self, org_role: str) -> bool:
        entry = self._perms.get(org_role, {})
        return bool(entry.get("can_switch_freely", False))

    def get_allowed_persona_labels(
        self,
        org_role: str,
        all_persona_labels: dict[str, str],
    ) -> list[str]:
        """
        Return ordered list of display labels for the personas this role may use.

        Parameters
        ----------
        org_role : str
            The user's organisational role key.
        all_persona_labels : dict[str, str]
            Mapping of {persona_key: display_label} for every persona the app knows.
        """
        allowed_keys = self.get_allowed_personas(org_role)
        return [
            all_persona_labels[k]
            for k in allowed_keys
            if k in all_persona_labels
        ]

    def is_persona_allowed(self, org_role: str, persona_key: str) -> bool:
        return persona_key in self.get_allowed_personas(org_role)
