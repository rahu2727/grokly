"""
grokly/agents/application_router.py — Routes queries to the right application(s).

Business/manager roles search all applications (cross-app mode).
Developer/uat_tester roles detect which application the question is about
and filter ChromaDB results accordingly.

Reads grokly/config/applications.json — no code changes when adding apps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "applications.json"

# Roles that should see context from all applications
_CROSS_APP_ROLES = {"business_user", "manager", "end_user", "doc_generator", "consultant"}

# Roles that target a specific application
_APP_SPECIFIC_ROLES = {"developer", "system_admin", "uat_tester"}


class ApplicationRouter:
    """Determines which application(s) to search based on user role and question."""

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        with config_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        self._apps: list[dict] = [a for a in data["applications"] if a.get("enabled", True)]
        self._routing: dict = data.get("routing", {})
        self._default_app: str = self._routing.get("default_application", "")
        self._confidence_threshold: float = self._routing.get(
            "confidence_threshold_for_routing", 0.5
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_applications(self) -> list[dict]:
        """Return all enabled applications."""
        return list(self._apps)

    def get_application_labels(self) -> dict[str, str]:
        """Return {key: label} for all enabled applications."""
        return {a["key"]: a["label"] for a in self._apps}

    def route(
        self,
        question: str,
        role: str,
        selected_application: str = "",
    ) -> dict:
        """
        Determine routing context for a query.

        Returns a dict with:
          search_all    : bool   — True = search all apps (no app filter)
          application   : str    — key of the targeted app (empty if search_all)
          app_label     : str    — display label of the targeted app
          reason        : str    — human-readable routing reason
        """
        role = role.lower().replace(" ", "_")

        # Business-mode roles always search all applications
        if role in _CROSS_APP_ROLES:
            return {
                "search_all":  True,
                "application": "",
                "app_label":   "All applications",
                "reason":      f"role={role} uses cross-application search",
            }

        # Developer/UAT — honour explicit selection first
        if selected_application:
            app = self._get_app(selected_application)
            if app:
                return {
                    "search_all":  False,
                    "application": app["key"],
                    "app_label":   app["label"],
                    "reason":      f"explicit selection: {app['label']}",
                }

        # Auto-detect from question keywords
        detected = self._detect_application(question)
        if detected:
            return {
                "search_all":  False,
                "application": detected["key"],
                "app_label":   detected["label"],
                "reason":      f"auto-detected from question keywords: {detected['label']}",
            }

        # Only one app configured — use it directly
        if len(self._apps) == 1:
            app = self._apps[0]
            return {
                "search_all":  False,
                "application": app["key"],
                "app_label":   app["label"],
                "reason":      "single application configured",
            }

        # Fall back to default
        app = self._get_app(self._default_app)
        if app:
            return {
                "search_all":  False,
                "application": app["key"],
                "app_label":   app["label"],
                "reason":      f"default application: {app['label']}",
            }

        # No routing possible — search all
        return {
            "search_all":  True,
            "application": "",
            "app_label":   "All applications",
            "reason":      "no routing target found, searching all",
        }

    def build_search_filter(self, application_context: dict) -> dict | None:
        """
        Convert routing context into a ChromaDB `where` filter clause.

        Returns None when cross-application search is requested (no filter).
        """
        if application_context.get("search_all"):
            return None
        app_key = application_context.get("application", "")
        if not app_key:
            return None
        return {"application": {"$eq": app_key}}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_app(self, key: str) -> dict | None:
        for app in self._apps:
            if app["key"] == key:
                return app
        return None

    def _detect_application(self, question: str) -> dict | None:
        """Keyword match against each app's domain list and technical prefix."""
        q = question.lower()
        best: dict | None = None
        best_hits = 0

        for app in self._apps:
            hits = 0
            prefix = app.get("technical_prefix", "").lower()
            if prefix and prefix in q:
                hits += 3  # strong signal

            for domain in app.get("domains", []):
                if domain.lower() in q:
                    hits += 1

            if hits > best_hits:
                best_hits = hits
                best = app

        return best if best_hits > 0 else None
