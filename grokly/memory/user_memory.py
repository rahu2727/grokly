"""
grokly/memory/user_memory.py — Long-term user memory backed by ChromaDB.

Stores per-user profiles in a dedicated "grokly_user_profiles" collection.
Each profile is a document with metadata capturing role preferences, usage
stats, and recently explored topics.

ChromaDB metadata values must be primitives — lists are JSON-serialised to str.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from grokly.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_PROFILES_COLLECTION = "grokly_user_profiles"
_UNKNOWN_USER = "anonymous"


class UserMemory:
    def __init__(self, store: ChromaStore | None = None) -> None:
        base_store = store or ChromaStore()
        self._col = base_store._client.get_or_create_collection(
            name=_PROFILES_COLLECTION,
        )

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_or_create_profile(self, user_id: str) -> dict:
        """Return existing profile or create a blank one."""
        user_id = user_id.strip() or _UNKNOWN_USER
        try:
            result = self._col.get(ids=[user_id], include=["metadatas"])
            if result["ids"]:
                meta = result["metadatas"][0]
                return self._deserialise(meta)
        except Exception as exc:
            logger.debug("Profile lookup failed for %s: %s", user_id, exc)

        profile = {
            "user_id":         user_id,
            "preferred_role":  "business_user",
            "question_count":  0,
            "last_seen":       datetime.utcnow().isoformat(),
            "topics_explored": [],
        }
        self._save_profile(user_id, profile)
        return profile

    def update_profile(self, user_id: str, updates: dict) -> None:
        """Merge updates into the stored profile."""
        profile = self.get_or_create_profile(user_id)
        profile.update(updates)
        profile["last_seen"] = datetime.utcnow().isoformat()
        self._save_profile(user_id, profile)

    def record_question(self, user_id: str, question: str, role: str) -> None:
        """Increment question count, track role usage, and add topic."""
        profile = self.get_or_create_profile(user_id)

        profile["question_count"] = profile.get("question_count", 0) + 1
        profile["preferred_role"] = role

        topics: list[str] = profile.get("topics_explored", [])
        topic = question[:80]
        if topic not in topics:
            topics.insert(0, topic)
        profile["topics_explored"] = topics[:20]

        profile["last_seen"] = datetime.utcnow().isoformat()
        self._save_profile(user_id, profile)

    def get_preferred_role(self, user_id: str) -> str:
        profile = self.get_or_create_profile(user_id)
        return profile.get("preferred_role", "business_user")

    def get_stats(self, user_id: str) -> dict:
        profile = self.get_or_create_profile(user_id)
        return {
            "question_count":  profile.get("question_count", 0),
            "preferred_role":  profile.get("preferred_role", "business_user"),
            "last_seen":       profile.get("last_seen", ""),
            "topics_explored": profile.get("topics_explored", []),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_profile(self, user_id: str, profile: dict) -> None:
        """Upsert the profile document into the ChromaDB collection."""
        try:
            serialised = self._serialise(profile)
            # Document text is a readable summary for potential future search
            doc_text = (
                f"User {user_id} | role={profile.get('preferred_role')} "
                f"| questions={profile.get('question_count', 0)}"
            )
            self._col.upsert(
                ids=[user_id],
                documents=[doc_text],
                metadatas=[serialised],
            )
        except Exception as exc:
            logger.warning("Failed to save profile for %s: %s", user_id, exc)

    @staticmethod
    def _serialise(profile: dict) -> dict:
        """Convert list values to JSON strings for ChromaDB compatibility."""
        out = {}
        for k, v in profile.items():
            if isinstance(v, list):
                out[k] = json.dumps(v)
            elif isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    @staticmethod
    def _deserialise(meta: dict) -> dict:
        """Restore JSON-serialised list fields."""
        out = dict(meta)
        for k, v in out.items():
            if isinstance(v, str) and v.startswith("["):
                try:
                    out[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
        return out
