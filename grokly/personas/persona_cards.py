"""
grokly/personas/persona_cards.py — Persona routing configuration.

Maps persona keys to display labels and answer style instructions.
Labels are sourced from brand.py — change them there to rebrand.
"""

from __future__ import annotations

from grokly.brand import PERSONA_LABELS


PERSONA_CARDS: dict[str, dict] = {
    key: {
        "label":       label,
        "tone":        _TONES.get(key, "clear and helpful"),
        "detail":      _DETAIL.get(key, "medium"),
    }
    for key, label in PERSONA_LABELS.items()
}

_TONES = {
    "end_user":       "simple, step-by-step, no jargon",
    "business_user":  "business-focused, practical",
    "manager":        "summary-first, high-level impact",
    "developer":      "technical, precise, code-aware",
    "uat_tester":     "process-oriented, test-step focused",
    "doc_generator":  "structured, comprehensive, formal",
}

_DETAIL = {
    "end_user":       "low",
    "business_user":  "medium",
    "manager":        "low",
    "developer":      "high",
    "uat_tester":     "medium",
    "doc_generator":  "high",
}

# Rebuild after defining the helper dicts
PERSONA_CARDS = {
    key: {
        "label":  label,
        "tone":   _TONES.get(key, "clear and helpful"),
        "detail": _DETAIL.get(key, "medium"),
    }
    for key, label in PERSONA_LABELS.items()
}


def get_persona_card(persona_key: str) -> dict:
    """Return the persona card for *persona_key*, defaulting to end_user."""
    return PERSONA_CARDS.get(persona_key, PERSONA_CARDS["end_user"])
