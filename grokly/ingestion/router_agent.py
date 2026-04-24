"""
grokly/ingestion/router_agent.py — Detects code language and selects the expert prompt.

The RouterAgent inspects a file path and source code snippet, then returns
the name of the appropriate prompt from the YAML prompt registry.

Supported languages
-------------------
  python  — ERPNext / Frappe Python server-side code (.py)
  abap    — SAP ABAP programs, function modules, classes (.abap, .prog, .fugr)
  fiori   — SAP Fiori UI5 controllers and views (.controller.js, .view.xml,
             .fragment.xml)

Detection order
---------------
1. File extension — unambiguous match wins immediately.
2. Content indicators — count keyword hits per language; highest score wins.
3. Default — "python" if no signal found.

Public API
----------
    from grokly.ingestion.router_agent import RouterAgent
    router = RouterAgent()
    lang        = router.detect_language(file_path, content)
    prompt_name = router.get_prompt_name(file_path, content)
    expert      = router.get_expert_description(lang)
"""

from __future__ import annotations


class RouterAgent:
    """
    Detects code language from file path and content, then maps to an expert prompt.

    All detection logic is driven by LANGUAGE_RULES — add a new language block
    to support additional code types without changing any other code.
    """

    LANGUAGE_RULES: dict[str, dict] = {
        "python": {
            "extensions": [".py"],
            "prompt":     "commentary_python",
            "expert":     "Python/ERPNext Expert Agent",
            "indicators": ["def ", "import frappe", "class ", "self."],
        },
        "abap": {
            "extensions": [".abap", ".prog", ".fugr"],
            "prompt":     "commentary_abap",
            "expert":     "SAP ABAP Expert Agent",
            "indicators": ["FORM ", "FUNCTION ", "METHOD ", "DATA:", "TYPES:", "SELECT"],
        },
        "fiori": {
            "extensions": [".controller.js", ".view.xml", ".fragment.xml"],
            "prompt":     "commentary_fiori",
            "expert":     "Fiori UI5 Expert Agent",
            "indicators": ["sap.ui.define", "onInit", "getView()", "ODataModel"],
        },
    }

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def detect_language(self, file_path: str, content: str) -> str:
        """
        Detect the programming language of a file.

        Checks extension first (unambiguous), then scores content indicators,
        then defaults to "python".
        """
        lower_path = file_path.lower()

        # Extension match (highest priority)
        for language, rules in self.LANGUAGE_RULES.items():
            for ext in rules["extensions"]:
                if lower_path.endswith(ext):
                    print(
                        f"    [Router] '{file_path}' → {language} "
                        f"(extension match: '{ext}')"
                    )
                    return language

        # Content indicator scoring
        scores: dict[str, int] = {}
        for language, rules in self.LANGUAGE_RULES.items():
            score = sum(
                1 for indicator in rules["indicators"]
                if indicator in content
            )
            scores[language] = score

        best_language = max(scores, key=lambda k: scores[k])
        best_score    = scores[best_language]

        if best_score > 0:
            print(
                f"    [Router] '{file_path}' → {best_language} "
                f"(indicator score: {best_score})"
            )
            return best_language

        print(
            f"    [Router] '{file_path}' → python "
            f"(no extension or indicator match — default)"
        )
        return "python"

    # ------------------------------------------------------------------
    # Prompt and expert accessors
    # ------------------------------------------------------------------

    def get_prompt_name(self, file_path: str, content: str) -> str:
        """Return the prompt registry name for this file's language."""
        language = self.detect_language(file_path, content)
        return self.LANGUAGE_RULES[language]["prompt"]

    def get_expert_description(self, language: str) -> str:
        """
        Return a human-readable label for the expert handling this language.

        Falls back to the language name if not found in LANGUAGE_RULES.
        """
        rules = self.LANGUAGE_RULES.get(language)
        if rules:
            return rules["expert"]
        return f"{language.upper()} Expert Agent"
