"""
grokly/prompt_loader.py — Loads and caches AI prompts from grokly/prompts/*.yaml.

Prompts are managed as YAML files by prompt engineers and admins.
No Python changes are needed when updating a prompt — edit the YAML file only.

Public API
----------
    from grokly.prompt_loader import PromptLoader
    loader = PromptLoader()
    system = loader.get_system_prompt("commentary_python")
    user   = loader.format_user_prompt(
                 "commentary_python",
                 function_name="my_fn",
                 file_path="erpnext/buying/utils.py",
                 module_name="buying",
                 function_source_code="def my_fn(): ..."
             )
    print(loader.summary())
"""

from __future__ import annotations

from pathlib import Path

import yaml


class PromptLoader:
    """
    Loads prompt YAML files from grokly/prompts/ and provides typed accessors.

    Prompts are cached in memory after first load so each file is read once
    per process — safe to call repeatedly inside a processing loop.
    """

    def __init__(self, prompts_dir: str = "grokly/prompts") -> None:
        _project_root = Path(__file__).parent.parent
        p = Path(prompts_dir)
        self.prompts_dir: Path = p if p.is_absolute() else (_project_root / prompts_dir)
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Core loader
    # ------------------------------------------------------------------

    def load(self, prompt_name: str) -> dict:
        """
        Return the parsed YAML dict for *prompt_name*.

        Results are cached — the file is only read once per process.

        Raises
        ------
        FileNotFoundError
            If the .yaml file does not exist, with a helpful message listing
            available prompts.
        """
        if prompt_name in self._cache:
            return self._cache[prompt_name]

        path = self.prompts_dir / f"{prompt_name}.yaml"
        if not path.exists():
            available = self.list_prompts()
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
                f"Available prompts: {available}\n"
                f"Prompts directory: {self.prompts_dir}"
            )

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        self._cache[prompt_name] = data
        return data

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    def get_system_prompt(self, prompt_name: str) -> str:
        """Return the system prompt string for *prompt_name*."""
        return self.load(prompt_name)["system"]

    def format_user_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load the user_template for *prompt_name* and format it with **kwargs**.

        All {placeholder} tokens in the template must be supplied as keyword
        arguments; missing keys raise a KeyError.
        """
        template = self.load(prompt_name)["user_template"]
        return template.format(**kwargs)

    def get_settings(self, prompt_name: str) -> dict:
        """Return the settings dict (temperature, etc.) for *prompt_name*.

        Note: model and max_tokens are no longer stored in YAML — use
        grokly.model_config.get_agent_config() to retrieve those.
        """
        return self.load(prompt_name).get("settings", {})

    def get_version(self, prompt_name: str) -> str:
        """Return the version string for *prompt_name*."""
        return str(self.load(prompt_name).get("version", "unknown"))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_prompts(self) -> list[str]:
        """Return prompt names (filenames without .yaml) sorted alphabetically."""
        if not self.prompts_dir.exists():
            return []
        return sorted(p.stem for p in self.prompts_dir.glob("*.yaml"))

    def summary(self) -> str:
        """Return a formatted table of all prompts with name, version, language, status."""
        from grokly.brand import APP_NAME
        prompts = self.list_prompts()
        if not prompts:
            return f"No prompts found in {self.prompts_dir}"

        lines: list[str] = [
            f"{APP_NAME} — Prompt Registry",
            "=" * 60,
            f"  {'Name':<28}  {'Ver':<6}  {'Language':<10}  Status",
            f"  {'-'*28}  {'-'*6}  {'-'*10}  ------",
        ]

        for name in prompts:
            try:
                data = self.load(name)
                version  = str(data.get("version", "?"))
                language = data.get("language", "?")
                desc     = data.get("description", "")
                if "PLACEHOLDER" in desc or "placeholder" in desc:
                    status = "Placeholder — needs testing"
                elif "Tested" in desc or "approved" in desc:
                    status = "Tested and approved"
                else:
                    status = "Unknown"
                lines.append(
                    f"  {name:<28}  {version:<6}  {language:<10}  {status}"
                )
            except Exception as exc:
                lines.append(f"  {name:<28}  [ERROR loading: {exc}]")

        lines.append("=" * 60)
        return "\n".join(lines)
