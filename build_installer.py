#!/usr/bin/env python3
"""
build_installer.py — Build a self-contained Windows .exe for GroklyAI.

This script MUST be run on Windows with PyInstaller installed.
It produces dist/GroklyAI-Setup.exe which can be sent to clients.
The .exe bundles Python and launches the setup_wizard.py flow.

Prerequisites:
    pip install pyinstaller

Usage:
    python build_installer.py

Output:
    dist/GroklyAI-Setup.exe  (~50–80 MB)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def build() -> None:
    print("Building GroklyAI Windows installer...")
    print("This requires PyInstaller and must run on Windows.\n")

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Install it with:")
        print("  pip install pyinstaller")
        sys.exit(1)

    icon_path = PROJECT_ROOT / "grokly" / "assets" / "icon.ico"
    icon_arg  = ["--icon", str(icon_path)] if icon_path.exists() else []

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--console",                       # keep console for wizard prompts
        "--name", "GroklyAI-Setup",
        "--add-data", "requirements.txt;.",
        "--add-data", "grokly/config;grokly/config",
        "--add-data", "app;app",
        *icon_arg,
        "setup_wizard.py",
    ]

    print("Running:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode == 0:
        exe_path = PROJECT_ROOT / "dist" / "GroklyAI-Setup.exe"
        size_mb  = exe_path.stat().st_size / (1024 ** 2) if exe_path.exists() else 0
        print(f"\nBuild complete: dist/GroklyAI-Setup.exe  ({size_mb:.0f} MB)")
        print("\nDistribute this single .exe file to clients.")
        print("They double-click it and follow the setup prompts.")
        print("Python does NOT need to be pre-installed on their machine.")
    else:
        print("\nBuild failed. Check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    if sys.platform != "win32":
        print("Note: PyInstaller .exe files must be built on Windows.")
        print("Run this script on a Windows machine.")
        sys.exit(1)
    build()
