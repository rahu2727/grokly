#!/usr/bin/env python3
"""
GroklyAI Launcher
Run this to start GroklyAI after initial setup.

Usage:
    python launch.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

if os.name == "nt":
    VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

URL = "http://localhost:8501"


def main() -> None:
    if not VENV_PYTHON.exists():
        print("GroklyAI is not set up yet.")
        print("Run:  python setup_wizard.py")
        sys.exit(1)

    print("Starting GroklyAI...")

    proc = subprocess.Popen(
        [
            str(VENV_PYTHON), "-m", "streamlit", "run",
            str(PROJECT_ROOT / "app" / "main.py"),
            "--server.headless", "true",
        ],
        cwd=str(PROJECT_ROOT),
    )

    time.sleep(4)
    webbrowser.open(URL)

    print(f"GroklyAI running at {URL}")
    print("Press Ctrl+C to stop.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nGroklyAI stopped.")


if __name__ == "__main__":
    main()
