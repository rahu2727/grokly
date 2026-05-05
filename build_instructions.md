# Building the GroklyAI Windows Installer

This produces a single `GroklyAI-Setup.exe` that clients can double-click.
Python does **not** need to be pre-installed on the client machine.

## When to use this

Use the `.exe` approach when delivering GroklyAI to a client who:
- Has no Python on their machine
- Should not need to install anything manually
- Just needs to double-click and follow prompts

For clients who already have Python 3.11+, the simpler `python setup_wizard.py`
approach is recommended.

## Prerequisites

Must be run on a **Windows** machine.

```
pip install pyinstaller
```

## Build

```
python build_installer.py
```

Output: `dist/GroklyAI-Setup.exe` (~50–80 MB)

## What the .exe does

1. Extracts bundled Python runtime and dependencies
2. Runs `setup_wizard.py` in a terminal window
3. The user follows the same wizard prompts as the Python version

## Distribute

Send the single `GroklyAI-Setup.exe` file to the client.

The client:
1. Double-clicks `GroklyAI-Setup.exe`
2. A terminal window opens with the setup wizard
3. Follows the prompts (org name, API key, knowledge sources)
4. GroklyAI launches in their browser when complete

## After first setup

The `.exe` only needs to be run once. After that, the client uses:

```
python launch.py
```

(inside the GroklyAI folder that was created during setup)

## Notes

- Build must be done on Windows — PyInstaller creates platform-specific binaries
- The `.exe` size is large (~50–80 MB) because it bundles Python itself
- Virus scanners sometimes flag PyInstaller `.exe` files as suspicious — this is a
  false positive. Submit to your AV vendor for whitelisting if needed.
- The alternative `.exe`-free approach (`python setup_wizard.py`) is preferred when
  clients have Python installed.
