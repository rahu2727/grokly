#!/usr/bin/env python3
"""
GroklyAI Setup Wizard
One command to install, configure and launch.

Usage:
    python setup_wizard.py
    python setup_wizard.py --dry-run   # show what would happen, no changes

Requirements:
    Python 3.11+   (checked automatically)
    Internet       (for package installation)
    Anthropic key  (wizard prompts for this)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import date, datetime
from pathlib import Path

# Wizard always uses development environment
os.environ.setdefault("GROKLY_ENV", "development")
try:
    from grokly.env_loader import load_environment
    load_environment()
except Exception:
    pass  # venv may not be active yet during first-time setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION      = "0.1.0"
MIN_PYTHON   = (3, 11)
MIN_DISK_GB  = 2
PROJECT_ROOT = Path(__file__).parent

if os.name == "nt":
    VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    VENV_PIP    = PROJECT_ROOT / "venv" / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
    VENV_PIP    = PROJECT_ROOT / "venv" / "bin" / "pip"

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

def _enable_ansi_windows() -> None:
    """Enable VT100 ANSI codes and UTF-8 output on Windows."""
    if os.name != "nt":
        return
    # Switch console to UTF-8 code page
    os.system("chcp 65001 >nul 2>&1")
    # Reconfigure stdout/stderr for UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    # Enable VT100 ANSI processing
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode   = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

_enable_ansi_windows()

_USE_ANSI = sys.stdout.isatty()

GRN  = "\033[92m" if _USE_ANSI else ""
YLW  = "\033[93m" if _USE_ANSI else ""
RED  = "\033[91m" if _USE_ANSI else ""
CYN  = "\033[96m" if _USE_ANSI else ""
BOLD = "\033[1m"  if _USE_ANSI else ""
DIM  = "\033[2m"  if _USE_ANSI else ""
RST  = "\033[0m"  if _USE_ANSI else ""

def _ok(msg: str)   -> str: return f"{GRN}✅{RST}  {msg}"
def _warn(msg: str) -> str: return f"{YLW}⚠️ {RST}  {msg}"
def _fail(msg: str) -> str: return f"{RED}✗{RST}  {msg}"
def _step(msg: str) -> str: return f"{CYN}⏳{RST}  {msg}"
def _dot(msg: str)  -> str: return f"○  {msg}"

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    lines = [
        "╔══════════════════════════════════════════════╗",
        "║          GroklyAI Setup Wizard               ║",
        "║    Enterprise Knowledge Assistant            ║",
        f"║    Version {VERSION:<34}║",
        "╚══════════════════════════════════════════════╝",
    ]
    print()
    for ln in lines:
        print(f"  {BOLD}{ln}{RST}")
    print()


def print_step(n: int, title: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RST}")
    print(f"{BOLD}  Step {n}: {title}{RST}")
    print(f"{BOLD}{'─' * 50}{RST}\n")


def _progress_bar(fraction: float, width: int = 28, label: str = "") -> str:
    filled = int(width * min(fraction, 1.0))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(min(fraction, 1.0) * 100)
    suffix = f" — {label}" if label else ""
    return f"[{bar}] {pct:>3}%{suffix}"


# ---------------------------------------------------------------------------
# Spinner (used during pip install)
# ---------------------------------------------------------------------------

class _Spinner:
    """Thread-backed terminal spinner with an updatable message."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "") -> None:
        self._msg   = message
        self._stop  = threading.Event()
        self._lock  = threading.Lock()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            with self._lock:
                msg = self._msg
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {frame} {msg:<60}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def update(self, msg: str) -> None:
        with self._lock:
            self._msg = msg

    def __enter__(self) -> "_Spinner":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()
        sys.stdout.write(f"\r{' ' * 70}\r")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Ingestion live display
# ---------------------------------------------------------------------------

class _IngestionDisplay:
    """In-place multi-line status board for ingestion progress."""

    STATUS_ICONS = {
        "waiting":  "○ ",
        "running":  f"{CYN}⏳{RST}",
        "done":     f"{GRN}✅{RST}",
        "skipped":  f"{YLW}⚠️ {RST}",
        "error":    f"{RED}✗{RST} ",
    }

    def __init__(self, sources: list[dict]) -> None:
        self._labels  = [s["label"] for s in sources]
        self._status  = {s["label"]: "waiting" for s in sources}
        self._detail  = {s["label"]: ""         for s in sources}
        self._printed = 0

    def update(self, label: str, status: str, detail: str = "") -> None:
        self._status[label] = status
        self._detail[label] = detail
        self._refresh()

    def _refresh(self) -> None:
        if self._printed and _USE_ANSI:
            sys.stdout.write(f"\033[{self._printed}A")
        for label in self._labels:
            icon   = self.STATUS_ICONS.get(self._status[label], "  ")
            detail = self._detail[label]
            line   = f"  {icon} {label:<22} {detail}"
            if _USE_ANSI:
                sys.stdout.write(f"\033[2K\r{line}\n")
            else:
                print(f"  {line}")
        sys.stdout.flush()
        self._printed = len(self._labels)

    def final(self) -> None:
        self._refresh()
        print()


# ---------------------------------------------------------------------------
# Step 1 — System checks
# ---------------------------------------------------------------------------

def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    return ok, label


def _check_disk() -> tuple[bool, str]:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = usage.free / (1024 ** 3)
        ok = free_gb >= MIN_DISK_GB
        return ok, f"{free_gb:.0f} GB available"
    except Exception:
        return True, "unknown (could not check)"


def _check_internet() -> tuple[bool, str]:
    try:
        socket.setdefaulttimeout(5)
        socket.create_connection(("pypi.org", 443))
        return True, "Connected"
    except OSError:
        return False, "No connection to pypi.org"


def _check_git() -> tuple[bool, str]:
    if shutil.which("git"):
        try:
            r = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            ver = r.stdout.strip().replace("git version ", "")
            return True, ver
        except Exception:
            pass
    return False, "not found — code ingestion unavailable"


def _check_opencv() -> tuple[bool, str]:
    try:
        import cv2
        return True, f"cv2 {cv2.__version__}"
    except ImportError:
        return False, "not installed — screen recording ingestion unavailable (install opencv-python)"


def run_system_checks() -> dict:
    print_step(1, "System Check")

    py_ok,   py_msg   = _check_python()
    disk_ok, disk_msg = _check_disk()
    net_ok,  net_msg  = _check_internet()
    git_ok,  git_msg  = _check_git()
    cv2_ok,  cv2_msg  = _check_opencv()

    print(_ok(f"{py_msg} — OK")           if py_ok   else _fail(f"{py_msg} — FAIL"))
    print(_ok(f"Disk space {disk_msg}")   if disk_ok else _warn(f"Disk space {disk_msg}"))
    print(_ok(f"Internet {net_msg}")      if net_ok  else _warn(f"Internet: {net_msg}"))
    print(_ok(f"Git {git_msg}")           if git_ok  else _warn(f"Git {git_msg}"))
    print(_ok(f"OpenCV {cv2_msg}")        if cv2_ok  else _warn(f"OpenCV: {cv2_msg}"))

    if not py_ok:
        print(f"\n{RED}Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required.{RST}")
        print("Download from: https://www.python.org/downloads/")

    return {
        "python_ok":  py_ok,
        "disk_ok":    disk_ok,
        "internet_ok": net_ok,
        "git_ok":     git_ok,
        "opencv_ok":  cv2_ok,
    }


# ---------------------------------------------------------------------------
# Step 2 — Virtual environment + dependencies
# ---------------------------------------------------------------------------

def setup_venv(dry_run: bool) -> bool:
    print_step(2, "Virtual Environment & Dependencies")

    venv_dir = PROJECT_ROOT / "venv"
    req_file = PROJECT_ROOT / "requirements.txt"

    if not req_file.exists():
        print(_fail("requirements.txt not found — cannot install dependencies"))
        return False

    # Create venv
    if venv_dir.exists():
        print(_ok("Virtual environment already exists — skipping creation"))
    else:
        if dry_run:
            print(_step("[DRY RUN] Would create virtual environment at venv/"))
        else:
            print(_step("Creating virtual environment..."))
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    check=True, capture_output=True,
                )
                print(_ok("Virtual environment created"))
            except subprocess.CalledProcessError as exc:
                print(_fail(f"Failed to create venv: {exc.stderr}"))
                return False

    # Install requirements
    if dry_run:
        total = sum(
            1 for ln in req_file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        )
        print(_step(f"[DRY RUN] Would install {total} packages from requirements.txt"))
        return True

    print()
    print("  Installing GroklyAI dependencies...")
    print("  (This takes 3–8 minutes on first run)\n")
    return _install_with_progress(req_file)


def _install_with_progress(req_file: Path) -> bool:
    """Run pip install showing a live progress bar."""
    total = sum(
        1 for ln in req_file.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    )
    collected = [0]
    current   = ["starting"]

    def _bar_line() -> str:
        frac = min(collected[0] / max(total * 3, 1), 0.95)
        return _progress_bar(frac, label=current[0])

    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stop_ev   = threading.Event()
    bar_lock  = threading.Lock()

    def _bar_thread() -> None:
        while not stop_ev.is_set():
            with bar_lock:
                sys.stdout.write(f"\r  {_bar_line():<65}")
                sys.stdout.flush()
            time.sleep(0.15)

    t = threading.Thread(target=_bar_thread, daemon=True)
    t.start()

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("Collecting"):
            pkg = line.split()[1].split("[")[0].split("==")[0].split(">=")[0]
            with bar_lock:
                collected[0] += 1
                current[0]    = pkg
        elif line.startswith("Successfully installed"):
            with bar_lock:
                current[0] = "finalising"

    rc = proc.wait()
    stop_ev.set()
    t.join()

    if rc == 0:
        sys.stdout.write(f"\r  {_progress_bar(1.0, label='complete'):<65}\n")
        sys.stdout.flush()
        print()
        print(_ok("Dependencies installed"))
    else:
        print(f"\n{_fail('pip install failed — check your internet connection')}")

    return rc == 0


# ---------------------------------------------------------------------------
# Step 3 — Organisation configuration
# ---------------------------------------------------------------------------

def _ask(prompt: str, example: str = "", default: str = "") -> str:
    """Prompt the user for input with optional example and default."""
    if example:
        print(f"  {DIM}Example: {example}{RST}")
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"  {BOLD}{prompt}{RST}: "
    while True:
        value = input(display).strip()
        if not value and default:
            return default
        if value:
            return value
        print(f"  {YLW}This field is required.{RST}")


def _ask_email(prompt: str) -> str:
    """Prompt for an email address with basic format validation."""
    while True:
        value = input(f"  {BOLD}{prompt}{RST}: ").strip()
        if "@" in value and "." in value.split("@")[-1]:
            return value
        print(f"  {YLW}Please enter a valid email address.{RST}")


def collect_org_config(dry_run: bool) -> dict:
    print_step(3, "Organisation Configuration")
    print("  This information is stored in grokly/config/deployment.json")
    print("  and shown in the GroklyAI header.\n")

    org_name   = _ask("What is your organisation name?",    "Acme Corporation")
    deploy_name = _ask("Give this deployment a name",        "ERPNext Pilot")
    admin_email = _ask_email("Enter the admin email address")
    admin_name  = _ask("Enter the admin's full name",        "Jane Smith")

    config = {
        "organisation":    org_name,
        "deployment_name": deploy_name,
        "deployment_date": str(date.today()),
        "deployment_id":   str(uuid.uuid4()),
        "admin_email":     admin_email,
        "admin_name":      admin_name,
        "version":         VERSION,
        "local_only":      True,
        "setup_complete":  False,
    }

    if dry_run:
        print(f"\n  {DIM}[DRY RUN] Would save deployment.json:{RST}")
        print(f"  Organisation:    {org_name}")
        print(f"  Deployment name: {deploy_name}")
        print(f"  Admin:           {admin_name} <{admin_email}>")
    else:
        _save_deployment_config(config)
        _create_admin_user(config)
        print(_ok(f"Configuration saved for {org_name}"))

    return config


def _save_deployment_config(config: dict) -> None:
    path = PROJECT_ROOT / "grokly" / "config" / "deployment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)


def _create_admin_user(config: dict) -> None:
    """Insert the admin into users_master.json (creates file if absent)."""
    path = PROJECT_ROOT / "grokly" / "config" / "users_master.json"

    if path.exists():
        with path.open(encoding="utf-8") as fh:
            master = json.load(fh)
    else:
        master = {
            "metadata": {
                "organisation":  config["organisation"],
                "description":   "GroklyAI user master file.",
                "file_version":  "1.0",
                "generated_date": str(date.today()),
                "generated_by":  "setup_wizard",
                "schema_version": "1.0",
            },
            "users": [],
        }

    existing_ids = {u["user_id"].lower() for u in master.get("users", [])}
    admin_id     = config["admin_email"].lower()

    if admin_id not in existing_ids:
        master["users"].append({
            "user_id":      config["admin_email"],
            "display_name": config["admin_name"],
            "employee_id":  "ADMIN001",
            "department":   "IT",
            "account_status": "active",
            "role_assignments": [
                {
                    "assignment_id":      "ASSIGN-WIZARD-001",
                    "grokly_role":        "it_developer",
                    "application_access": ["erpnext"],
                    "granted_by":         "setup_wizard",
                    "start_date":         str(date.today()),
                    "end_date":           None,
                    "reason":             "Initial administrator",
                    "status":             "active",
                }
            ],
            "temporary_access": [],
        })
        with path.open("w", encoding="utf-8") as fh:
            json.dump(master, fh, indent=2, ensure_ascii=False)
        print(_ok(f"Admin user {config['admin_email']} created"))
    else:
        print(_ok(f"Admin user {config['admin_email']} already exists — skipped"))


# ---------------------------------------------------------------------------
# Step 4 — API keys
# ---------------------------------------------------------------------------

def _ask_secret(prompt: str) -> str:
    """Read a secret without echoing. Falls back to normal input if unavailable."""
    import getpass
    try:
        return getpass.getpass(f"  {BOLD}{prompt}{RST}: ").strip()
    except Exception:
        return input(f"  {BOLD}{prompt}{RST}: ").strip()


def collect_api_keys(config: dict, dry_run: bool) -> None:
    print_step(4, "API Key Configuration")

    print("  GroklyAI needs an Anthropic API key to power its AI features.")
    print(f"  Get yours at: {CYN}https://console.anthropic.com{RST}\n")

    # Check if .env already has a key
    env_path    = PROJECT_ROOT / ".env"
    existing_ak = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY=") and "sk-ant-" in line:
                existing_ak = line.split("=", 1)[1].strip()
                break

    if existing_ak:
        print(_ok("Anthropic API key already configured in .env"))
        ans = input("  Replace it? (y/N): ").strip().lower()
        if ans != "y":
            anthropic_key = existing_ak
        else:
            anthropic_key = ""
    else:
        anthropic_key = ""

    while not anthropic_key:
        key = _ask_secret("Enter your Anthropic API key (starts with sk-ant-)")
        if not key:
            print(f"  {YLW}API key is required.{RST}")
            continue
        if not key.startswith("sk-ant-") or len(key) < 20:
            print(f"  {YLW}Key must start with 'sk-ant-' and be at least 20 characters.{RST}")
            continue
        anthropic_key = key
        break

    print()
    print(f"  Tavily enables web search for GroklyAI (optional).")
    print(f"  Get a free key at: {CYN}https://app.tavily.com{RST}")
    tavily_key = _ask_secret("Tavily API key (press Enter to skip)")

    if dry_run:
        print(f"\n  {DIM}[DRY RUN] Would write .env with ANTHROPIC_API_KEY and "
              f"{'TAVILY_API_KEY' if tavily_key else 'no Tavily key'}{RST}")
    else:
        _write_env(anthropic_key, tavily_key, config)
        print(_ok("API key saved securely to .env"))
        if tavily_key:
            print(_ok("Tavily key saved"))


def _write_env(anthropic_key: str, tavily_key: str, config: dict) -> None:
    """Write (or update) the .env file, preserving any existing entries."""
    env_path = PROJECT_ROOT / ".env"

    # Load existing entries (preserve any others)
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    existing["ANTHROPIC_API_KEY"] = anthropic_key
    if tavily_key:
        existing["TAVILY_API_KEY"] = tavily_key
    existing["GROKLY_ORG"]        = config.get("organisation", "")
    existing["GROKLY_DEPLOYMENT"] = config.get("deployment_name", "GroklyAI")

    lines = [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 5 — Knowledge source selection
# ---------------------------------------------------------------------------

_SOURCE_MENU = [
    {
        "num":    "1",
        "label":  "Code repository",
        "desc":   "GitHub URL or local folder (Python/JS)",
        "key":    "code",
        "ask_path": True,
        "path_prompt": "GitHub URL or local folder path",
        "path_example": "https://github.com/frappe/erpnext  or  C:\\Projects\\MyApp",
    },
    {
        "num":    "2",
        "label":  "Documents folder",
        "desc":   "PDF, Word, PowerPoint files",
        "key":    "docs",
        "ask_path": True,
        "path_prompt": "Path to your documents folder",
        "path_example": "C:\\Documents\\ERPNext Guides",
    },
    {
        "num":    "3",
        "label":  "Q&A pairs",
        "desc":   "Excel or CSV file with questions and answers",
        "key":    "forum",
        "ask_path": True,
        "path_prompt": "Path to your Q&A Excel or CSV file",
        "path_example": "C:\\FAQs\\common_questions.xlsx",
    },
    {
        "num":    "4",
        "label":  "Screenshots folder",
        "desc":   "PNG, JPG images (planned — not yet available)",
        "key":    "screenshots",
        "ask_path": True,
        "path_prompt": "Path to your screenshots folder",
        "path_example": "C:\\Screenshots\\ERPNext",
        "coming_soon": True,
    },
    {
        "num":    "5",
        "label":  "Screen recordings",
        "desc":   "MP4, AVI, MOV files (planned — not yet available)",
        "key":    "recordings",
        "ask_path": True,
        "path_prompt": "Path to your recordings folder",
        "path_example": "C:\\Recordings",
        "coming_soon": True,
    },
    {
        "num":    "6",
        "label":  "Skip for now",
        "desc":   "Add sources later with python ingest.py",
        "key":    "skip",
        "ask_path": False,
    },
]


def select_knowledge_sources() -> list[dict]:
    print_step(5, "Knowledge Sources")
    print("  What would you like GroklyAI to learn from?")
    print("  Enter the numbers of sources to include, separated by commas.\n")

    for s in _SOURCE_MENU:
        flag = f"  {DIM}(coming soon){RST}" if s.get("coming_soon") else ""
        print(f"  {BOLD}{s['num']}.{RST} {s['label']:<22} {DIM}{s['desc']}{RST}{flag}")

    print()
    while True:
        raw = input(f"  {BOLD}Enter numbers (e.g. 1,2,3) or 6 to skip{RST}: ").strip()
        choices = {c.strip() for c in raw.split(",") if c.strip()}
        if not choices:
            print(f"  {YLW}Please enter at least one number.{RST}")
            continue

        if "6" in choices:
            print(_ok("Skipping ingestion — you can run it later with: python ingest.py"))
            return []

        valid_nums = {s["num"] for s in _SOURCE_MENU}
        invalid    = choices - valid_nums
        if invalid:
            print(f"  {YLW}Invalid choices: {', '.join(sorted(invalid))}{RST}")
            continue
        break

    selected: list[dict] = []
    for s in _SOURCE_MENU:
        if s["num"] not in choices:
            continue
        if s.get("coming_soon"):
            print(f"\n  {YLW}{s['label']} is planned for a future release — skipping.{RST}")
            continue
        if not s.get("ask_path"):
            selected.append(dict(s))
            continue

        print(f"\n  {BOLD}{s['label']}{RST}")
        print(f"  {DIM}Example: {s['path_example']}{RST}")
        while True:
            path_str = input(f"  {BOLD}{s['path_prompt']}{RST}: ").strip()
            if not path_str:
                print(f"  {YLW}Path is required for this source.{RST}")
                continue

            # GitHub URL — no validation needed here (wizard will clone)
            if path_str.startswith("http://") or path_str.startswith("https://"):
                entry = dict(s)
                entry["path"] = path_str
                entry["is_url"] = True
                print(_ok(f"Repository URL noted: {path_str}"))
                selected.append(entry)
                break

            # Local path
            p = Path(path_str)
            if s["key"] == "forum":
                if not p.exists():
                    print(f"  {YLW}File not found: {path_str}{RST}")
                    continue
                entry = dict(s)
                entry["path"] = path_str
                print(_ok(f"File found: {p.name}"))
                selected.append(entry)
                break
            else:
                if not p.exists():
                    print(f"  {YLW}Folder not found: {path_str}{RST}")
                    continue
                count = sum(1 for _ in p.rglob("*") if _.is_file())
                entry = dict(s)
                entry["path"] = path_str
                print(_ok(f"Folder found: {count} files"))
                selected.append(entry)
                break

    return selected


# ---------------------------------------------------------------------------
# Step 6 — Ingestion
# ---------------------------------------------------------------------------

# Rough cost estimates per source type
_COST_ESTIMATES: dict[str, str] = {
    "code":  "$0.006 per function × function count",
    "docs":  "$0.01 per page × page count",
    "forum": "free (no API calls)",
}


def _estimate_cost(sources: list[dict]) -> tuple[str, str]:
    """Return (time_estimate, cost_estimate)."""
    has_code  = any(s["key"] == "code"  for s in sources)
    has_docs  = any(s["key"] == "docs"  for s in sources)
    has_forum = any(s["key"] == "forum" for s in sources)

    minutes = 5 * has_forum + 20 * has_docs + 30 * has_code
    cost    = 0.0
    cost   += 5.0 if has_code else 0.0    # rough
    cost   += 1.0 if has_docs else 0.0
    return f"~{max(minutes, 5)} minutes", f"~${cost:.2f}"


def _clone_repo(url: str, dest: Path) -> bool:
    """Clone a GitHub repository, return True on success."""
    print(f"  Cloning {url} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(_fail(f"Clone failed: {exc.stderr.decode()[:200]}"))
        return False


def _run_ingest_source(key: str, path: str | None = None) -> tuple[int, str]:
    """
    Run `python ingest.py --source <key>` via the venv Python.
    Returns (chunk_count, status_string).
    """
    cmd = [str(VENV_PYTHON), str(PROJECT_ROOT / "ingest.py"), "--source", key]
    if path:
        cmd += ["--path", path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=3600,
        )
        output = result.stdout + result.stderr

        # Parse chunk count from output
        chunks = 0
        for line in output.splitlines():
            lower = line.lower()
            if "added" in lower or "chunks" in lower or "ingested" in lower:
                for word in line.split():
                    try:
                        n = int(word.replace(",", ""))
                        if 0 < n < 1_000_000:
                            chunks = max(chunks, n)
                    except ValueError:
                        pass

        if result.returncode != 0:
            return 0, f"error (exit {result.returncode})"
        return chunks, f"{chunks:,} chunks added" if chunks else "complete"

    except subprocess.TimeoutExpired:
        return 0, "timed out after 60 min"
    except FileNotFoundError:
        return 0, "venv not found — run setup first"


def run_ingestion(sources: list[dict], dry_run: bool) -> dict:
    print_step(6, "Building Knowledge Base")

    time_est, cost_est = _estimate_cost(sources)
    print(f"  Estimated time:      {BOLD}{time_est}{RST}")
    print(f"  Estimated API cost:  {BOLD}{cost_est}{RST}")
    print()

    if dry_run:
        for s in sources:
            print(_step(f"[DRY RUN] Would ingest: {s['label']}"))
        return {}

    ans = input(f"  {BOLD}Proceed with ingestion? (y/n){RST}: ").strip().lower()
    if ans != "y":
        print(_warn("Ingestion skipped — run 'python ingest.py' later."))
        return {}

    print()
    print(f"  {BOLD}Building GroklyAI knowledge base...{RST}")
    print(f"  {'─' * 45}")

    display = _IngestionDisplay(sources)
    display.update(sources[0]["label"], "waiting")  # initial render
    totals: dict[str, int] = {}

    for s in sources:
        label = s["label"]
        key   = s["key"]
        path  = s.get("path")

        display.update(label, "running", "Processing...")

        # Clone repo if GitHub URL
        if s.get("is_url") and path:
            clone_dest = PROJECT_ROOT / "repos" / key
            if not clone_dest.exists():
                ok = _clone_repo(path, clone_dest)
                if not ok:
                    display.update(label, "error", "clone failed")
                    totals[label] = 0
                    continue
            path = str(clone_dest)

        count, detail = _run_ingest_source(key, path)
        totals[label] = count
        display.update(label, "done", detail)

    display.final()

    # Summary
    grand_total = sum(totals.values())
    print(f"  {'─' * 45}")
    print(f"  {BOLD}Knowledge base complete!{RST}")
    for label, count in totals.items():
        print(f"    {label:<22} {count:>6,} chunks")
    print(f"    {'─' * 35}")
    print(f"    {'Total':<22} {grand_total:>6,} chunks")
    print(f"  {'─' * 45}\n")

    return totals


# ---------------------------------------------------------------------------
# Step 7 — Launch
# ---------------------------------------------------------------------------

def offer_launch(dry_run: bool) -> None:
    print_step(7, "Launch")
    print(f"  {GRN}GroklyAI is configured and ready.{RST}\n")
    print("  1. Launch now    (opens browser automatically)")
    print("  2. Launch later  (run: python launch.py)")
    print("  3. Exit wizard\n")

    while True:
        choice = input(f"  {BOLD}Enter 1, 2, or 3{RST}: ").strip()
        if choice in ("1", "2", "3"):
            break
        print(f"  {YLW}Please enter 1, 2, or 3.{RST}")

    if choice == "3":
        print("\n  Setup complete. Run 'python launch.py' to start GroklyAI.")
        return

    if choice == "2":
        print("\n" + _ok("Setup complete."))
        print("  Start GroklyAI any time with:  python launch.py")
        return

    # choice == "1"
    if dry_run:
        print(_step("[DRY RUN] Would launch Streamlit at http://localhost:8501"))
        return

    _do_launch()


def _do_launch() -> None:
    import webbrowser

    print(f"\n  {_step('Starting GroklyAI...')}")

    proc = subprocess.Popen(
        [
            str(VENV_PYTHON), "-m", "streamlit", "run",
            str(PROJECT_ROOT / "app" / "main.py"),
            "--server.headless", "true",
        ],
        cwd=str(PROJECT_ROOT),
    )

    time.sleep(4)
    webbrowser.open("http://localhost:8501")

    url = "http://localhost:8501"
    print(f"\n  {_ok(f'GroklyAI is running at {BOLD}{url}{RST}')}")
    print()
    print(f"  {DIM}Share this address with your team.")
    print(f"  Everyone on the same network can access it.")
    print(f"  To find your IP run:  ipconfig  (Windows) or  ifconfig  (Mac/Linux){RST}")
    print()
    print(f"  {BOLD}To stop:    Press Ctrl+C in this terminal{RST}")
    print(f"  {BOLD}To restart: python launch.py{RST}")
    print()

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print(f"\n  {_ok('GroklyAI stopped.')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GroklyAI Setup Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print_banner()

    if args.dry_run:
        print(f"  {YLW}{BOLD}DRY RUN MODE — no files will be written{RST}\n")

    # Step 1
    checks = run_system_checks()
    if not checks["python_ok"]:
        sys.exit(1)

    # Dry-run: show summary and exit without prompting
    if args.dry_run:
        print(f"\n  {DIM}System checks complete. Steps that would run on a real setup:{RST}\n")
        steps = [
            (2, "Create virtual environment and install all packages from requirements.txt"),
            (3, "Prompt for organisation name, deployment name, admin email and name"),
            (4, "Prompt for Anthropic API key (and optional Tavily key), write to .env"),
            (5, "Select knowledge sources (code, docs, Q&A) and collect their paths"),
            (6, "Run ingestion for each selected source via ingest.py"),
            (7, "Optionally launch Streamlit and open browser"),
        ]
        for n, desc in steps:
            print(f"  {BOLD}Step {n}:{RST} {desc}")
        print()
        print(_ok("Dry run complete — no files written, no packages installed."))
        return

    # Live setup — interactive
    try:
        if not setup_venv(args.dry_run):
            print(_fail("Setup failed at virtual environment step."))
            sys.exit(1)

        config  = collect_org_config(args.dry_run)
        collect_api_keys(config, args.dry_run)
        sources = select_knowledge_sources()

        if sources:
            run_ingestion(sources, args.dry_run)

        # Mark setup complete
        dep_path = PROJECT_ROOT / "grokly" / "config" / "deployment.json"
        if dep_path.exists():
            data = json.loads(dep_path.read_text(encoding="utf-8"))
            data["setup_complete"] = True
            dep_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        offer_launch(args.dry_run)

    except KeyboardInterrupt:
        print(f"\n\n  {YLW}Setup interrupted. Run 'python setup_wizard.py' to resume.{RST}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
