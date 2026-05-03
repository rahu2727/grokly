"""
grokly/scripts/daily_maintenance.py — Daily GroklyAI maintenance job.

Run via Windows Task Scheduler:

  schtasks /create /tn "GroklyAI Maintenance" ^
    /tr "C:\\path\\to\\venv\\Scripts\\python.exe C:\\path\\to\\daily_maintenance.py" ^
    /sc daily /st 06:00

Or via cron (Linux/macOS):
  0 6 * * * /path/to/venv/bin/python /path/to/daily_maintenance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from grokly.identity.admin import UserAdmin  # noqa: E402


def main() -> None:
    admin = UserAdmin()

    print("GroklyAI Daily Maintenance")
    print("=" * 40)

    print("\n1. Expiring outdated access...")
    expired = admin.expire_temporary_access()

    print("\n2. Access expiring in next 7 days...")
    admin.list_expiring(days=7)

    print(f"\nMaintenance complete. {expired} assignment(s) expired today.")


if __name__ == "__main__":
    main()
