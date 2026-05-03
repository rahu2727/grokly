"""
grokly/scripts/admin_cli.py — Command-line tool for managing GroklyAI users.

Usage examples:

  # Add a new user with permanent role
  python grokly/scripts/admin_cli.py add-user \\
    --email john@company.com --name "John Smith" \\
    --employee-id EMP010 --department IT \\
    --role it_developer --apps erpnext \\
    --granted-by admin@company.com

  # Add temporary access
  python grokly/scripts/admin_cli.py add-temp \\
    --email jane@company.com --role manager \\
    --apps erpnext --end-date 2026-06-30 \\
    --reason "Acting manager cover" \\
    --granted-by admin@company.com

  # Deactivate a user
  python grokly/scripts/admin_cli.py deactivate \\
    --email leaver@company.com --by admin@company.com

  # Show expiring access
  python grokly/scripts/admin_cli.py expiring --days 14

  # Run daily maintenance
  python grokly/scripts/admin_cli.py daily-maintenance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from grokly.identity.admin import UserAdmin  # noqa: E402
from grokly.identity.user_manager import UserManager  # noqa: E402


def cmd_add_user(args: argparse.Namespace) -> None:
    UserAdmin().add_user(
        user_id=args.email,
        display_name=args.name,
        employee_id=args.employee_id,
        department=args.department,
        grokly_role=args.role,
        application_access=args.apps,
        granted_by=args.granted_by,
        start_date=args.start_date,
        end_date=None,
        reason=args.reason,
    )


def cmd_add_temp(args: argparse.Namespace) -> None:
    # For add-temp the user may already exist; use a placeholder display_name
    mgr  = UserManager()
    user = mgr.get_user(args.email)
    UserAdmin().add_user(
        user_id=args.email,
        display_name=user.get("display_name", args.email) if user else args.email,
        employee_id=user.get("employee_id", "") if user else "",
        department=user.get("department", "") if user else "",
        grokly_role=args.role,
        application_access=args.apps,
        granted_by=args.granted_by,
        start_date=args.start_date,
        end_date=args.end_date,
        reason=args.reason,
    )


def cmd_deactivate(args: argparse.Namespace) -> None:
    UserAdmin().deactivate_user(
        user_id=args.email,
        deactivated_by=args.by,
    )


def cmd_expiring(args: argparse.Namespace) -> None:
    UserAdmin().list_expiring(days=args.days)


def cmd_daily_maintenance(_args: argparse.Namespace) -> None:
    admin = UserAdmin()
    print("GroklyAI Daily Maintenance")
    print("=" * 40)
    print("\n1. Expiring outdated access...")
    admin.expire_temporary_access()
    print("\n2. Access expiring in next 7 days...")
    admin.list_expiring(days=7)
    print("\nMaintenance complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admin_cli",
        description="GroklyAI user management CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── add-user ──────────────────────────────────────────────────────────────
    p_add = sub.add_parser("add-user", help="Add user with permanent role")
    p_add.add_argument("--email",       required=True)
    p_add.add_argument("--name",        required=True)
    p_add.add_argument("--employee-id", required=True, dest="employee_id")
    p_add.add_argument("--department",  required=True)
    p_add.add_argument("--role",        required=True)
    p_add.add_argument("--apps",        required=True, nargs="+")
    p_add.add_argument("--granted-by",  required=True, dest="granted_by")
    p_add.add_argument("--start-date",  default=None,  dest="start_date")
    p_add.add_argument("--reason",      default="")
    p_add.set_defaults(func=cmd_add_user)

    # ── add-temp ──────────────────────────────────────────────────────────────
    p_tmp = sub.add_parser("add-temp", help="Add temporary role access")
    p_tmp.add_argument("--email",      required=True)
    p_tmp.add_argument("--role",       required=True)
    p_tmp.add_argument("--apps",       required=True, nargs="+")
    p_tmp.add_argument("--end-date",   required=True, dest="end_date")
    p_tmp.add_argument("--granted-by", required=True, dest="granted_by")
    p_tmp.add_argument("--start-date", default=None,  dest="start_date")
    p_tmp.add_argument("--reason",     default="")
    p_tmp.set_defaults(func=cmd_add_temp)

    # ── deactivate ────────────────────────────────────────────────────────────
    p_deact = sub.add_parser("deactivate", help="Deactivate a user account")
    p_deact.add_argument("--email", required=True)
    p_deact.add_argument("--by",    required=True, help="Admin performing the action")
    p_deact.set_defaults(func=cmd_deactivate)

    # ── expiring ──────────────────────────────────────────────────────────────
    p_exp = sub.add_parser("expiring", help="List access expiring soon")
    p_exp.add_argument("--days", type=int, default=7)
    p_exp.set_defaults(func=cmd_expiring)

    # ── daily-maintenance ─────────────────────────────────────────────────────
    p_maint = sub.add_parser("daily-maintenance", help="Run expiry + expiring report")
    p_maint.set_defaults(func=cmd_daily_maintenance)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
