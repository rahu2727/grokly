"""
Brand configuration for Grokly.
Change values here to rebrand — zero code changes needed.
Technical package name (grokly/) never changes.

Model configuration is in grokly/model_config.py and controlled via .env.
Do not hardcode model names here.
"""

APP_NAME        = "GroklyAI"
APP_TAGLINE     = "Your codebase, understood."
APP_DESCRIPTION = "AI knowledge assistant for enterprise systems"
APP_VERSION     = "0.1.0"
APP_AUTHOR      = "Rahul Chaudhary"
APP_GITHUB      = "github.com/rahu2727/grokly"

AGENT_NAMES = {
    "detective": "The Detective",
    "tracker":   "The Tracker",
    "counsel":   "The Counsel",
    "briefer":   "The Briefer",
}

PERSONA_LABELS = {
    "end_user":       "End User",
    "business_user":  "Business User",
    "manager":        "Manager",
    "developer":      "Developer",
    "uat_tester":     "UAT Tester",
    "doc_generator":  "Documentation",
}

BRAND_COLOUR_PRIMARY   = "#1a9e6e"
BRAND_COLOUR_SECONDARY = "#c84b11"

# Identity mode for user profiles
# Change to "prompt" for multi-user pilot
# Change to "auth" when Azure AD is integrated
# Current: "machine" — one user per machine
IDENTITY_MODE = "machine"
# Options: "machine", "prompt", "role", "auth"
# machine = Windows hostname + username (default)
# prompt  = ask user their name on first visit
# role    = track by role not individual
# auth    = Azure AD (future sprint)
