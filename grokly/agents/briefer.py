"""
grokly/agents/briefer.py — The Briefer: persona-aware answer delivery.

Responsibilities:
  - Loads the role-specific persona prompt from grokly/prompts/
  - Reformats raw_answer to match the user's role and communication style
  - Preserves all factual content while adapting tone, structure, and depth
  - Returns final_answer ready for display
"""

from pathlib import Path

import anthropic
from dotenv import load_dotenv

from grokly.pipeline.state import GroklyState

load_dotenv()

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_ROLE_PERSONA_FILE: dict[str, str] = {
    "end_user":      "end_user.txt",
    "business_user": "business_user.txt",
    "manager":       "manager.txt",
    "developer":     "developer.txt",
    "uat_tester":    "uat_tester.txt",
    "doc_generator": "doc_generator.txt",
    "system_admin":  "system_admin.txt",
    "consultant":    "consultant.txt",
}

_FALLBACK_PERSONA = (
    "You are Grokly, a helpful ERPNext assistant. "
    "Deliver the answer clearly and concisely for the user's role."
)


def briefer_node(state: GroklyState) -> dict:
    """Format raw_answer using the role-specific persona prompt."""
    raw_answer = state.get("raw_answer", "")
    role = state.get("user_role", "business_user").lower().replace(" ", "_")
    question = state["user_question"]
    sources = state.get("sources", [])

    if not raw_answer:
        return {"final_answer": "I was unable to generate an answer. Please try again."}

    persona = _load_persona(role)
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=persona,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Answer to format:\n{raw_answer}\n\n"
                    f"Sources used: {', '.join(sources) if sources else 'Grokly knowledge base'}\n\n"
                    "Reformat this answer for the user's role. "
                    "Keep all factual content intact — only adjust structure, tone, and depth."
                ),
            }
        ],
    )

    return {"final_answer": response.content[0].text.strip()}


def _load_persona(role: str) -> str:
    filename = _ROLE_PERSONA_FILE.get(role, "business_user.txt")
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return _FALLBACK_PERSONA
