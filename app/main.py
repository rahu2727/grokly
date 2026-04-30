"""
app/main.py — GroklyAI Streamlit application (Sprint 4C).

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from grokly.brand import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    IDENTITY_MODE,
    PERSONA_LABELS,
)
from grokly.memory.session_memory import SessionMemory
from grokly.memory.user_memory import UserMemory
from grokly.model_config import AGENT_MODEL_KEYS, get_model, print_model_summary
from grokly.pipeline.pipeline import run as pipeline_run
from grokly.store.chroma_store import ChromaStore

_DEBUG = os.getenv("GROKLY_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "model_summary_shown" not in st.session_state:
    print_model_summary()
    st.session_state.model_summary_shown = True

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "session_memory" not in st.session_state:
    st.session_state.session_memory = SessionMemory(max_turns=10)

if "user_id" not in st.session_state:
    if IDENTITY_MODE == "machine":
        st.session_state.user_id: str = UserMemory.get_user_id()
    else:
        st.session_state.user_id: str = ""

# ---------------------------------------------------------------------------
# Shared store and user memory (cached per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_store() -> ChromaStore:
    return ChromaStore()


@st.cache_resource
def _get_user_memory() -> UserMemory:
    return UserMemory(store=_get_store())


store = _get_store()
user_memory = _get_user_memory()

# ---------------------------------------------------------------------------
# Persona config
# ---------------------------------------------------------------------------

_PERSONA_KEYS = list(PERSONA_LABELS.keys())

_ROLE_DESCRIPTIONS = {
    "end_user":      "Step by step guidance",
    "business_user": "Process and policy detail",
    "manager":       "Approval rules and governance",
    "developer":     "Technical depth and code refs",
    "uat_tester":    "Edge cases and test scenarios",
    "doc_generator": "Documentation generation",
}

# Pre-fill role from user memory on first visit
if "selected_role" not in st.session_state:
    uid = st.session_state.user_id
    if uid and IDENTITY_MODE != "role":
        preferred = user_memory.get_preferred_role(uid)
        if preferred in _PERSONA_KEYS:
            st.session_state.selected_role = preferred

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    st.divider()

    # 3. Role selector — always visible
    st.subheader("Your role")
    selected_persona: str = st.selectbox(
        "I am a...",
        options=_PERSONA_KEYS,
        format_func=lambda x: PERSONA_LABELS[x],
        key="selected_role",
        label_visibility="collapsed",
    )
    st.caption(_ROLE_DESCRIPTIONS.get(selected_persona, ""))

    if selected_persona == "doc_generator":
        st.info(
            "Answers formatted as structured documentation. "
            "Use Export to download."
        )

    st.divider()

    # 5. Knowledge base stats — high-level only, no internal chunk types
    st.subheader("Knowledge base")
    chunk_count = store.count()
    st.metric("Chunks indexed", f"{chunk_count:,}")

    by_src = store.stats().get("by_source", {})
    if by_src:
        docs_count = by_src.get("docs", 0)
        forum_count = by_src.get("forum", 0)
        code_count = by_src.get("code_commentary", 0)
        st.caption(f"📄 Documentation  ·  {docs_count:,}")
        st.caption(f"💬 Q&A pairs  ·  {forum_count:,}")
        st.caption(f"🧠 Code understood  ·  {code_count:,}")
    else:
        st.warning("Knowledge base empty. Run `python ingest.py` first.")

    st.divider()

    # 6. Knowledge status — change monitor checkpoints
    _STATE_FILE = Path(__file__).parent.parent / "grokly" / "agents" / "monitor_state.json"
    if _STATE_FILE.exists():
        try:
            _monitor_state: dict = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            if _monitor_state:
                st.subheader("Knowledge status")
                for _repo, _info in _monitor_state.items():
                    _commit = _info.get("commit_hash", "?")[:8]
                    _ts     = _info.get("updated_at", "")[:10]
                    st.caption(f"📦 {_repo}  ·  `{_commit}`  ·  {_ts}")
                st.divider()
        except Exception:
            pass

    # 7. Conversation memory stats
    st.subheader("This conversation")
    mem: SessionMemory = st.session_state.session_memory
    turns = len(mem.turns)
    last_topic = mem.get_last_topic() if turns else ""

    st.caption(f"Questions: **{turns}**" + (f"  ·  Last: *{last_topic}*" if last_topic else ""))
    if mem.context_summary:
        st.caption("+ compressed earlier context")

    if st.button("New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.session_memory = SessionMemory(max_turns=10)
        st.rerun()

    st.divider()

    # 9. User profile expander at bottom
    uid = st.session_state.user_id
    if uid and IDENTITY_MODE != "role":
        with st.expander("Your profile", expanded=False):
            stats = user_memory.get_stats(uid)
            st.caption(f"ID: `{uid}`")
            st.caption(f"Questions asked: **{stats['question_count']}**")
            st.caption(f"Preferred role: `{stats['preferred_role']}`")
            if stats.get("topics_explored"):
                st.markdown("**Recent topics**")
                for t in stats["topics_explored"][:5]:
                    st.caption(f"- {t}")
    elif IDENTITY_MODE == "prompt":
        user_id_input = st.text_input(
            "Your name / ID",
            value=uid,
            placeholder="e.g. alice",
            help="Enables personalised memory across sessions.",
        )
        if user_id_input != uid:
            st.session_state.user_id = user_id_input

    st.caption(f"v{APP_VERSION}")

    with st.expander("⚙️ Model config", expanded=False):
        for _agent in ["counsel", "briefer", "tracker", "memory", "proactive"]:
            _m = get_model(_agent)
            _short = (
                _m.replace("claude-", "")
                  .replace("-20250514", "")
                  .replace("-20251001", "")
                  .replace("-4-6", " 4.6")
            )
            st.caption(f"{_agent}: `{_short}`")

# ---------------------------------------------------------------------------
# Derived vars from sidebar state
# ---------------------------------------------------------------------------

selected_label: str = PERSONA_LABELS[selected_persona]

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_tool_badges(tools: list[str]) -> None:
    if not tools:
        return
    labels = []
    for t in tools:
        try:
            label = t.split(":")[1].split("(")[0]
        except IndexError:
            label = t
        labels.append(f"`{label}`")
    st.caption("Tools called: " + " · ".join(labels))


def _render_details(entry: dict) -> None:
    confidence = entry.get("confidence",    0.0)
    iterations = entry.get("iterations",    0)
    quality    = entry.get("quality_score", 0.0)
    sources    = entry.get("sources",       [])

    with st.expander("Details", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{confidence:.0%}")
        c2.metric("Iterations", iterations)
        c3.metric("Quality",    f"{quality:.1f} / 5.0")
        if sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in sorted(sources)))


def _render_proactive(insights: dict, role: str, key_prefix: str = "") -> None:
    print(f"[UI] proactive_insights type: {type(insights)}")
    print(f"[UI] has_insights: {insights.get('has_insights')}")
    print(f"[UI] suggestions: {len(insights.get('related', {}).get('suggestions', []))}")

    if _DEBUG:
        st.caption(
            f"Debug: has_insights={insights.get('has_insights')}, "
            f"suggestions={len(insights.get('related', {}).get('suggestions', []))}"
        )

    if not insights.get("has_insights"):
        return

    st.divider()
    st.markdown("### 💡 You might also want to know")

    # Gap alert
    gap = insights.get("gap_alert", {})
    if gap and gap.get("triggered"):
        st.warning(f"⚠️ {gap.get('message', '')}")

    # Related suggestions
    suggestions = insights.get("related", {}).get("suggestions", [])
    if suggestions:
        for idx, s in enumerate(suggestions):
            label = s.get("label", "Related")
            if role == "developer":
                display_name = s.get("function") or s.get("topic") or s.get("scenario") or "?"
            else:
                display_name = s.get("topic") or s.get("scenario") or s.get("function") or "?"
            desc   = s.get("description", "")
            prompt = s.get("prompt", "")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{label}:** {display_name}")
                if desc:
                    st.caption(desc[:100])
            with col2:
                if prompt:
                    btn_key = f"proactive_{key_prefix}{idx}_{hash(prompt) % 100_000}"
                    if st.button("Ask this →", key=btn_key, use_container_width=True):
                        st.session_state.pending_question = prompt
                        st.rerun()

    # Staleness warning
    stale = insights.get("staleness", {})
    if stale and stale.get("triggered"):
        st.info(
            f"ℹ️ Commentary is **{stale.get('days_old', '?')} days old**. "
            f"Run `{stale.get('action', '')}` to refresh."
        )


def _render_export(entry: dict) -> None:
    if entry.get("persona_key") != "doc_generator":
        return
    query_slug = entry["query"][:40].replace(" ", "_").replace("/", "-")
    filename   = f"grokly_doc_{query_slug}.txt"
    content    = f"Question: {entry['query']}\n\n{entry['answer']}"
    st.download_button(
        label="Export as .txt",
        data=content,
        file_name=filename,
        mime="text/plain",
        key=f"export_{hash(entry['query'] + entry['answer'][:20])}",
    )


# ---------------------------------------------------------------------------
# Main area — title + conversation history
# ---------------------------------------------------------------------------

st.title(APP_NAME)
st.caption(APP_TAGLINE)
st.divider()

for _i, entry in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(f"**[{entry['persona_label']}]** {entry['query']}")

    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        _render_tool_badges(entry.get("tools_used", []))
        _render_details(entry)
        _render_export(entry)
        _render_proactive(
            entry.get("proactive_insights", {}),
            entry["persona_key"],
            key_prefix=f"h{_i}_",
        )

# ---------------------------------------------------------------------------
# Chat input — picks up proactive suggestion clicks via auto_question
# ---------------------------------------------------------------------------

query = st.chat_input(placeholder="e.g. How do I submit a leave application?")

# Pick up question set by "Ask this →" proactive buttons
if not query and "pending_question" in st.session_state:
    query = st.session_state.pending_question
    del st.session_state["pending_question"]

if query:
    mem = st.session_state.session_memory
    uid = st.session_state.user_id or None

    with st.chat_message("user"):
        st.markdown(f"**[{selected_label}]** {query}")

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking as {selected_label}..."):
            result = pipeline_run(
                query,
                role=selected_persona,
                session_memory=mem,
                user_memory=user_memory if uid else None,
                user_id=uid,
            )

        answer = result["answer"]
        resolved = result.get("resolved_question", query)
        st.markdown(answer)

        if resolved != query:
            st.caption(f"*Interpreted as: {resolved}*")

        tools_used = result.get("tools_used", [])
        _render_tool_badges(tools_used)

        entry: dict = {
            "query":              query,
            "resolved_query":     resolved,
            "persona_key":        selected_persona,
            "persona_label":      selected_label,
            "answer":             answer,
            "tools_used":         tools_used,
            "confidence":         result.get("confidence",       0.0),
            "iterations":         result.get("iteration_count",  0),
            "quality_score":      result.get("quality_score",   0.0),
            "sources":            result.get("sources",          []),
            "proactive_insights": result.get("proactive_insights", {}),
        }

        _render_details(entry)
        _render_export(entry)
        _render_proactive(entry["proactive_insights"], selected_persona, key_prefix="new_")

    st.session_state.history.append(entry)
