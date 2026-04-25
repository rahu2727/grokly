"""
app/main.py — Grokly Streamlit application (Sprint 4B).

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import streamlit as st

from grokly.brand import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    BRAND_COLOUR_PRIMARY,
    PERSONA_LABELS,
)
from grokly.memory.session_memory import SessionMemory
from grokly.memory.user_memory import UserMemory
from grokly.pipeline.pipeline import run as pipeline_run
from grokly.store.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "session_memory" not in st.session_state:
    st.session_state.session_memory = SessionMemory(max_turns=10)

if "user_id" not in st.session_state:
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

_PERSONA_KEYS = [
    "end_user",
    "business_user",
    "manager",
    "developer",
    "uat_tester",
    "doc_generator",
]
_PERSONA_OPTIONS = [PERSONA_LABELS[k] for k in _PERSONA_KEYS]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## {APP_NAME}")
    st.markdown(f"*{APP_TAGLINE}*")
    st.divider()

    # User identity
    user_id_input = st.text_input(
        "Your name / ID",
        value=st.session_state.user_id,
        placeholder="e.g. alice",
        help="Optional: enables personalised memory across sessions.",
    )
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input

    # User memory stats
    if st.session_state.user_id:
        stats = user_memory.get_stats(st.session_state.user_id)
        st.caption(
            f"Questions asked: **{stats['question_count']}** · "
            f"Preferred role: `{stats['preferred_role']}`"
        )
        if stats["topics_explored"]:
            with st.expander("Recent topics", expanded=False):
                for t in stats["topics_explored"][:5]:
                    st.caption(f"- {t}")

    st.divider()

    # Knowledge base stats
    chunk_count = store.count()
    st.metric("Knowledge chunks", f"{chunk_count:,}")

    db_stats = store.stats()
    if db_stats["by_source"]:
        st.markdown("**By source**")
        for src, cnt in sorted(db_stats["by_source"].items()):
            st.markdown(f"- `{src}`: {cnt:,}")
    else:
        st.info("Knowledge base is empty. Run `python ingest.py` first.")

    st.divider()

    # Session memory stats
    mem: SessionMemory = st.session_state.session_memory
    turns_in_window = len(mem.turns)
    has_summary = bool(mem.context_summary)
    st.caption(
        f"Session memory: **{turns_in_window}** turn(s) in window"
        + (" + compressed summary" if has_summary else "")
    )

    col_clear, col_mem = st.columns(2)
    with col_clear:
        if st.session_state.history:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.history = []
                st.session_state.session_memory = SessionMemory(max_turns=10)
                st.rerun()
    with col_mem:
        if turns_in_window > 0:
            if st.button("Clear memory", use_container_width=True):
                st.session_state.session_memory = SessionMemory(max_turns=10)
                st.rerun()

    st.divider()
    st.caption(f"v{APP_VERSION} · Sprint 4B")
    st.caption("LangGraph pipeline · session + user memory")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title(APP_NAME)
st.caption(APP_TAGLINE)
st.divider()

# Role selector — pre-fill from user memory if we have a returning user
_default_role_idx = 0
if st.session_state.user_id:
    preferred = user_memory.get_preferred_role(st.session_state.user_id)
    if preferred in _PERSONA_KEYS:
        _default_role_idx = _PERSONA_KEYS.index(preferred)

col1, col2 = st.columns([2, 3])
with col1:
    selected_label = st.selectbox(
        "I am a...",
        options=_PERSONA_OPTIONS,
        index=_default_role_idx,
        help=(
            "Select your role so Grokly tailors the answer to you.\n\n"
            "Documentation mode generates export-ready structured output."
        ),
    )
selected_persona = _PERSONA_KEYS[_PERSONA_OPTIONS.index(selected_label)]
is_doc_mode = selected_persona == "doc_generator"

if is_doc_mode:
    st.info(
        "**Documentation mode** — answers are formatted as structured documentation "
        "ready to copy into your knowledge base or SOP. Use the Export button to download."
    )

# ---------------------------------------------------------------------------
# Conversation history display
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


for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**[{entry['persona_label']}]** {entry['query']}")

    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        _render_tool_badges(entry.get("tools_used", []))
        _render_details(entry)
        _render_export(entry)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

query = st.chat_input(
    placeholder="e.g. How do I submit a leave application?",
)

if query:
    mem: SessionMemory = st.session_state.session_memory

    # Resolve pronouns / vague references against session history
    resolved_query = mem.resolve_references(query)

    with st.chat_message("user"):
        if resolved_query != query:
            st.markdown(f"**[{selected_label}]** {query}")
            st.caption(f"*Interpreted as: {resolved_query}*")
        else:
            st.markdown(f"**[{selected_label}]** {query}")

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking as {selected_label}..."):
            session_ctx = mem.get_context()
            result = pipeline_run(
                resolved_query,
                role=selected_persona,
                session_context=session_ctx,
            )

        answer = result["answer"]
        st.markdown(answer)

        tools_used = result.get("tools_used", [])
        _render_tool_badges(tools_used)

        entry: dict = {
            "query":         query,
            "resolved_query": resolved_query,
            "persona_key":   selected_persona,
            "persona_label": selected_label,
            "answer":        answer,
            "tools_used":    tools_used,
            "confidence":    result.get("confidence",      0.0),
            "iterations":    result.get("iteration_count", 0),
            "quality_score": result.get("quality_score",  0.0),
            "sources":       result.get("sources",         []),
        }

        _render_details(entry)
        _render_export(entry)

    # Update session memory
    mem.add_turn(
        question=resolved_query,
        answer=answer,
        role=selected_persona,
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )

    # Update long-term user memory
    if st.session_state.user_id:
        user_memory.record_question(
            st.session_state.user_id,
            query,
            selected_persona,
        )

    st.session_state.history.append(entry)
