"""
app/main.py — Grokly Streamlit application (Sprint 3).

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
    # Each entry: query, persona_key, persona_label, answer,
    #             tools_used, confidence, iterations, quality_score
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Shared store (cached so it is created once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_store() -> ChromaStore:
    return ChromaStore()


store = _get_store()

# ---------------------------------------------------------------------------
# Persona config — 5 core personas + Documentation as a named mode
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

    chunk_count = store.count()
    st.metric("Knowledge chunks", f"{chunk_count:,}")

    stats = store.stats()
    if stats["by_source"]:
        st.markdown("**By source**")
        for src, cnt in sorted(stats["by_source"].items()):
            st.markdown(f"- `{src}`: {cnt:,}")
    else:
        st.info("Knowledge base is empty. Run `python ingest.py` first.")

    st.divider()

    if st.session_state.history:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption(f"v{APP_VERSION} · Sprint 3")
    st.caption("LangGraph pipeline · 4-agent ReAct loop")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title(APP_NAME)
st.caption(APP_TAGLINE)
st.divider()

# Role selector
col1, col2 = st.columns([2, 3])
with col1:
    selected_label = st.selectbox(
        "I am a...",
        options=_PERSONA_OPTIONS,
        index=0,
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
    """Show compact tool call badges below an answer."""
    if not tools:
        return
    # Extract readable labels: "tracker:search_commentary({...})" → "search_commentary"
    labels = []
    for t in tools:
        try:
            label = t.split(":")[1].split("(")[0]
        except IndexError:
            label = t
        labels.append(f"`{label}`")
    st.caption("Tools called: " + " · ".join(labels))


def _render_details(entry: dict) -> None:
    """Expandable confidence / iteration details section."""
    confidence   = entry.get("confidence",      0.0)
    iterations   = entry.get("iterations",      0)
    quality      = entry.get("quality_score",   0.0)
    sources      = entry.get("sources",         [])

    with st.expander("Details", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence",  f"{confidence:.0%}")
        c2.metric("Iterations",  iterations)
        c3.metric("Quality",     f"{quality:.1f} / 5.0")
        if sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in sorted(sources)))


def _render_export(entry: dict) -> None:
    """Download button — only shown for doc_generator persona."""
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
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(f"**[{selected_label}]** {query}")

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking as {selected_label}..."):
            result = pipeline_run(query, role=selected_persona)

        answer = result["answer"]
        st.markdown(answer)

        tools_used = result.get("tools_used", [])
        _render_tool_badges(tools_used)

        entry: dict = {
            "query":        query,
            "persona_key":  selected_persona,
            "persona_label": selected_label,
            "answer":       answer,
            "tools_used":   tools_used,
            "confidence":   result.get("confidence",      0.0),
            "iterations":   result.get("iteration_count", 0),
            "quality_score": result.get("quality_score",  0.0),
            "sources":      result.get("sources",         []),
        }

        _render_details(entry)
        _render_export(entry)

    st.session_state.history.append(entry)
