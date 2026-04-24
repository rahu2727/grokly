# Grokly
### Your codebase, understood.

> AI knowledge assistant for enterprise systems.
> Ask Grokly anything about your codebase.
> Get the right answer for your role.

---

## What it does

Grokly ingests your enterprise codebase (ERPNext, SAP, Frappe HRMS) into a
semantic knowledge base, then answers questions about it — tailored to who's
asking. An end user gets a plain step-by-step answer. A developer gets the
function-level technical detail. A manager gets a one-line summary.

---

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/rahu2727/grokly.git
cd grokly
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Add API keys
copy .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. Build the knowledge base
python ingest.py               # forum + docs + code (free)
python ingest.py --source commentary   # AI commentary (costs ~$7)

# 4. Test queries
python query_test.py

# 5. Launch the UI
streamlit run app/main.py
```

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐
│  Detective  │  Retrieves relevant chunks from ChromaDB
└──────┬──────┘
       │
    ┌──▼──────┐
    │ Tracker │  Evaluates context quality, re-queries if needed
    └──┬──────┘
       │
    ┌──▼──────┐
    │ Counsel │  Synthesises answer via Claude, persona-aware
    └──┬──────┘
       │
    ┌──▼──────┐
    │ Briefer │  Formats response for the UI
    └─────────┘
```

**Sprint 3:** agents are wired into a LangGraph graph. Current version
runs them sequentially as a direct pipeline.

---

## Ingestion sources

| Source      | Command                                | Cost     |
|-------------|----------------------------------------|----------|
| Forum Q&A   | `python ingest.py --source forum`      | Free     |
| Docs crawl  | `python ingest.py --source docs`       | Free     |
| Raw code    | `python ingest.py --source code`       | Free     |
| Commentary  | `python ingest.py --source commentary` | ~$7      |
| Call graph  | `python ingest.py --source call_graph` | Free     |

---

## Built with

Python · Claude AI · LangGraph · ChromaDB
Sentence Transformers · Streamlit
