# GroklyAI — Quick Start

## What you need

- Windows 10/11 or Mac
- Python 3.11 or newer — download at [python.org](https://www.python.org/downloads/)
- An Anthropic API key — free at [console.anthropic.com](https://console.anthropic.com)
- Internet connection

## Setup (one time, ~15 minutes)

**1. Download GroklyAI**

Unzip the GroklyAI folder to somewhere convenient, e.g. `C:\GroklyAI`.

**2. Open a terminal in the GroklyAI folder**

- Windows: Right-click the folder → *Open in Terminal*
- Mac: Right-click the folder → *New Terminal at Folder*

**3. Run the setup wizard**

```
python setup_wizard.py
```

Follow the prompts. The wizard will:
- Install all required packages automatically
- Ask for your organisation name and admin email
- Ask for your Anthropic API key
- Let you add knowledge sources (documents, code, Q&A files)
- Launch GroklyAI in your browser when ready

## Start GroklyAI after setup

```
python launch.py
```

GroklyAI opens in your browser automatically.

## Share with your team

Once GroklyAI is running, share this address with anyone on the same network:

```
http://YOUR-IP-ADDRESS:8501
```

**Find your IP address:**

- Windows: open a terminal and run `ipconfig`, look for *IPv4 Address*
- Mac/Linux: run `ifconfig` and look for *inet* under your network adapter

## Add more knowledge sources

```
python ingest.py --source docs
python ingest.py --source code
python ingest.py --source forum
```

Run `python ingest.py --help` to see all available sources.

## Update knowledge after code changes

```
python ingest.py --source monitor
```

This detects changed files and re-ingests only what has changed.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not found | Install Python 3.11+ from python.org and tick *Add to PATH* during install |
| API key error | Check your key starts with `sk-ant-` and has no extra spaces |
| Browser does not open | Navigate to http://localhost:8501 manually |
| "Knowledge base empty" warning | Run `python ingest.py` to build the knowledge base |
| Port 8501 already in use | Stop the existing process or change port with `--server.port 8502` |

## File structure

```
GroklyAI/
├── setup_wizard.py     ← Run this first
├── launch.py           ← Run this to start after setup
├── ingest.py           ← Add knowledge sources
├── app/                ← Streamlit UI
├── grokly/             ← Core application code
│   └── config/         ← Your configuration files
│       ├── deployment.json     ← Organisation settings
│       ├── users_master.json   ← User accounts
│       ├── applications.json   ← Application registry
│       └── role_permissions.json ← Role access rules
└── chroma_db/          ← Knowledge base (created on first ingest)
```
