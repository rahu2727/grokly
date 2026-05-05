# GroklyAI Docker Deployment

Run GroklyAI as a Docker container — no Python installation required on the host.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- An Anthropic API key

## Quick start

**1. Create your environment file**

```bash
cp .env.example .env
```

Open `.env` and add your API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**2. Launch**

```bash
docker-compose up -d
```

Open: [http://localhost:8501](http://localhost:8501)

**3. Stop**

```bash
docker-compose down
```

## Run ingestion inside the container

```bash
# Index code
docker-compose exec groklyai python ingest.py --source code

# Index documents
docker-compose exec groklyai python ingest.py --source docs

# Index Q&A pairs
docker-compose exec groklyai python ingest.py --source forum

# Check what's been indexed
docker-compose exec groklyai python ingest.py --stats
```

## View logs

```bash
docker-compose logs -f groklyai
```

## Persistent data

The following are mounted as volumes so your data survives container restarts:

| Host path | Container path | Contents |
|-----------|---------------|----------|
| `./grokly/config` | `/app/grokly/config` | Users, roles, org config |
| `./chroma_db` | `/app/chroma_db` | Knowledge base vectors |
| `./.env` | `/app/.env` | API keys |

## Rebuild after code changes

```bash
docker-compose build
docker-compose up -d
```

## Deploy on a server (team access)

To make GroklyAI available to your whole team, run it on a server and expose port 8501:

```bash
# On the server
docker-compose up -d

# Team members access via:
# http://SERVER-IP:8501
```

For production, add a reverse proxy (nginx/Caddy) in front of port 8501 for HTTPS.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Powers AI responses |
| `TAVILY_API_KEY` | No | Enables web search fallback |
| `GROKLY_ORG` | No | Organisation name shown in UI |
| `GROKLY_DEPLOYMENT` | No | Deployment label shown in UI |
| `GROKLY_DEBUG` | No | Set to `true` for debug output |
