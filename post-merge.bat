@echo off
REM post-merge.bat — Windows alternative to .git/hooks/post-merge
REM
REM Run manually after git pull, or wire into your workflow.
REM Requires: Python environment with grokly installed (pip install -e .)

echo [GroklyAI] Checking for knowledge base updates ...
python ingest.py --source monitor --auto-approve
