"""Verification script for model configuration refactor."""
import os
from grokly.model_config import get_model, get_agent_config, print_model_summary

print_model_summary()

agents = ["commentary", "counsel", "tracker", "briefer", "memory", "proactive", "monitor"]
print("Verification:")
for agent in agents:
    cfg = get_agent_config(agent)
    print(f"  {agent}: {cfg['model']} (max_tokens={cfg['max_tokens']})")

# Fallback test: blank override → default
os.environ["GROKLY_MODEL_COUNSEL"] = ""
model   = get_model("counsel")
default = os.getenv("GROKLY_DEFAULT_MODEL", "claude-sonnet-4-6")
print(f"\nFallback test: {model}")
assert model == default, f"{model!r} != {default!r}"
print("Fallback: PASS")
