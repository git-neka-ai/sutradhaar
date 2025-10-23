# orion: Extracted environment-driven configuration constants from editor.py to a dedicated module to centralize settings and allow other modules to import them without circular dependencies.

import os

# OpenAI env (Chat Completions / Responses)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-5")  # OpenAI model id

# Output token budget
MAX_COMPLETION_TOKENS = int(os.environ.get("ORION_MAX_COMPLETION_TOKENS", "48192"))

# Line cap for post-apply check
LINE_CAP = int(os.environ.get("ORION_LINE_CAP", "1000"))

# Conversation history cap on load
CONV_CAP_TURNS = int(os.environ.get("ORION_CONV_CAP_TURNS", "200"))

# Summaries max bytes
SUMMARY_MAX_BYTES = int(os.environ.get("ORION_SUMMARY_MAX_BYTES", str(2_000_000)))  # 2 MB default

# External dependency projects (flat directory, no subfolders)
# This is the root folder containing Project Descriptions (PDs) as files,
# and a .orion/ subfolder containing Project Orion Summaries (POS) named <filename>.json
ORION_EXTERNAL_DIR = os.environ.get("ORION_EXTERNAL_DIR", "").strip()

# Optional: TTL in seconds to force POS regeneration even if hash matches (omit/0 to disable)
ORION_DEP_TTL_SEC = int(os.environ.get("ORION_DEP_TTL_SEC", "0") or "0")
