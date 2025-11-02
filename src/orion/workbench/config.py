# orion: Centralize configuration constants; removed ORION_EXTERNAL_DIR (external PD directory is now provided exclusively via CLI --external-dir/-e). Other env-based settings remain unchanged.

import os

# OpenAI env (Chat Completions / Responses)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-5")  # OpenAI model id
# orion: Add AI_IMAGE_SUMMARY_MODEL to configure a vision-capable model for image summarization (defaults to gpt-4o).
AI_IMAGE_SUMMARY_MODEL = os.environ.get("AI_IMAGE_SUMMARY_MODEL", "gpt-5")

# Output token budget
MAX_COMPLETION_TOKENS = int(os.environ.get("ORION_MAX_COMPLETION_TOKENS", "48192"))

# Line cap for post-apply check
LINE_CAP = int(os.environ.get("ORION_LINE_CAP", "1000"))

# Conversation history cap on load
CONV_CAP_TURNS = int(os.environ.get("ORION_CONV_CAP_TURNS", "200"))

# Summaries max bytes
SUMMARY_MAX_BYTES = int(os.environ.get("ORION_SUMMARY_MAX_BYTES", str(2_000_000)))  # 2 MB default

# orion: Keep POS TTL as an environment-driven knob; this remains independent of how the external directory is provided.
# Optional: TTL in seconds to force POS regeneration even if hash matches (omit/0 to disable)
ORION_DEP_TTL_SEC = int(os.environ.get("ORION_DEP_TTL_SEC", "0") or "0")
