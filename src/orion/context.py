# orion: Moved user I/O Context and repository Storage (metadata and history) out of editor.py to encapsulate persistence and console I/O concerns. Added docstrings and inline comments documenting metadata defaults/merging and history rotation.

import pathlib
import shutil
from typing import Any, Dict, List, Optional

from .config import CONV_CAP_TURNS
from .fs import now_ts, short_id, read_json, write_json, append_jsonl, read_jsonl


# orion: Add class-level docstring and method docstrings to clarify responsibilities of Context (console I/O and logging only).
class Context:
    """
    Thin wrapper around console I/O and logging used by Orion.

    This abstraction exists to decouple direct stdout/stderr usage from business
    logic, and to enable easier testing/substitution in the future.
    """

    # orion: Add trivial docstring for completeness and future extension.
    def __init__(self) -> None:
        """Initialize a default console-based context."""
        pass

    # orion: Clarify side-effect: prints to stdout.
    def send_to_user(self, message: str) -> None:
        """Send a user-facing message to stdout."""
        print(message)

    # orion: Clarify formatting and intended use for lightweight tracing.
    def log(self, message: str) -> None:
        """Emit a lightweight log line to stdout, prefixed for readability."""
        print(f"[LOG] {message}")

    # orion: Clarify that this targets stderr to distinguish from user messages.
    def error_message(self, message: str) -> None:
        """Print an error message to stderr."""
        import sys
        print(f"Error: {message}", file=sys.stderr)


# orion: Add class-level docstring and annotate behaviors like history trimming and metadata defaults.
class Storage:
    """
    Persistence wrapper for Orion conversation history and repository metadata.

    All paths are resolved relative to the repo_root. History is stored as JSONL
    for append-only writes; metadata is a compact JSON file.
    """

    def __init__(self, repo_root: pathlib.Path) -> None:
        """Initialize storage with the repository root path."""
        self.repo_root = repo_root
        self.metadata_file = repo_root / ".orion" / "orion-metadata.json"
        self.conv_file = repo_root / ".orion" / "orion-conversation.jsonl"
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # orion: Document the default structure and meaning of keys.
    def default_metadata(self) -> Dict[str, Any]:
        """Return a fresh default metadata dictionary for new repositories."""
        return {
            "plan_state": {
                "plan_id": short_id("plan"),
                "commit_log": []
            },
            "pending_changes": [],
            "batches_since_last_consolidation": 0,
            # summaries support
            "path_to_digest": {}
        }

    # orion: Clarify how legacy keys are culled and how missing keys are populated.
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from disk, populating missing keys with sensible defaults.

        Returns:
            The metadata dictionary ready for in-memory use.
        """
        md = read_json(self.metadata_file, self.default_metadata())
        if "plan_state" not in md:
            md["plan_state"] = {"plan_id": short_id("plan"), "commit_log": []}
        if "pending_changes" not in md:
            md["pending_changes"] = []
        if "batches_since_last_consolidation" not in md:
            md["batches_since_last_consolidation"] = 0
        if "path_to_digest" not in md:
            md["path_to_digest"] = {}
        # purge any legacy keys if present
        md.pop("summaries_by_digest", None)
        return md

    # orion: Document that writes are atomic via fs.write_json helper.
    def save_metadata(self, md: Dict[str, Any]) -> None:
        """Persist metadata atomically to disk."""
        write_json(self.metadata_file, md)

    # orion: Add docstring and comment on trimming to CONV_CAP_TURNS for bounded history size.
    def load_history(self) -> List[Dict[str, Any]]:
        """Load the conversation history (JSONL) and trim to CONV_CAP_TURNS most recent entries."""
        hist = read_jsonl(self.conv_file)
        if len(hist) > CONV_CAP_TURNS:
            # orion: Keep only the most recent N items to cap memory and token usage.
            hist = hist[-CONV_CAP_TURNS:]
        return hist

    # orion: Document convenience wrapper for appending a common shape (role/content plus timestamp). 
    def append_history(self, role: str, content: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Append a simple role/content entry to the JSONL conversation history."""
        entry = {"ts": now_ts(), "role": role, "content": content}
        if extra:
            entry.update(extra)
        append_jsonl(self.conv_file, entry)

    # orion: Existing docstring retained; clarify that this preserves raw tool_call structures for exact replay.
    def append_raw_message(self, msg: Dict[str, Any]) -> None:
        """
        Append a raw chat message to the conversation log. Message either be an input with a role and content or a function_call or function_call_output.
        
        """
        _type = msg.get("type")
        if _type not in ("message", "function_call", "function_call_output"):
            raise ValueError("append_raw_message requires a message of type 'message', 'function_call', or 'function_call_output'")
        entry = dict(msg)
        entry.setdefault("ts", now_ts())
        append_jsonl(self.conv_file, entry)

    # orion: Add docstring explaining rotation behavior and rationale (keeps a backup for debugging/audit).
    def clear_history(self) -> None:
        """Rotate the current conversation JSONL to a timestamped .bak.jsonl file if present."""
        if self.conv_file.exists():
            backup = self.conv_file.with_name(f"{self.conv_file.stem}-{int(now_ts())}.bak.jsonl")
            shutil.move(str(self.conv_file), str(backup))
