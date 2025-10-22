# orion: Moved user I/O Context and repository Storage (metadata and history) out of editor.py to encapsulate persistence and console I/O concerns.

import pathlib
import shutil
from typing import Any, Dict, List, Optional

from .config import CONV_CAP_TURNS
from .fs import now_ts, short_id, read_json, write_json, append_jsonl, read_jsonl


class Context:
    def __init__(self) -> None:
        pass

    def send_to_user(self, message: str) -> None:
        print(message)

    def log(self, message: str) -> None:
        print(f"[LOG] {message}")

    def error_message(self, message: str) -> None:
        import sys
        print(f"Error: {message}", file=sys.stderr)


class Storage:
    def __init__(self, repo_root: pathlib.Path) -> None:
        self.repo_root = repo_root
        self.metadata_file = repo_root / "orion-metadata.json"
        self.conv_file = repo_root / "orion-conversation.jsonl"

    def default_metadata(self) -> Dict[str, Any]:
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

    def load_metadata(self) -> Dict[str, Any]:
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

    def save_metadata(self, md: Dict[str, Any]) -> None:
        write_json(self.metadata_file, md)

    def load_history(self) -> List[Dict[str, Any]]:
        hist = read_jsonl(self.conv_file)
        if len(hist) > CONV_CAP_TURNS:
            hist = hist[-CONV_CAP_TURNS:]
        return hist

    def append_history(self, role: str, content: str, extra: Optional[Dict[str, Any]] = None) -> None:
        entry = {"ts": now_ts(), "role": role, "content": content}
        if extra:
            entry.update(extra)
        append_jsonl(self.conv_file, entry)

    def append_raw_message(self, msg: Dict[str, Any]) -> None:
        """
        Append a raw chat message to the conversation log. Message must include 'role'.
        We preserve additional fields (tool_calls, tool_call_id, etc.) for replay.
        """
        if "role" not in msg:
            raise ValueError("append_raw_message requires a message with a 'role' field")
        entry = dict(msg)
        entry.setdefault("ts", now_ts())
        append_jsonl(self.conv_file, entry)

    def clear_history(self) -> None:
        if self.conv_file.exists():
            backup = self.conv_file.with_name(f"{self.conv_file.stem}-{int(now_ts())}.bak.jsonl")
            shutil.move(str(self.conv_file), str(backup))
