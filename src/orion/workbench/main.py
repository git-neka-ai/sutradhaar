# orion: Replace bootstrap flow with a persistent, authoritative system_state message and ensure it is included at the front of every model call. Add builders/helpers to create, ensure, and select the latest system_state, and update handle_user_input/apply to use it.

import pathlib
import sys

# orion: Ported the core Orion class, tool definitions, command handlers, bootstrap/summaries flow, and change-spec validators from editor.py into a focused module. Adjusted imports to use the new modular structure. Externalized conversation/apply system prompts via orion.prompts.get_prompt. Added docstrings and inline comments where flow or validation is non-obvious.

import json
import pathlib
from typing import Any, Dict, List, Optional

from .config import LINE_CAP
from .context import Context, Storage
from .client import ChatCompletionsClient
from .external import ext_dir_valid, list_project_descriptions
from .fs import (
    list_all_nonignored_files,
    normalize_path,
    read_file,
    write_file,
    count_lines,
    colocated_summary_path,
    now_ts,
    short_id,
)
from .summarizers import ensure_all_pos, summarize_file
# orion: Switch to reflective tool utilities; individual tool functions are no longer imported or wired here.
from .tools import discover_tools, list_tool_names, run_tool
from .prompts import get_prompt
from .settings import load_settings

# orion: Import centralized Pydantic models and validation utilities; this replaces ad-hoc schema dicts.
from pydantic import ValidationError, TypeAdapter
from .models import (
    ConversationResponse,
    ApplyResponse,
    ChangeSpec,
    ChangeItem,
    InfoSummary,
)


# -----------------------------
# Strict schemas and change-spec helpers
# -----------------------------

# orion: Add helper docstrings to clarify expected shapes used by the model and apply pipeline.

def make_change_spec(id_str: str, title: str, description: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a change spec object with the required keys for Orion's apply flow."""
    return {
        "id": id_str,
        "title": title,
        "description": description,
        "items": items,
    }


# orion: Document allowed change_type values for model guidance consistency.

def make_change_item(path: str, change_type: str, summary_of_change: str) -> Dict[str, Any]:
    """Build a single change item with normalized path and a concise summary."""
    if change_type not in ["modify", "create", "delete", "move", "rename"]:
        raise ValueError("Invalid change_type")
    return {"path": normalize_path(path), "change_type": change_type, "summary_of_change": summary_of_change}


# orion: Simplify validators by delegating to Pydantic models; retained as thin wrappers for compatibility.

def validate_change_specs(changes: Any) -> List[Dict[str, Any]]:
    """Validate a list of change specs using Pydantic and return normalized dicts."""
    try:
        parsed: List[ChangeSpec] = TypeAdapter(List[ChangeSpec]).validate_python(changes)
    except ValidationError:
        return []
    # Ensure enum values are dumped as strings and paths are normalized via model validators.
    return [c.model_dump() for c in parsed]


def validate_apply_response(resp: Dict[str, Any]) -> (bool, str):
    """Validate ApplyResponse shape using Pydantic; returns (ok, error)."""
    try:
        ApplyResponse.model_validate(resp)
        return True, ""
    except ValidationError as e:
        return False, str(e)


# -----------------------------
# Tools exposed to the model (definitions)
# -----------------------------

# orion: Provide external dependency tool schemas; internal tools are discovered via reflection and get a synthetic reason_for_call injected solely in their model schema.

def _external_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "list_project_descriptions",
            "description": "List dependency Project Descriptions (filenames) from the external directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    # orion: Synthetic reason_for_call is exposed to the model but stripped before invocation.
                    "reason_for_call": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_project_orion_summary",
            "description": "Return the Project Orion Summary (POS) for a given PD filename; regenerates if stale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    # orion: Synthetic reason_for_call for consistency with internal tools.
                    "reason_for_call": {"type": "string"},
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    ]


# -----------------------------
# Orion main class
# -----------------------------


class Orion:
    """
    High-level orchestrator for interactive code changes and repository management.

    Responsibilities include:
      - Managing conversation/history and metadata
      - Wiring tool registry for function-calling
      - Refreshing/validating summaries (local files and external PDs)
      - Building requests to the model and applying file changes safely
    """

    def __init__(self, repo_root: pathlib.Path, external_dir: Optional[str] = None) -> None:
        """Initialize Orion with a repository root and supporting services.

        Args:
            repo_root: Absolute or relative path to the repository root.
            external_dir: Optional path to the external Project Descriptions directory (flat). When None or invalid,
                          external dependency features are disabled. Provided via CLI --external-dir/-e.
        """
        self.repo_root = repo_root.resolve()
        self.storage = Storage(self.repo_root)
        self.md = self.storage.load_metadata()
        self.history = self.storage.load_history()
        # orion: Load settings once; expose on Context and use for client overrides.
        self.settings = load_settings(self.repo_root)
        # orion: Autodetect provider and values in the client using settings and env; no need to pass base_url explicitly.
        self.client = ChatCompletionsClient(settings=self.settings)

        # orion: Resolve the external dependency root from the CLI-provided value only (environment variable removed).
        self.ext_root: Optional[pathlib.Path] = ext_dir_valid(external_dir)

        # orion: Discover internal tools reflectively and keep a name set for dispatch.
        self.internal_tool_specs = discover_tools()
        self.internal_tool_names = set(list_tool_names())

    # ---------- External tools (flat) ----------

    def _tool_list_pds(self, ctx: Context) -> Dict[str, Any]:
        """List PD filenames from the external directory (flat). Raw payload; no envelopes here."""
        if not self.ext_root:
            return {"_meta_error": "external directory not set or invalid."}
        try:
            items = list_project_descriptions(self.ext_root)
            return {"filenames": items}
        except Exception as e:
            return {"_meta_error": f"list_pds failed: {e}"}

    def _tool_get_pos(self, ctx: Context, filename: str) -> Dict[str, Any]:
        """Return or (re)generate POS for a given PD filename. Raw payload; no envelopes here."""
        if not self.ext_root:
            return {"_meta_error": "external directory not set or invalid."}
        if not filename:
            return {"_meta_error": "filename required"}
        try:
            from .summarizers import ensure_pos

            pos = ensure_pos(ctx, self.ext_root, filename, self.client)
            if not pos:
                return {"_meta_error": f"no POS available for {filename}"}
            return {"filename": filename, "summary": pos}
        except Exception as e:
            return {"_meta_error": f"get_pos failed for {filename}: {e}"}

    # ---------- System State helpers ----------

    # orion: Build the authoritative system_state from current repo paths and colocated summaries.
    def _build_system_state_message(self, ctx: Context) -> Dict[str, Any]:
        files = list_all_nonignored_files(self.repo_root)
        files_map: Dict[str, Any] = {}
        for p in files:
            summ = None
            sp = colocated_summary_path(self.repo_root, p)
            if sp.exists():
                try:
                    with sp.open("r", encoding="utf-8") as f:
                        summ = json.load(f)
                except Exception:
                    summ = None
            files_map[p] = {"kind": "summary", "body": summ, "meta": {"has_summary": bool(summ)}}
        # Conversations: include last 5 archived summaries (id, ts, s)
        archives = self.md.get("conversation_archives", []) if isinstance(self.md, dict) else []
        recent_archives = archives[-5:] if archives else []
        conv_obj = {
            "recent_archives": [
                {"id": a.get("id"), "ts": a.get("ts"), "s": a.get("s", "")} for a in recent_archives
            ]
        }
        # Pending changes: expose current list so the model can consult/modify them per turn
        pending_changes = self.md.get("pending_changes", []) if isinstance(self.md, dict) else []
        payload: Dict[str, Any] = {
            "type": "system_state",
            "version": 1,
            "files": files_map,
            "conversations": conv_obj,
            "pending_changes": pending_changes,
        }
        return {"type": "message", "role": "system", "content": json.dumps(payload, ensure_ascii=False)}

    # orion: Ensure a system_state is present as the first stored message; refresh colocated summaries first.
    def _ensure_system_state(self, ctx: Context) -> None:
        self._refresh_summaries(ctx)
        new_state_msg = self._build_system_state_message(ctx)
        # If no history or first item isn't a system_state, insert one and persist.
        first_is_state = False
        if self.history:
            try:
                if self.history[0].get("role") == "system":
                    c = self.history[0].get("content", "")
                    obj = json.loads(c) if isinstance(c, str) else {}
                    first_is_state = isinstance(obj, dict) and obj.get("type") == "system_state"
            except Exception:
                first_is_state = False
        if (self.history == []) or (not first_is_state):
            self.storage.append_raw_message(new_state_msg)
            self.history.insert(0, new_state_msg)
        else:
            # Keep first message as the refreshed authoritative state for the next call.
            self.history[0] = new_state_msg

    # orion: Locate the most recent system_state payload in history; returns a message dict or None.
    def _latest_system_state_message(self) -> Optional[Dict[str, Any]]:
        for msg in reversed(self.history):
            if msg.get("role") != "system":
                continue
            try:
                content = msg.get("content", "")
                obj = json.loads(content) if isinstance(content, str) else None
                if isinstance(obj, dict) and obj.get("type") == "system_state":
                    return {"type": "message", "role": "system", "content": json.dumps(obj, ensure_ascii=False)}
            except Exception:
                continue
        return None

    # ---------- Commands ----------

    def cmd_help(self, ctx: Context) -> None:
        """Print a list of supported commands and brief descriptions."""
        ctx.error_message("Commands:")
        ctx.send_to_user(":preview              - Show pending changes")
        ctx.send_to_user(":apply                - Apply all pending changes")
        ctx.send_to_user(":discard-change <id>  - Discard a pending change by id")
        # orion: Update help text to include :history and both clear commands (:clear-changes and alias :clearChanges) for discoverability.
        ctx.send_to_user(":history              - Show last 10 user↔assistant turns from history")
        # orion: Add :rerun to replay the most recent user message without adding a duplicate user entry.
        ctx.send_to_user(":rerun                - Rerun the last user message without duplicating it")
        ctx.send_to_user(":clear-changes        - Clear all pending changes")
        ctx.send_to_user(":clearChanges         - Alias for :clear-changes")
        ctx.send_to_user(":refresh              - Rescan repo and refresh summaries")
        ctx.send_to_user(":reset-state         - Refresh summaries and rebuild base system_state")
        ctx.send_to_user(":refresh-deps         - Refresh Project Orion Summaries for external dependencies")
        ctx.send_to_user(":status               - Show status summary")
        ctx.send_to_user(":consolidate          - Coalesce duplicate change batches")
        ctx.send_to_user(":help                 - Show this help")
        ctx.send_to_user(":quit                 - Exit")
        # orion: Document per-turn model override directive; parsed before command routing and applied only to the next conversation call.
        ctx.send_to_user(":model <model> <message> - Use model for only this turn")
        # orion: Remove token counting command; keep ad-hoc file splitting only.
        ctx.send_to_user(":splitFile <path>     - Split a file via model-assisted refactor and write results immediately")

    def cmd_preview(self, ctx: Context) -> None:
        """Display all pending change specs in a human-readable format."""
        changes = self.md["pending_changes"]
        if not changes:
            ctx.send_to_user("No pending changes.")
            return
        ctx.send_to_user(f"Pending changes ({len(changes)}):")
        for ch in changes:
            ctx.send_to_user(f"- {ch['id']} | {ch['title']}")
            ctx.send_to_user(f"  {ch['description']}")
            for it in ch.get("items", []):
                ctx.send_to_user(f"  * {it['change_type']} {it['path']} — {it['summary_of_change']}")

    def cmd_discard_change(self, ctx: Context, change_id: str) -> None:
        """Remove a single change by id from the pending list and persist metadata."""
        before = len(self.md["pending_changes"])
        self.md["pending_changes"] = [c for c in self.md["pending_changes"] if c.get("id") != change_id]
        after = len(self.md["pending_changes"])
        if before == after:
            ctx.send_to_user(f"No change with id {change_id} found.")
        else:
            self.storage.save_metadata(self.md)
            ctx.send_to_user(f"Discarded change {change_id}.")

    def _refresh_summaries(self, ctx: Context, only_paths: Optional[List[str]] = None) -> None:
        """
        Generate or update colocated per-file summaries for files in the repo.

        Args:
            only_paths: If provided, limit the refresh to this subset of paths.
        """
        # Determine file list
        if only_paths is not None:
            files = [normalize_path(p) for p in only_paths]
        else:
            files = list_all_nonignored_files(self.repo_root)

        if not files:
            ctx.log("No files found for summarization.")
            return

        path_to_digest = self.md.get("path_to_digest", {})
        self.md["path_to_digest"] = path_to_digest

        changed: List[str] = []
        skipped = 0
        created = 0
        from .config import SUMMARY_MAX_BYTES
        from .fs import _safe_abs, sha256_bytes

        for rel in files:
            # skip our internal .orion dirs (already excluded by listing, but guard anyway)
            if ".orion" in pathlib.Path(rel).parts:
                continue
            # size cap
            try:
                ap = _safe_abs(self.repo_root, rel)
                if ap.stat().st_size > SUMMARY_MAX_BYTES:
                    skipped += 1
                    continue
            except Exception:
                continue
            # compute digest
            try:
                b = ap.read_bytes()
            except Exception:
                continue
            digest = sha256_bytes(b)
            prev = path_to_digest.get(rel)
            need = prev != digest
            if need:
                res = summarize_file(ctx, self.client, self.repo_root, rel)
                if res:
                    path_to_digest[rel] = digest
                    changed.append(rel)
                    created += 1
                    self.storage.save_metadata(self.md)
        if only_paths is not None:
            ctx.log(f"Refreshed summaries for {len(changed)} file(s); skipped {skipped} large file(s).")
        else:
            ctx.log(f"Summarization complete. Updated {len(changed)} file(s); skipped {skipped} large file(s).")

    def cmd_refresh(self, ctx: Context) -> None:
        """Refresh summaries for local files and external PDs (if configured)."""
        self._refresh_summaries(ctx)
        # Also refresh external dependency POS if configured
        if self.ext_root:
            from .summarizers import ensure_all_pos

            ensure_all_pos(ctx, self.ext_root, self.client)
            ctx.log("Refreshed external dependency Project Orion Summaries.")

    def cmd_refresh_deps(self, ctx: Context) -> None:
        """Refresh Project Orion Summaries for all PDs in the external directory (if set)."""
        if not self.ext_root:
            ctx.log("External directory not set or invalid; nothing to refresh.")
            return
        from .summarizers import ensure_all_pos

        ensure_all_pos(ctx, self.ext_root, self.client)
        ctx.log("Refreshed external dependency Project Orion Summaries.")

    def cmd_reset_state(self, ctx: Context) -> None:
        """Rebuild the base system_state from current summaries and rewrite history.

        Steps:
          - Refresh summaries
          - Drop all prior system_state messages
          - Insert a single new summaries-only system_state at the front
          - Replay the remaining messages in order
        """
        # orion: Refresh summaries first to ensure the new system_state reflects the latest repo view.
        self._refresh_summaries(ctx)

        # Build a fresh summaries-only system_state message.
        new_state_msg = self._build_system_state_message(ctx)

        # Filter out all prior system_state messages from history, keep other messages as-is.
        remaining: List[Dict[str, Any]] = []
        for m in self.history:
            if m.get("type") == "message" and m.get("role") == "system":
                try:
                    content = m.get("content", "")
                    obj = json.loads(content) if isinstance(content, str) else None
                    if isinstance(obj, dict) and obj.get("type") == "system_state":
                        continue
                except Exception:
                    pass
            remaining.append(m)

        # Rewrite stored history: clear, then write new system_state, then replay remaining messages in order.
        self.storage.clear_history()
        self.history = []
        self.storage.append_raw_message(new_state_msg)
        self.history.append(new_state_msg)
        for m in remaining:
            self.storage.append_raw_message(m)
            self.history.append(m)

        ctx.send_to_user("Reset system_state from summaries and rebuilt conversation history.")

    def cmd_status(self, ctx: Context) -> None:
        """Print a one-line-per-field status report about the current Orion session and repo."""
        total_files = len(list_all_nonignored_files(self.repo_root))
        dep_count = 0
        if self.ext_root:
            try:
                dep_count = len(list_project_descriptions(self.ext_root))
            except Exception:
                dep_count = 0
        ctx.send_to_user(f"Repo root: {self.repo_root}")
        ctx.send_to_user(f"Plan ID: {self.md['plan_state']['plan_id']}")
        ctx.send_to_user(f"Pending changes: {len(self.md['pending_changes'])}")
        ctx.send_to_user(f"Batches since last consolidation: {self.md['batches_since_last_consolidation']}")
        ctx.send_to_user(f"Commit log entries: {len(self.md['plan_state']['commit_log'])}")
        ctx.send_to_user(f"Files (non-ignored): {total_files}")
        ctx.send_to_user(f"Tracked digests: {len(self.md.get('path_to_digest', {}))}")
        if self.ext_root:
            ctx.send_to_user(f"External PD root: {self.ext_root} (flat). PD files: {dep_count}")
        ctx.send_to_user(f"Archived conversations: {len(self.md.get('conversation_archives', []))}")

    def cmd_consolidate(self, ctx: Context) -> None:
        """Coalesce duplicate change batches and reset the consolidation counter."""
        changes = self.md["pending_changes"]
        seen = set()
        consolidated = []
        for ch in changes:
            key = (ch["title"], tuple(sorted(it["path"] for it in ch.get("items", []))))
            if key in seen:
                continue
            seen.add(key)
            consolidated.append(ch)
        self.md["pending_changes"] = consolidated
        self.md["batches_since_last_consolidation"] = 0
        self.storage.save_metadata(self.md)
        ctx.log(f"Consolidated. Pending changes now: {len(self.md['pending_changes'])}")

    # orion: Merge helper for pending ChangeSpecs. Replaces by id, deletes on empty items, appends new ids.
    def _merge_pending_changes(self, ctx: Context, new_changes: List[Dict[str, Any]]) -> None:
        """Merge a list of ChangeSpecs into the pending set with deterministic semantics.

        Rules:
          - If an incoming ChangeSpec's id matches an existing one:
              * items == []  -> delete that spec
              * items != []  -> replace that spec (entire object)
          - If the id does not exist:
              * items == []  -> ignore (no-op)
              * items != []  -> append as new
        """
        if not new_changes:
            return
        normalized = validate_change_specs(new_changes)
        if not normalized:
            ctx.log("No valid change specs to merge.")
            return
        existing_list = self.md.get("pending_changes", [])
        existing_map: Dict[str, Dict[str, Any]] = {
            c.get("id"): c for c in existing_list if isinstance(c, dict) and c.get("id")
        }
        replaced = 0
        deleted = 0
        appended = 0
        for nc in normalized:
            cid = nc.get("id")
            items = nc.get("items", [])
            if not cid:
                continue
            if cid in existing_map:
                if len(items) == 0:
                    del existing_map[cid]
                    deleted += 1
                else:
                    existing_map[cid] = nc
                    replaced += 1
            else:
                if len(items) == 0:
                    # deletion request for non-existent id: no-op
                    continue
                existing_map[cid] = nc
                appended += 1
        new_list = list(existing_map.values())
        if new_list != existing_list:
            self.md["pending_changes"] = new_list
            self.md["batches_since_last_consolidation"] += 1
            self.storage.save_metadata(self.md)
            ctx.log(f"Merged pending changes: +{appended}, ~{replaced}, -{deleted}")
        else:
            ctx.log("No changes to pending changes after merge.")

    # orion: Add helper used by both :clear-changes and :clearChanges to clear pending changes and persist metadata.
    def cmd_clear_changes(self, ctx: Context) -> None:
        """Clear all pending changes and persist metadata."""
        self.md["pending_changes"] = []
        self.storage.save_metadata(self.md)
        ctx.send_to_user("Cleared all pending changes.")

    # orion: Add :history command to print the last 10 user↔assistant turns, excluding system/tool messages.
    def cmd_history(self, ctx: Context, max_turns: int = 10) -> None:
        """Print the last N user↔assistant turns from stored history."""
        # Filter to only simple chat messages between user and assistant
        msgs = [m for m in self.history if m.get("type") == "message" and m.get("role") in ("user", "assistant")]
        turns: List[tuple[Optional[str], Optional[str]]] = []
        pending_user: Optional[str] = None
        for m in msgs:
            role = m.get("role")
            content = str(m.get("content", ""))
            if role == "user":
                if pending_user is not None:
                    # orion: Close a user-only turn when no assistant reply arrived before the next user message.
                    turns.append((pending_user, None))
                pending_user = content
            else:  # assistant
                if pending_user is None:
                    # orion: Record assistant-only outputs that may precede any user turn (e.g., system boot echoes excluded above).
                    turns.append((None, content))
                else:
                    turns.append((pending_user, content))
                    pending_user = None
        if pending_user is not None:
            turns.append((pending_user, None))
        if not turns:
            ctx.send_to_user("No user or assistant messages in history.")
            return
        last = turns[-max_turns:]
        ctx.send_to_user(f"Last {len(last)} turn(s):")
        for u, a in last:
            if u is not None:
                ctx.send_to_user(f"User: {u}")
            if a is not None:
                ctx.send_to_user(f"Assistant: {a}")

    # orion: Implement :rerun to resend the most recent user message without duplicating it in history.
    # It rebuilds the message list up to and including that user turn, excluding any later assistant/tool outputs.
    def cmd_rerun(self, ctx: Context) -> None:
        # Locate the most recent user message in history
        last_user_idx: Optional[int] = None
        for i in range(len(self.history) - 1, -1, -1):
            m = self.history[i]
            if m.get("type") == "message" and m.get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            ctx.send_to_user("No previous user message to rerun.")
            return

        # orion: Ensure system_state exists and is fresh before model calls, as in normal conversation.
        self._ensure_system_state(ctx)

        # orion: Allow the same project-level prompt customization used in handle_user_input. Duplicated here for clarity.
        system_text = get_prompt("prompt_conversation_system.txt")
        try:
            proj_prompt = read_file(self.repo_root, ".orion/prompt_conversation_system.txt")
            if proj_prompt.strip():
                system_text = proj_prompt
                ctx.log("Using project-level conversation system prompt: .orion/prompt_conversation_system.txt")
            else:
                raise ValueError("empty project prompt")
        except Exception:
            try:
                override_text = read_file(self.repo_root, ".orion/prompt_conversation_system.override.txt")
                if override_text.strip():
                    system_text = override_text
                    ctx.log("Using project-level conversation prompt override: .orion/prompt_conversation_system.override.txt")
            except Exception:
                pass
            try:
                addendum_text = read_file(self.repo_root, ".orion/prompt_conversation_system.addendum.txt")
                if addendum_text.strip():
                    system_text = f"{system_text.rstrip()}\n{addendum_text}"
                    ctx.log("Applied project-level conversation prompt addendum: .orion/prompt_conversation_system.addendum.txt")
            except Exception:
                pass

        response_schema = ConversationResponse.model_json_schema()

        # orion: Build messages by trimming history to the last user message (inclusive), excluding any prior system_state echoes.
        messages: List[Dict[str, Any]] = []
        latest_state = self._latest_system_state_message()
        if latest_state:
            messages.append(latest_state)
        messages.append({"type": "message", "role": "system", "content": system_text})
        for idx, h in enumerate(self.history):
            if idx > last_user_idx:
                break
            if h.get("role") == "system":
                try:
                    obj = json.loads(h.get("content", ""))
                    if isinstance(obj, dict) and obj.get("type") == "system_state":
                        continue
                except Exception:
                    pass
            # orion: Replay up to and including the last user message; do not append a new user entry.
            msg = h.copy()
            if "ts" in msg:
                del msg["ts"]
            messages.append(msg)

        # Tools and runner mirror normal conversation flow
        tools: List[Dict[str, Any]] = []
        tools.extend(self.internal_tool_specs)
        tools.extend(_external_tool_definitions())

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            reason = str(args.get("reason_for_call") or "")
            if name in self.internal_tool_names:
                return run_tool(ctx, name, args)
            if name == "list_project_descriptions":
                payload = self._tool_list_pds(ctx)
                return {"reason_for_call": reason, "result": payload}
            if name == "get_project_orion_summary":
                filename = str(args.get("filename") or "")
                payload = self._tool_get_pos(ctx, filename)
                return {"reason_for_call": reason, "result": payload}
            return {"reason_for_call": reason, "result": {"_meta_error": f"unknown tool {name}", "_args_echo": args}}

        def sink(msg: Dict[str, Any]) -> None:
            self.storage.append_raw_message(msg)
            self.history.append(msg)

        ctx.log("Calling model for rerun of the last user message...")
        final_json = self.client.call_responses(
            ctx,
            messages,
            tools,
            response_schema,
            interactive_tool_runner=runner,
            message_sink=sink,
            call_type="conversation",
        )

        try:
            parsed = ConversationResponse.model_validate(final_json)
        except ValidationError:
            ctx.log("Model returned invalid conversation response on rerun.")
            return

        assistant_msg = parsed.assistant_message
        changes = [c.model_dump() for c in parsed.changes]
        ctx.send_to_user(assistant_msg)
        self.storage.append_raw_message({"type":"message", "role": "assistant", "content": assistant_msg})
        self.history.append({"type":"message", "role": "assistant", "content": assistant_msg})

        if changes:
            self._merge_pending_changes(ctx, changes)

    # orion: Remove the token counting command (:tokenCount) and its tiktoken dependency; only :splitFile remains.
    # orion: New command to split a file by invoking a dedicated split prompt and applying returned patches immediately.
    def cmd_split_file(self, ctx: Context, path: str) -> None:
        np = normalize_path(path)
        try:
            content = read_file(self.repo_root, np)
        except Exception as e:
            ctx.error_message(f"Could not read {np}: {e}")
            return
        # Build split prompt and schema
        system_text = get_prompt("prompt_split_system.txt")
        user_payload = {
            "path": np,
            "line_cap": LINE_CAP,
            "content": content,
        }
        schema = ApplyResponse.model_json_schema()

        # Build minimal tool surface (internal + external) for optional lookups.
        tools: List[Dict[str, Any]] = []
        tools.extend(self.internal_tool_specs)
        tools.extend(_external_tool_definitions())

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            reason = str(args.get("reason_for_call") or "")
            if name in self.internal_tool_names:
                return run_tool(ctx, name, args)
            if name == "list_project_descriptions":
                payload = self._tool_list_pds(ctx)
                return {"reason_for_call": reason, "result": payload}
            if name == "get_project_orion_summary":
                filename = str(args.get("filename") or "")
                payload = self._tool_get_pos(ctx, filename)
                return {"reason_for_call": reason, "result": payload}
            return {"reason_for_call": reason, "result": {"_meta_error": f"unknown tool {name}", "_args_echo": args}}

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        ctx.log(f"Calling model to split file: {np}")
        final_json = self.client.call_responses(
            ctx,
            messages=messages,
            tools=tools,
            response_schema=schema,
            interactive_tool_runner=runner,
            call_type="split",
            _timeout=1800,
        )

        try:
            parsed = ApplyResponse.model_validate(final_json)
        except ValidationError as e:
            ctx.error_message(f"Split failed: invalid response from model: {e}")
            return

        if parsed.mode == "incompatible":
            ctx.send_to_user("Split operation reported incompatibility:")
            for issue in parsed.issues:
                ctx.send_to_user(f"- {issue.reason}: {', '.join(issue.paths)}")
            ctx.send_to_user(f"Explanation: {parsed.explanation}")
            return

        # Write files and refresh summaries
        written_paths: List[str] = []
        for f in parsed.files:
            # orion: Field renamed from 'code' to 'contents'; write the new field.
            write_file(self.repo_root, f.path, f.contents)
            written_paths.append(f.path)
        if written_paths:
            self._refresh_summaries(ctx, only_paths=written_paths)

        # Record a commit-log style entry for traceability
        self.md["plan_state"]["commit_log"].append(
            {"id": short_id("split"), "ts": now_ts(), "paths": written_paths, "explanation": parsed.explanation}
        )
        self.storage.save_metadata(self.md)
        ctx.send_to_user(f"Wrote {len(written_paths)} file(s). Explanation: {parsed.explanation}")

    def cmd_apply(self, ctx: Context) -> None:
        """
        Apply all pending changes by asking the model to produce a file map.

        The model returns a strict JSON payload that is validated before any disk
        writes occur. On success, files are written, commit log updated, summaries
        refreshed for touched paths, and the conversation is reset.
        """
        pending = self.md["pending_changes"]
        if not pending:
            ctx.send_to_user("No pending changes to apply.")
            return

        # orion: Ensure system_state exists and is fresh before model calls.
        self._ensure_system_state(ctx)

        # Collect affected paths
        affected: set[str] = set()
        for ch in pending:
            for it in ch.get("items", []):
                affected.add(it["path"])

        # orion: Promote affected files to 'full' in system_state so it becomes the single source of truth for contents.
        # This avoids duplicating raw file text in the user payload and aligns with how get_file_contents promotions work.
        # Steps: read current contents + meta, update the latest system_state.files[path] = {kind:"full", body, meta}, bump version,
        # and use the upgraded system_state as the first message of the apply call.
        content_map: Dict[str, Dict[str, Any]] = {}
        for p in sorted(affected):
            content = ""
            abs_path = self.repo_root / p
            if abs_path.exists():
                try:
                    content = read_file(self.repo_root, p)
                except Exception:
                    content = ""
            try:
                bcount = len(content.encode("utf-8"))
            except Exception:
                bcount = 0
            lcount = count_lines(content)
            content_map[p] = {"content": content, "meta": {"line_count": lcount, "bytes": bcount}}

        latest_state = self._latest_system_state_message()
        if not latest_state:
            # Fallback: ensure and rebuild one if missing
            self._ensure_system_state(ctx)
            latest_state = self._latest_system_state_message()
        if latest_state:
            try:
                state_obj = json.loads(latest_state["content"]) if isinstance(latest_state.get("content"), str) else None
            except Exception:
                state_obj = None
            if isinstance(state_obj, dict) and state_obj.get("type") == "system_state":
                files_map = state_obj.get("files") or {}
                for path_key, info in content_map.items():
                    entry = files_map.get(path_key)
                    if isinstance(entry, dict) and entry.get("kind") == "full":
                        # Already full; skip overwrite to avoid redundant churn
                        continue
                    files_map[path_key] = {
                        "kind": "full",
                        "body": info["content"],
                        "meta": {"line_count": info["meta"]["line_count"], "bytes": info["meta"]["bytes"]},
                    }
                state_obj["files"] = files_map
                try:
                    state_obj["version"] = int(state_obj.get("version", 0) or 0) + 1
                except Exception:
                    state_obj["version"] = 1
                latest_state = {"type": "message", "role": "system", "content": json.dumps(state_obj, ensure_ascii=False)}

        # orion: Build Apply prompt and a compact user payload that references no raw file contents.
        # We intentionally send an empty files array because the upgraded system_state now carries full contents.
        files_payload: List[Dict[str, Any]] = []

        # orion: Load Apply system prompt from resources so it can be maintained externally.
        system_text = get_prompt("prompt_apply_system.txt")
        user_text = json.dumps({"changes": pending, "files": files_payload}, ensure_ascii=False)

        # orion: Replace inline JSON Schema with centralized Pydantic model schema.
        response_schema = ApplyResponse.model_json_schema()

        # orion: Build tools from reflective internal specs + explicit external PD specs.
        tools = []
        tools.extend(self.internal_tool_specs)
        tools.extend(_external_tool_definitions())

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            # orion: Generic runner that injects/strips reason_for_call and returns a wrapped result.
            reason = str(args.get("reason_for_call") or "")
            if name in self.internal_tool_names:
                # Internal: defer to reflection runner which returns the envelope already.
                return run_tool(ctx, name, args)
            # External tools: call typed wrappers and wrap result.
            if name == "list_project_descriptions":
                payload = self._tool_list_pds(ctx)
                return {"reason_for_call": reason, "result": payload}
            if name == "get_project_orion_summary":
                filename = str(args.get("filename") or "")
                payload = self._tool_get_pos(ctx, filename)
                return {"reason_for_call": reason, "result": payload}
            return {"reason_for_call": reason, "result": {"_meta_error": f"unknown tool {name}", "_args_echo": args}}

        # orion: Build messages: upgraded latest system_state at the front, then the apply system prompt, then the user payload.
        messages: List[Dict[str, Any]] = []
        if latest_state:
            messages.append(latest_state)
        else:
            # In the unlikely event state is still missing, ensure one more time (non-fatal if still None)
            self._ensure_system_state(ctx)
            ls2 = self._latest_system_state_message()
            if ls2:
                messages.append(ls2)
        messages.append({"type": "message", "role": "system", "content": system_text})
        messages.append({"type": "message", "role": "user", "content": user_text})

        ctx.log("Calling model to apply changes...")
        # orion: Migrate :apply flow to Responses API using Pydantic schema; tool runner preserved.
        final_json = self.client.call_responses(
            ctx,
            messages,
            tools,
            response_schema,
            interactive_tool_runner=runner,
            call_type="apply",
            _timeout=3000,
        )

        # orion: Parse and validate with Pydantic; surface helpful error messages on failure.
        try:
            parsed = ApplyResponse.model_validate(final_json)
        except ValidationError as e:
            ctx.error_message(f"Apply failed: invalid response from model: {e}")
            return

        if parsed.mode == "incompatible":
            ctx.send_to_user("Model reported incompatibility:")
            for issue in parsed.issues:
                ctx.send_to_user(f"- {issue.reason}: {', '.join(issue.paths)}")
            ctx.send_to_user(f"Explanation: {parsed.explanation}")
            return

        # Write files
        written_paths: List[str] = []
        for f in parsed.files:
            # orion: Field renamed from 'code' to 'contents'; write the new field.
            write_file(self.repo_root, f.path, f.contents)
            written_paths.append(f.path)

        # Commit log entry
        self.md["plan_state"]["commit_log"].append(
            {"id": short_id("commit"), "ts": now_ts(), "paths": [f.path for f in parsed.files], "explanation": parsed.explanation}
        )
        self.storage.save_metadata(self.md)
        ctx.log(f"Wrote {len(parsed.files)} files. Explanation: {parsed.explanation}")

        # Refresh summaries for written files (including brand-new files)
        if written_paths:
            self._refresh_summaries(ctx, only_paths=written_paths)

        # Archive-time summary and rotation
        try:
            # Append an apply-role marker with changed paths
            apply_marker = {"type": "message", "role": "apply", "content": json.dumps(written_paths, ensure_ascii=False)}
            self.storage.append_raw_message(apply_marker)
            self.history.append(apply_marker)

            # Build transcript of user/assistant/apply messages only
            transcript: List[Dict[str, str]] = []
            for m in self.history:
                if m.get("type") == "message" and m.get("role") in ("user", "assistant", "apply"):
                    transcript.append({
                        "role": str(m.get("role")),
                        "content": str(m.get("content", ""))
                    })

            # Summarize via InfoSummary using the archive prompt
            archive_system = get_prompt("prompt_archive_summary_system.txt")
            archive_schema = InfoSummary.model_json_schema()
            archive_messages = [
                {"role": "system", "content": archive_system},
                {"role": "user", "content": json.dumps({"transcript": transcript}, ensure_ascii=False)},
            ]
            archive_json = self.client.call_responses(
                ctx,
                messages=archive_messages,
                tools=[],
                response_schema=archive_schema,
                call_type="archive_summary",
            )
            try:
                info = InfoSummary.model_validate(archive_json)
                summary_text = info.s
            except ValidationError:
                summary_text = f"Applied {len(written_paths)} change(s)."

            # Rotate current conversation to a stable-id backup and record metadata
            archive_id = short_id("conv")
            backup_path = self.storage.clear_history(archive_id)
            record = {
                "id": archive_id,
                "ts": now_ts(),
                "filename": str(backup_path) if backup_path else "",
                "s": summary_text,
            }
            archives = self.md.get("conversation_archives", [])
            archives.append(record)
            if len(archives) > 200:
                archives = archives[-200:]
            self.md["conversation_archives"] = archives
            self.storage.save_metadata(self.md)
        except Exception:
            # Non-fatal: proceed with cleanup even if archiving/summarization fails
            pass

        # Clear conversation and pending changes (history already rotated above)
        self.history = []
        self.md["pending_changes"] = []
        self.md["batches_since_last_consolidation"] = 0
        self.storage.save_metadata(self.md)
        ctx.log("Cleared conversation history and pending change log.")

        # orion: Removed post-apply LINE_CAP auto-splitting; users can now run :splitFile when desired.

    def handle_user_input(self, ctx: Context, text: str) -> None:
        """
        Handle a line of user input: either execute a command or advance the conversation.
        """
        # orion: Support per-turn model override via ':model <model> <message>'. Parse this before command routing;
        # on success, strip the directive, log usage, and proceed down the conversation path with an override.
        override_model: Optional[str] = None
        was_model_override = False
        if text.startswith(":model"):
            parts_model = text.strip().split(maxsplit=2)
            if len(parts_model) >= 3 and parts_model[1] and parts_model[2].strip():
                override_model = parts_model[1]
                text = parts_model[2].strip()
                was_model_override = True
                ctx.log(f"Using per-turn model override: {override_model}")
            else:
                ctx.error_message("Usage: :model <model> <message>")
                return

        # Commands
        if (not was_model_override) and text.startswith(":"):
            parts = text.strip().split()
            cmd = parts[0]
            if cmd == ":help":
                self.cmd_help(ctx)
            elif cmd == ":preview":
                self.cmd_preview(ctx)
            elif cmd == ":apply":
                self.cmd_apply(ctx)
            elif cmd == ":discard-change":
                if len(parts) < 2:
                    ctx.error_message("Usage: :discard-change <id>")
                else:
                    self.cmd_discard_change(ctx, parts[1])
            # orion: Route :history to new cmd_history for compact display of last user↔assistant turns.
            elif cmd == ":history":
                self.cmd_history(ctx)
            # orion: Wire :rerun to replay the most recent user message without creating a duplicate entry.
            elif cmd == ":rerun":
                self.cmd_rerun(ctx)
            # orion: Use a shared helper for both kebab-case and camelCase clear commands.
            elif cmd == ":clear-changes" or cmd == ":clearChanges":
                self.cmd_clear_changes(ctx)
            elif cmd == ":refresh":
                self.cmd_refresh(ctx)
            elif cmd == ":reset-state":
                self.cmd_reset_state(ctx)
            elif cmd == ":refresh-deps":
                self.cmd_refresh_deps(ctx)
            elif cmd == ":status":
                self.cmd_status(ctx)
            elif cmd == ":consolidate":
                self.cmd_consolidate(ctx)
            # orion: Remove :tokenCount; keep :splitFile utility command wired into the REPL.
            elif cmd == ":splitFile":
                if len(parts) < 2:
                    ctx.error_message("Usage: :splitFile <path>")
                else:
                    self.cmd_split_file(ctx, parts[1])
            elif cmd == ":quit":
                ctx.send_to_user("Goodbye.")
                import sys

                sys.exit(0)
            else:
                ctx.error_message(f"Unknown command: {cmd}. Type :help for help.")
            return

        # orion: New conversation bootstrap via system_state. Ensure authoritative state exists at session start.
        self._ensure_system_state(ctx)

        # Append user turn (raw). When a per-turn model override is used, only <message> (without the directive) is appended.
        self.storage.append_raw_message({"type": "message", "role": "user", "content": text})
        self.history.append({"type": "message", "role": "user", "content": text})

        # orion: Allow project-level conversation prompt customization with clear precedence and safe fallbacks.
        # Priority:
        # 1) Use .orion/prompt_conversation_system.txt if present and non-empty (verbatim, conversation-only scope).
        # 2) Else start from packaged default; if .orion/prompt_conversation_system.override.txt exists and is non-empty, replace base.
        # 3) Then, if .orion/prompt_conversation_system.addendum.txt exists and is non-empty, append to the base.
        system_text = get_prompt("prompt_conversation_system.txt")
        try:
            proj_prompt = read_file(self.repo_root, ".orion/prompt_conversation_system.txt")
            if proj_prompt.strip():
                system_text = proj_prompt
                ctx.log("Using project-level conversation system prompt: .orion/prompt_conversation_system.txt")
            else:
                raise ValueError("empty project prompt")
        except Exception:
            try:
                override_text = read_file(self.repo_root, ".orion/prompt_conversation_system.override.txt")
                if override_text.strip():
                    system_text = override_text
                    ctx.log("Using project-level conversation prompt override: .orion/prompt_conversation_system.override.txt")
            except Exception:
                pass
            try:
                addendum_text = read_file(self.repo_root, ".orion/prompt_conversation_system.addendum.txt")
                if addendum_text.strip():
                    system_text = f"{system_text.rstrip()}\n{addendum_text}"
                    ctx.log("Applied project-level conversation prompt addendum: .orion/prompt_conversation_system.addendum.txt")
            except Exception:
                pass

        # orion: Replace inline JSON Schema with centralized Pydantic model schema.
        response_schema = ConversationResponse.model_json_schema()

        # Rebuild messages: prepend latest system_state, then conversation system prompt, then replay history excluding previous system_state echoes
        messages: List[Dict[str, Any]] = []
        latest_state = self._latest_system_state_message()
        if latest_state:
            messages.append(latest_state)
        messages.append({"type": "message", "role": "system", "content": system_text})
        for h in self.history:
            # Filter out prior system_state messages to avoid duplication; the latest is already prepended.
            if h.get("role") == "system":
                try:
                    obj = json.loads(h.get("content", ""))
                    if isinstance(obj, dict) and obj.get("type") == "system_state":
                        continue
                except Exception:
                    pass
            msg = h.copy()
            if "ts" in msg:
                del msg["ts"]
            messages.append(msg)

        # orion: Build tools from reflective internal specs + explicit external PD specs.
        tools = []
        tools.extend(self.internal_tool_specs)
        tools.extend(_external_tool_definitions())

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            # orion: Generic runner that injects/strips reason_for_call and returns a wrapped result.
            reason = str(args.get("reason_for_call") or "")
            if name in self.internal_tool_names:
                return run_tool(ctx, name, args)
            if name == "list_project_descriptions":
                payload = self._tool_list_pds(ctx)
                return {"reason_for_call": reason, "result": payload}
            if name == "get_project_orion_summary":
                filename = str(args.get("filename") or "")
                payload = self._tool_get_pos(ctx, filename)
                return {"reason_for_call": reason, "result": payload}
            return {"reason_for_call": reason, "result": {"_meta_error": f"unknown tool {name}", "_args_echo": args}}

        # Sink to persist assistant/tool messages during the call
        def sink(msg: Dict[str, Any]) -> None:
            self.storage.append_raw_message(msg)
            self.history.append(msg)

        ctx.log("Calling model for conversation response...")
        # orion: Pass the optional per-turn model override to Responses API; when None, the client's default model is used.
        final_json = self.client.call_responses(
            ctx,
            messages,
            tools,
            response_schema,
            interactive_tool_runner=runner,
            message_sink=sink,
            call_type="conversation",
            model=override_model,
        )

        # orion: Parse with Pydantic models, then convert to plain dicts for storage.
        try:
            parsed = ConversationResponse.model_validate(final_json)
        except ValidationError:
            ctx.log("Model returned invalid conversation response.")
            return

        assistant_msg = parsed.assistant_message
        changes = [c.model_dump() for c in parsed.changes]
        ctx.send_to_user(assistant_msg)
        self.storage.append_raw_message({"type":"message", "role": "assistant", "content": assistant_msg})
        self.history.append({"type":"message", "role": "assistant", "content": assistant_msg})

        if changes:
            self._merge_pending_changes(ctx, changes)

    def run(self) -> None:
        """Start the interactive REPL loop for Orion."""
        print(f"Orion ready at repo root: {self.repo_root}")
        if self.ext_root:
            print(f"External dependency PD root: {self.ext_root} (flat)")
        else:
            print("External dependency PD root not set (use --external-dir to enable).")
        print("Type :help for commands.")
        # orion: Recommend ignoring .httpcalls logs; this is a one-line, non-fatal tip shown at startup if .gitignore lacks an entry for .httpcalls.
        try:
            gi = (self.repo_root / ".gitignore")
            needs_tip = True
            if gi.exists():
                try:
                    if ".httpcalls" in gi.read_text(encoding="utf-8"):
                        needs_tip = False
                except Exception:
                    pass
            if needs_tip:
                print("Tip: add .httpcalls to .gitignore to avoid committing request logs.")
        except Exception:
            pass
        ctx = Context(self.repo_root, settings=self.settings)
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            if not text:
                continue
            self.handle_user_input(ctx, text)


# orion: Update docstring to reflect CLI-driven external directory configuration and revised help text.

def main() -> None:
    """
    Orion CLI entrypoint.

    Usage:
        orion [--external-dir PATH|-e PATH] [repo_root]

    Notes:
        - OPENAI_API_KEY and AI_MODEL must be set in the environment.
        - Optional: ORION_DEP_TTL_SEC influences dependency summary TTL behavior.
        - If repo_root is not supplied, the current directory is used.

    Options:
        -e, --external-dir PATH   External Project Description directory (flat). When omitted, external dependency features are disabled.
    """
    args = sys.argv[1:]

    # Help handling (recognized anywhere in argv)
    if any(a in ("-h", "--help") for a in args):
        print("Usage: orion [--external-dir PATH|-e PATH] [repo_root]")
        print("Options:")
        print("  -e, --external-dir PATH   External Project Description directory (flat). When omitted, external deps are disabled.")
        print("Environment:")
        print("  OPENAI_API_KEY, AI_MODEL, ORION_DEP_TTL_SEC")
        return

    # orion: Implement minimal flag parsing for --external-dir/-e (supports --external-dir=PATH and -e PATH). The first non-flag argument is treated as repo_root.
    external_dir = None
    repo_root_arg = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-e":
            if i + 1 >= len(args):
                print("error: -e requires a PATH argument")
                return
            external_dir = args[i + 1]
            i += 2
            continue
        if a == "--external-dir":
            if i + 1 >= len(args):
                print("error: --external-dir requires a PATH argument")
                return
            external_dir = args[i + 1]
            i += 2
            continue
        if a.startswith("--external-dir="):
            external_dir = a.split("=", 1)[1]
            i += 1
            continue
        if a.startswith("-"):
            print(f"error: unknown option: {a}")
            return
        # First non-flag is repo_root
        if repo_root_arg is None:
            repo_root_arg = a
        i += 1

    repo_root = pathlib.Path(repo_root_arg).resolve() if repo_root_arg else pathlib.Path(".").resolve()
    Orion(repo_root, external_dir=external_dir).run()


if __name__ == "__main__":
    main()
