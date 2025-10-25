# orion: Ported the core Orion class, tool definitions, command handlers, bootstrap/summaries flow, and change-spec validators from editor.py into a focused module. Adjusted imports to use the new modular structure. Externalized conversation/apply system prompts via orion.prompts.get_prompt. Added docstrings and inline comments where flow or validation is non-obvious.

import json
import pathlib
from typing import Any, Dict, List, Optional

from .config import OPENAI_API_KEY, AI_MODEL, LINE_CAP
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
from .tools import (
    tool_list_paths,
    tool_get_file_contents,
    tool_get_file_snippet,
    tool_get_summary,
    tool_search_code,
    tool_ask_user,
)
from .prompts import get_prompt

# orion: Import centralized Pydantic models and validation utilities; this replaces ad-hoc schema dicts.
from pydantic import ValidationError, TypeAdapter
from .models import (
    ConversationResponse,
    ApplyResponse,
    ChangeSpec,
    ChangeItem,
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

# orion: Tools include local repo operations and external PD summary accessors; documented for clarity.

def tool_definitions() -> List[Dict[str, Any]]:
    """Return the list of tool definitions exposed to the model for function-calling."""
    # Local repo tools + external dependency tools
    defs = [
        {
            "type": "function",
            "name": "list_paths",
            "description": "List repository files; optionally filter by glob.",
            "parameters": {"type": "object", "properties": {"glob": {"type": "string"}}, "required": [], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "get_file_contents",
            "description": "Return full contents for a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "get_file_snippet",
            "description": "Return a line-range snippet for a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_summary",
            "description": "Return a brief machine-oriented summary for a local repo file, if available.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "search_code",
            "description": "Search files for a substring; returns paths.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "ask_user",
            "description": "Ask the user for a clarification.",
            "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"], "additionalProperties": False},
        },
    ]

    # External dependency tools (flat directory)
    defs.extend(
        [
            {
                "type": "function",
                "name": "list_project_descriptions",
                "description": "List dependency Project Descriptions (filenames) from the external directory.",
                "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            },
            {
                "type": "function",
                "name": "get_project_orion_summary",
                "description": "Return the Project Orion Summary (POS) for a given PD filename; regenerates if stale.",
                "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"], "additionalProperties": False},
            },
        ]
    )
    return defs


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

    def __init__(self, repo_root: pathlib.Path) -> None:
        """Initialize Orion with a repository root and supporting services."""
        self.repo_root = repo_root.resolve()
        self.storage = Storage(self.repo_root)
        self.md = self.storage.load_metadata()
        self.history = self.storage.load_history()
        self.client = ChatCompletionsClient(OPENAI_API_KEY, AI_MODEL)

        # External dependency root (flat)
        from .config import ORION_EXTERNAL_DIR

        self.ext_root: Optional[pathlib.Path] = ext_dir_valid(ORION_EXTERNAL_DIR)

        # orion: Build tools registry mapping tool names to callables that capture repo_root and other context.
        self.tools_registry = {
            # Local repo
            "list_paths": lambda ctx, args: tool_list_paths(ctx, self.repo_root, args),
            "get_file_contents": lambda ctx, args: tool_get_file_contents(ctx, self.repo_root, args),
            "get_file_snippet": lambda ctx, args: tool_get_file_snippet(ctx, self.repo_root, args),
            "get_summary": lambda ctx, args: tool_get_summary(ctx, self, args),
            "search_code": lambda ctx, args: tool_search_code(ctx, self.repo_root, args),
            "ask_user": lambda ctx, args: tool_ask_user(ctx, args),
            # External dependencies (flat)
            "list_project_descriptions": lambda ctx, args: self._tool_list_pds(ctx, args),
            "get_project_orion_summary": lambda ctx, args: self._tool_get_pos(ctx, args),
        }

    # ---------- External tools (flat) ----------

    def _tool_list_pds(self, ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper: list PD filenames from ORION_EXTERNAL_DIR (flat)."""
        if not self.ext_root:
            return {"_meta_error": "ORION_EXTERNAL_DIR not set or invalid.", "_args_echo": args}
        try:
            items = list_project_descriptions(self.ext_root)
            return {"filenames": items, "_args_echo": args}
        except Exception as e:
            return {"_meta_error": f"list_pds failed: {e}", "_args_echo": args}

    def _tool_get_pos(self, ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper: return or (re)generate POS for a given PD filename."""
        if not self.ext_root:
            return {"_meta_error": "ORION_EXTERNAL_DIR not set or invalid.", "_args_echo": args}
        filename = str(args.get("filename") or "")
        if not filename:
            return {"_meta_error": "filename required", "_args_echo": args}
        try:
            from .summarizers import ensure_pos

            pos = ensure_pos(ctx, self.ext_root, filename, self.client)
            if not pos:
                return {"_meta_error": f"no POS available for {filename}", "_args_echo": args}
            return {"filename": filename, "summary": pos, "_args_echo": args}
        except Exception as e:
            return {"_meta_error": f"get_pos failed for {filename}: {e}", "_args_echo": args}

    # ---------- Bootstrap helpers ----------

    def _build_bootstrap_message(self, ctx: Context, pending_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a one-time bootstrap system message with the full file list and summaries.

        Includes colocated per-file summaries (if present) and lightweight heads
        for external dependency projects if configured.
        """
        files = list_all_nonignored_files(self.repo_root)
        items = []
        for p in files:
            summ = None
            sp = colocated_summary_path(self.repo_root, p)
            if sp.exists():
                try:
                    with sp.open("r", encoding="utf-8") as f:
                        summ = json.load(f)
                except Exception:
                    summ = None
            items.append({"path": p, "summary": summ})

        payload: Dict[str, Any] = {
            "type": "orion_bootstrap",
            "note": "This is the complete list of files in the repository at this time; treat it as authoritative.And these are the complete list of pending changes.",
            "complete_list": True,
            "files": items,
            "pending_changes": pending_changes,
        }

        # External dependency heads (filenames + small counts)
        if self.ext_root:
            try:
                heads = ensure_all_pos(ctx, self.ext_root, self.client)
            except Exception:
                heads = []
            payload["dependency_projects"] = heads

        return {"type":"message","role": "system", "content": json.dumps(payload, ensure_ascii=False)}

    def _ensure_bootstrap(self, ctx: Context, pending_changes: List[Dict[str, Any]]) -> None:
        """
        If a conversation has not yet started, refresh local/external summaries and
        persist an initial bootstrap system message for the model.
        """
        self._refresh_summaries(ctx)
        # Ensure external POS (if configured)
        if self.ext_root:
            ensure_all_pos(ctx, self.ext_root, self.client)
        bootstrap_msg = self._build_bootstrap_message(ctx, pending_changes)
        if ( self.history == [] ) or ( self.history[0].get("type") != "orion_bootstrap" ):
            self.storage.append_raw_message(bootstrap_msg)
            self.history.insert(0, bootstrap_msg)
        else:
            self.history[0] = bootstrap_msg

    # ---------- Commands ----------

    def cmd_help(self, ctx: Context) -> None:
        """Print a list of supported commands and brief descriptions."""
        ctx.error_message("Commands:")
        ctx.send_to_user(":preview              - Show pending changes")
        ctx.send_to_user(":apply                - Apply all pending changes")
        ctx.send_to_user(":discard-change <id>  - Discard a pending change by id")
        ctx.send_to_user(":refresh              - Rescan repo and refresh summaries")
        ctx.send_to_user(":refresh-deps         - Refresh Project Orion Summaries for external dependencies")
        ctx.send_to_user(":status               - Show status summary")
        ctx.send_to_user(":consolidate          - Manually consolidate pending changes")
        ctx.send_to_user(":help                 - Show this help")
        ctx.send_to_user(":quit                 - Exit")

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
        """Refresh Project Orion Summaries for all PDs in ORION_EXTERNAL_DIR (if set)."""
        if not self.ext_root:
            ctx.log("ORION_EXTERNAL_DIR not set or invalid; nothing to refresh.")
            return
        from .summarizers import ensure_all_pos

        ensure_all_pos(ctx, self.ext_root, self.client)
        ctx.log("Refreshed external dependency Project Orion Summaries.")

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

    def auto_consolidate_if_needed(self, ctx: Context) -> None:
        """Perform consolidation automatically every few batches to limit duplication."""
        n = self.md["batches_since_last_consolidation"]
        if n >= 3 and n % 3 == 0:
            ctx.log("Auto-consolidating changes...")
            self.cmd_consolidate(ctx)

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
        # Collect affected paths and current contents
        affected = set()
        for ch in pending:
            for it in ch.get("items", []):
                affected.add(it["path"])
        files_payload = []
        for p in sorted(affected):
            content = ""
            abs_path = self.repo_root / p
            if abs_path.exists():
                try:
                    content = read_file(self.repo_root, p)
                except Exception:
                    content = ""
            files_payload.append({"path": p, "content": content})

        # orion: Load Apply system prompt from resources so it can be maintained externally.
        system_text = get_prompt("prompt_apply_system.txt")
        user_text = json.dumps({"changes": pending, "files": files_payload}, ensure_ascii=False)

        # orion: Replace inline JSON Schema with centralized Pydantic model schema.
        response_schema = ApplyResponse.model_json_schema()

        tools = tool_definitions()

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            fn = self.tools_registry.get(name)
            if not fn:
                return {"_meta_error": f"unknown tool {name}", "_args_echo": args}
            return fn(ctx, args)

        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
        ctx.log("Calling model to apply changes...")
        # orion: Migrate :apply flow to Responses API using Pydantic schema; tool runner preserved.
        final_json = self.client.call_responses(
            ctx,
            messages,
            tools,
            response_schema,
            interactive_tool_runner=runner,
            call_type="apply",
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
            write_file(self.repo_root, f.path, f.code)
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

        # Clear conversation and pending changes
        self.storage.clear_history()
        self.history = []
        self.md["pending_changes"] = []
        self.md["batches_since_last_consolidation"] = 0
        self.storage.save_metadata(self.md)
        ctx.log("Cleared conversation history and pending change log.")

        # Post-apply LINE_CAP follow-ups
        added_splits = 0
        for f in parsed.files:
            lines = count_lines(f.code)
            if lines > LINE_CAP:
                path = f.path
                stem = pathlib.Path(path).stem
                ext = pathlib.Path(path).suffix
                target2 = normalize_path(str(pathlib.Path(path).with_name(f"{stem}_part2{ext}")))
                change_id = short_id("split")
                split_change = make_change_spec(
                    change_id,
                    f"Split {path} to meet LINE_CAP",
                    "Split required to reduce line count",
                    [
                        make_change_item(path, "modify", f"Reduce lines from {lines} to below {LINE_CAP}"),
                        make_change_item(target2, "create", "Create second part of split"),
                    ],
                )
                self.md["pending_changes"].append(split_change)
                added_splits += 1
        if added_splits:
            self.storage.save_metadata(self.md)
            ctx.log(f"Added {added_splits} split follow-up change(s) due to LINE_CAP.")

    def handle_user_input(self, ctx: Context, text: str) -> None:
        """
        Handle a line of user input: either execute a command or advance the conversation.
        """
        # Commands
        if text.startswith(":"):
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
            elif cmd == ":clear-changes":
                self.md["pending_changes"] = []
                self.storage.save_metadata(self.md)
                ctx.send_to_user("Cleared all pending changes.")
            elif cmd == ":refresh":
                self.cmd_refresh(ctx)
            elif cmd == ":refresh-deps":
                self.cmd_refresh_deps(ctx)
            elif cmd == ":status":
                self.cmd_status(ctx)
            elif cmd == ":consolidate":
                self.cmd_consolidate(ctx)
            elif cmd == ":quit":
                ctx.send_to_user("Goodbye.")
                import sys

                sys.exit(0)
            else:
                ctx.error_message(f"Unknown command: {cmd}. Type :help for help.")
            return

        # New conversation bootstrap if needed (no history file present)
        self._ensure_bootstrap(ctx,self.md["pending_changes"])

        # Append user turn (raw)
        self.storage.append_raw_message({"type": "message", "role": "user", "content": text})
        self.history.append({"type": "message", "role": "user", "content": text})

        # orion: Load Conversation system prompt from resources so it can be maintained externally.
        system_text = get_prompt("prompt_conversation_system.txt")

        # orion: Replace inline JSON Schema with centralized Pydantic model schema.
        response_schema = ConversationResponse.model_json_schema()

        # Rebuild messages: prepend conversation system prompt, then replay full stored history verbatim
        messages = [{"type": "message", "role": "system", "content": system_text}]
        for h in self.history:
            msg = h.copy()
            if "ts" in msg:
                del msg["ts"]
            messages.append(msg)

        tools = tool_definitions()

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            fn = self.tools_registry.get(name)
            if not fn:
                return {"_meta_error": f"unknown tool {name}", "_args_echo": args}
            return fn(ctx, args)

        # Sink to persist assistant/tool messages during the call
        def sink(msg: Dict[str, Any]) -> None:
            self.storage.append_raw_message(msg)
            self.history.append(msg)

        ctx.log("Calling model for conversation response...")
        # orion: Migrate conversation flow to Responses API; include message sink and set call_type="conversation".
        final_json = self.client.call_responses(
            ctx,
            messages,
            tools,
            response_schema,
            interactive_tool_runner=runner,
            message_sink=sink,
            call_type="conversation",
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
            self.md["pending_changes"].extend(changes)
            self.md["batches_since_last_consolidation"] += 1
            self.storage.save_metadata(self.md)
            self.auto_consolidate_if_needed(ctx)

    def run(self) -> None:
        """Start the interactive REPL loop for Orion."""
        print(f"Orion ready at repo root: {self.repo_root}")
        if self.ext_root:
            print(f"External dependency PD root: {self.ext_root} (flat)")
        else:
            print("External dependency PD root not set (ORION_EXTERNAL_DIR unset or invalid).")
        print("Type :help for commands.")
        ctx = Context()
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            if not text:
                continue
            self.handle_user_input(ctx, text)
