#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import uuid
import shutil
import pathlib
import hashlib
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import requests
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# -----------------------------
# Configuration and environment
# -----------------------------

# OpenAI env (Chat Completions)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-5")  # OpenAI model id

# Output token budget
MAX_COMPLETION_TOKENS = int(os.environ.get("ORION_MAX_COMPLETION_TOKENS", "48192"))

# Line cap for post-apply check
LINE_CAP = int(os.environ.get("ORION_LINE_CAP", "1000"))

# Conversation history cap on load
CONV_CAP_TURNS = int(os.environ.get("ORION_CONV_CAP_TURNS", "200"))

# Summaries
SUMMARY_MAX_BYTES = int(os.environ.get("ORION_SUMMARY_MAX_BYTES", str(2_000_000)))  # 2 MB default

# External dependency projects (flat directory, no subfolders)
# This is the root folder containing Project Descriptions (PDs) as files,
# and a .orion/ subfolder containing Project Orion Summaries (POS) named <filename>.json
ORION_EXTERNAL_DIR = os.environ.get("ORION_EXTERNAL_DIR", "").strip()

# Optional: TTL in seconds to force POS regeneration even if hash matches (omit/0 to disable)
ORION_DEP_TTL_SEC = int(os.environ.get("ORION_DEP_TTL_SEC", "0") or "0")

# -----------------------------
# Utilities
# -----------------------------

def now_ts() -> float:
    return time.time()

def short_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def normalize_path(p: str) -> str:
    return str(pathlib.Path(p).as_posix())

def read_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: pathlib.Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def append_jsonl(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    return lines

def count_lines(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _safe_abs(repo_root: pathlib.Path, rel: str) -> pathlib.Path:
    abs_path = (repo_root / rel).resolve()
    # ensure inside repo_root
    try:
        abs_path.relative_to(repo_root)
    except Exception:
        raise ValueError(f"Path escapes repo root: {rel}")
    return abs_path

def read_file(repo_root: pathlib.Path, path: str) -> str:
    abs_path = _safe_abs(repo_root, path)
    with abs_path.open("r", encoding="utf-8") as f:
        return f.read()

def write_file(repo_root: pathlib.Path, path: str, content: str) -> None:
    abs_path = _safe_abs(repo_root, path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with abs_path.open("w", encoding="utf-8") as f:
        f.write(content)

def read_bytes(repo_root: pathlib.Path, path: str) -> bytes:
    abs_path = _safe_abs(repo_root, path)
    with abs_path.open("rb") as f:
        return f.read()

def list_repo_paths(repo_root: pathlib.Path) -> List[str]:
    paths = []
    for root, dirs, files in os.walk(repo_root):
        # prune .git and any colocated .orion directories
        dirs[:] = [d for d in dirs if d not in (".git", ".orion")]
        for name in files:
            full = pathlib.Path(root) / name
            rel = os.path.relpath(full, repo_root)
            # Skip Orion internal files (we keep them untracked ideally)
            if rel.endswith("orion-metadata.json") or rel.endswith("orion-conversation.jsonl"):
                continue
            # Skip any file that is within a .orion folder
            parts = pathlib.Path(rel).parts
            if ".orion" in parts:
                continue
            paths.append(normalize_path(rel))
    return sorted(paths)

def colocated_summary_path(repo_root: pathlib.Path, rel_path: str) -> pathlib.Path:
    """
    For a source file at a/b/c.ext, return a/b/.orion/c.ext.json
    """
    rel = pathlib.Path(rel_path)
    summary_dir = rel.parent / ".orion"
    summary_name = rel.name + ".json"
    return (repo_root / summary_dir / summary_name).resolve()

# -----------------------------
# Git helpers for file discovery
# -----------------------------

def run_git(args: List[str], cwd: pathlib.Path) -> Tuple[int, str, str]:
    proc = subprocess.Popen(["git"] + args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def git_list_tracked(repo_root: pathlib.Path) -> List[str]:
    rc, out, err = run_git(["ls-files", "-z"], repo_root)
    if rc != 0:
        return []
    files = [p for p in out.split("\x00") if p]
    return sorted(files)

def git_list_untracked_unignored(repo_root: pathlib.Path) -> List[str]:
    rc, out, err = run_git(["ls-files", "-o", "--exclude-standard", "-z"], repo_root)
    if rc != 0:
        return []
    files = [p for p in out.split("\x00") if p]
    return sorted(files)

def list_all_nonignored_files(repo_root: pathlib.Path) -> List[str]:
    # Prefer Git if available
    rc, out, err = run_git(["rev-parse", "--is-inside-work-tree"], repo_root)
    def is_internal(p: str) -> bool:
        # Skip runtime metadata and any file inside a .orion directory
        if p.endswith("orion-metadata.json") or p.endswith("orion-conversation.jsonl"):
            return True
        return ".orion" in pathlib.Path(p).parts
    if rc == 0 and out.strip() == "true":
        tracked = set(git_list_tracked(repo_root))
        untracked = set(git_list_untracked_unignored(repo_root))
        everything = [normalize_path(p) for p in sorted((tracked | untracked)) if not is_internal(p)]
        return everything
    # Fallback: full walk without .git and skipping .orion
    return list_repo_paths(repo_root)

# -----------------------------
# Strict schemas
# -----------------------------

# Conversation change spec and item
def make_change_spec(id_str: str, title: str, description: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": id_str,
        "title": title,
        "description": description,
        "items": items
    }

def make_change_item(path: str, change_type: str, summary_of_change: str) -> Dict[str, Any]:
    if change_type not in ["modify", "create", "delete", "move", "rename"]:
        raise ValueError("Invalid change_type")
    return {
        "path": normalize_path(path),
        "change_type": change_type,
        "summary_of_change": summary_of_change
    }

def validate_change_specs(changes: Any) -> List[Dict[str, Any]]:
    if not isinstance(changes, list):
        return []
    prepared = []
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        required_keys = ["id", "title", "description", "items"]
        if any(k not in ch for k in required_keys):
            continue
        items = ch.get("items", [])
        if not isinstance(items, list):
            continue
        ok_items = True
        for it in items:
            if not isinstance(it, dict):
                ok_items = False
                break
            if any(ik not in it for ik in ["path", "change_type", "summary_of_change"]):
                ok_items = False
                break
            if it.get("change_type") not in ["modify", "create", "delete", "move", "rename"]:
                ok_items = False
                break
        if not ok_items:
            continue
        for it in items:
            it["path"] = normalize_path(it["path"])
        prepared.append({
            "id": ch["id"],
            "title": ch["title"],
            "description": ch["description"],
            "items": items
        })
    return prepared

# Apply response schema validator
def validate_apply_response(resp: Dict[str, Any]) -> Tuple[bool, str]:
    required_keys = ["mode", "explanation", "files", "issues"]
    for k in required_keys:
        if k not in resp:
            return False, f"ApplyResponse missing required field: {k}"
    if resp["mode"] not in ["ok", "incompatible"]:
        return False, "ApplyResponse.mode must be 'ok' or 'incompatible'"
    if not isinstance(resp["explanation"], str):
        return False, "ApplyResponse.explanation must be string"
    if not isinstance(resp["files"], list) or not isinstance(resp["issues"], list):
        return False, "ApplyResponse.files and issues must be arrays"
    for f in resp["files"]:
        for fk in ["path", "is_new", "code"]:
            if fk not in f:
                return False, f"ApplyResponse.files item missing field: {fk}"
        if not isinstance(f["path"], str):
            return False, "file.path must be string"
        if not isinstance(f["is_new"], bool):
            return False, "file.is_new must be boolean"
        if not isinstance(f["code"], str):
            return False, "file.code must be string"
    for issue in resp["issues"]:
        for ik in ["reason", "paths"]:
            if ik not in issue:
                return False, f"Issue missing field: {ik}"
        if not isinstance(issue["reason"], str):
            return False, "issue.reason must be string"
        if not isinstance(issue["paths"], list) or not all(isinstance(p, str) for p in issue["paths"]):
            return False, "issue.paths must be array of strings"
    return True, ""

# -----------------------------
# Context for user interaction
# -----------------------------
class Context:
    def __init__(self) -> None:
        pass
    pass

    def send_to_user(self, message: str) -> None:
        print(message)
    pass

    def log(self, message: str) -> None:
        print(f"[LOG] {message}")
    pass

    def error_message(self, message: str) -> None:
        print(f"Error: {message}", file=sys.stderr)
    pass
pass


# -----------------------------
# Metadata and conversation persistence
# -----------------------------

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

# -----------------------------
# Tools exposed to the model (local repo)
# -----------------------------

def tool_list_paths(ctx: Context, repo_root: pathlib.Path, args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("glob") or args.get("query")
    paths = list_repo_paths(repo_root)
    if query:
        import fnmatch
        paths = [p for p in paths if fnmatch.fnmatch(p, query)]
    return {"paths": paths[:2000], "_args_echo": args}

def tool_get_file_contents(ctx: Context, repo_root: pathlib.Path, args: Dict[str, Any]) -> Dict[str, Any]:
    path = normalize_path(args.get("path", ""))
    try:
        content = read_file(repo_root, path)
    except Exception:
        return {"_meta_error": f"Could not read {path}", "_args_echo": args}
    return {"path": path, "content": content, "line_count": count_lines(content), "_args_echo": args}

def tool_get_file_snippet(ctx: Context, repo_root: pathlib.Path, args: Dict[str, Any]) -> Dict[str, Any]:
    path = normalize_path(args.get("path", ""))
    start_line = int(args.get("start_line", 1))
    end_line = int(args.get("end_line", start_line + 200))
    try:
        content = read_file(repo_root, path)
    except Exception:
        return {"_meta_error": f"Could not read {path}", "_args_echo": args}
    lines = content.splitlines()
    start_line = max(1, start_line)
    end_line = min(len(lines), end_line if end_line >= start_line else start_line)
    snippet = "\n".join(lines[start_line - 1:end_line])
    return {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "content": snippet,
        "_args_echo": args
    }

def tool_get_summary(ctx: Context, orion: "Orion", args: Dict[str, Any]) -> Dict[str, Any]:
    # Return summary for a file from the colocated .orion path only.
    path = normalize_path(args.get("path", ""))
    try:
        sp = colocated_summary_path(orion.repo_root, path)
        if sp.exists():
            with sp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {"path": path, "summary": data, "_args_echo": args, "_meta_note": "summary returned (colocated)"}
        return {"_meta_error": f"no summary available for {path}", "_args_echo": args}
    except Exception as e:
        return {"_meta_error": f"summary error for {path}: {e}", "_args_echo": args}

def tool_search_code(ctx: Context, repo_root: pathlib.Path, args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query", "") or "")
    if not query:
        return {"matches": [], "_args_echo": args}
    matches = []
    paths = list_repo_paths(repo_root)
    for p in paths:
        try:
            content = read_file(repo_root, p)
        except Exception:
            continue
        lower = content.lower()
        q = query.lower()
        if q in lower:
            matches.append({"path": p})
        if len(matches) >= int(args.get("max_results", 100)):
            break
    return {"matches": matches, "_args_echo": args}

def tool_ask_user(ctx:Context, args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = args.get("prompt") or args.get("question") or "Model requests input:"
    ctx.send_to_user(f"Model asks: {prompt}")
    ctx.send_to_user("Enter a response (or leave empty to cancel): ")
    try:
        ans = input().strip()
    except EOFError:
        ans = ""
    return {"answer": ans, "_args_echo": args}

# -----------------------------
# External dependency projects (flat) — PD & POS helpers
# -----------------------------

def ext_dir_valid(root: Optional[str]) -> Optional[pathlib.Path]:
    if not root:
        return None
    p = pathlib.Path(root).resolve()
    if not p.exists() or not p.is_dir():
        return None
    return p

def ext_orion_dir(ext_root: pathlib.Path) -> pathlib.Path:
    return (ext_root / ".orion").resolve()

def list_project_descriptions(ext_root: pathlib.Path) -> List[str]:
    """
    List PD filenames (flat). Excludes the .orion directory and any subdirectories.
    """
    items: List[str] = []
    for entry in os.scandir(ext_root):
        if entry.is_dir():
            if entry.name == ".orion":
                continue
            # The external folder is assumed flat; skip any other directories if present.
            continue
        if entry.is_file():
            items.append(entry.name)
    return sorted(items)

def pos_path_for_filename(ext_root: pathlib.Path, filename: str) -> pathlib.Path:
    return (ext_orion_dir(ext_root) / f"{filename}.json").resolve()

def read_pos(ext_root: pathlib.Path, filename: str) -> Optional[Dict[str, Any]]:
    p = pos_path_for_filename(ext_root, filename)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_pos(ext_root: pathlib.Path, filename: str, obj: Dict[str, Any]) -> None:
    p = pos_path_for_filename(ext_root, filename)
    write_json(p, obj)

def hash_pd(ext_root: pathlib.Path, filename: str) -> Optional[str]:
    pd_path = (ext_root / filename).resolve()
    if not pd_path.exists() or not pd_path.is_file():
        return None
    try:
        data = pd_path.read_bytes()
    except Exception:
        return None
    return sha256_bytes(data)

# -----------------------------
# Summarizer (local files, colocated .orion/* summaries)
# -----------------------------

class CustomBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class FileSummary(CustomBaseModel):
    # Compact keys
    v: int = Field(..., description="Schema version - use 1")
    p: str = Field(..., description="File path")
    b: str = Field(..., description="Working-tree sha256 digest")
    l: str = Field(..., description="Language of the file")
    lc: int = Field(..., description="Line count")
    sz: int = Field(..., description="Size in bytes")
    ex: List[str] = Field(..., description="List of exports/symbols")
    im: List[str] = Field(..., description="List of imports/dependencies")
    fx: List[str] = Field(..., description="List of functions")
    cl: List[str] = Field(..., description="List of classes")
    io: List[str] = Field(..., description="List of side effects")
    cfg: List[str] = Field(..., description="List of configs/environment variables used")
    r: List[str] = Field(..., description="List of risks/constraints")
    sm: List[str] = Field(..., description="List of notes on safe-to-modify areas")

def summarizer_system_text(line_cap: int) -> str:
    return (
        "You are Orion's file summarizer. Produce a highly compressed, machine-oriented summary JSON for a single file.\n"
        "- Minimize tokens and be lossless for downstream planning.\n"
        "- Do NOT include code; summarize structure, symbols, imports, configs, side-effects, risks, and safe-to-modify notes.\n"
        "- Use the strict JSON schema provided.\n"
        "- language keys: py, ts, tsx, js, jsx, sh.\n"
        f"- Note: Hard limit: no file may exceed {line_cap} lines in edits (for context only).\n"
    )

def guess_language(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    return {
        ".py": "py",
        ".ts": "ts",
        ".tsx": "tsx",
        ".js": "js",
        ".jsx": "jsx",
        ".sh": "sh",
        ".bash": "sh",
        ".zsh": "sh",
    }.get(ext, "txt")

def summarize_file(ctx: Context, client: "ChatCompletionsClient", repo_root: pathlib.Path, rel_path: str) -> Optional[Dict[str, Any]]:
    abs_path = _safe_abs(repo_root, rel_path)
    # Skip very large files
    try:
        size_bytes = abs_path.stat().st_size
        if size_bytes > SUMMARY_MAX_BYTES:
            ctx.log(f"Skipping summary for large file (> {SUMMARY_MAX_BYTES} bytes): {rel_path}")
            return None
    except Exception:
        pass

    try:
        data = abs_path.read_bytes()
    except Exception as e:
        ctx.error_message(f"Failed to read {rel_path}: {e}")
        return None

    digest = sha256_bytes(data)
    text = data.decode("utf-8", errors="replace")
    info = {
        "path": rel_path,
        "language": guess_language(rel_path),
        "line_count": count_lines(text),
        "size_bytes": len(data),
        "sha256": digest,
    }

    system_txt = summarizer_system_text(LINE_CAP)
    user_txt = json.dumps({"info": info, "content": text}, ensure_ascii=False)
    schema = FileSummary.model_json_schema()

    messages = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt}
    ]

    ctx.log(f"Summarizing file: {rel_path} (size: {len(data)} bytes, lines: {info['line_count']})")
    final_json = client.call_responses(ctx,messages=messages, tools=None, response_schema=schema, max_completion_tokens=None, interactive_tool_runner=None, reasoning_effort="minimal")

    try:
        fs = FileSummary.model_validate(final_json)
    except ValidationError as ve:
        ctx.error_message(f"Summary validation failed for {rel_path}: {ve}")
        return None

    obj = fs.model_dump()
    obj["p"] = rel_path
    obj["b"] = digest

    sp = colocated_summary_path(repo_root, rel_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    return obj

# -----------------------------
# External PD → POS summarizer (flat)
# -----------------------------

class ProjectOrionSummary(CustomBaseModel):
    # Minimal enforced POS schema (token-optimized)
    v: int = Field(..., description="Schema version - use 1")
    f: str = Field(..., description="Project Description filename")
    h: str = Field(..., description="sha256 digest of PD raw bytes")
    ex: List[Any] = Field(..., description="Exported surface (classes/functions/config handles)")
    u: List[str] = Field(..., description="Usage notes/hints (bullets; no code)")
    r: List[str] = Field(..., description="Risks/constraints (bullets)")
    # Optional: language list, coordinates, counts, etc., if present
    # Keep compact by allowing extra only via explicit fields above.

def pd_summarizer_system_text() -> str:
    return (
        "You are Orion's Project Description summarizer. Produce a compact, machine-oriented project summary JSON.\n"
        "- Do NOT include code snippets; write textual bullet hints only.\n"
        "- Capture exported surface (classes/functions/config keys) and short usage notes and risks.\n"
        "- Use the strict JSON schema provided.\n"
        "- Keep tokens minimal but sufficient for planning.\n"
    )

def summarize_project_description(ctx:Context, client: "ChatCompletionsClient", ext_root: pathlib.Path, filename: str, pd_hash: str) -> Optional[Dict[str, Any]]:
    pd_path = (ext_root / filename).resolve()
    try:
        data = pd_path.read_bytes()
    except Exception as e:
        ctx.error_message(f"Failed to read PD {filename}: {e}")
        return None

    text = data.decode("utf-8", errors="replace")
    info = {"filename": filename, "size_bytes": len(data), "sha256": pd_hash}

    system_txt = pd_summarizer_system_text()
    user_txt = json.dumps({"info": info, "content": text}, ensure_ascii=False)
    schema = ProjectOrionSummary.model_json_schema()

    messages = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt}
    ]

    ctx.log(f"Summarizing Project Description: {filename} (size: {len(data)} bytes)")
    final_json = client.call_chatcompletions(ctx, messages=messages, tools=None, response_schema=schema, max_completion_tokens=None, interactive_tool_runner=None)

    try:
        ps = ProjectOrionSummary.model_validate(final_json)
    except ValidationError as ve:
        ctx.error_message(f"Project summary validation failed for {filename}: {ve}")
        return None

    obj = ps.model_dump()
    obj["f"] = filename
    obj["h"] = pd_hash
    return obj

def ensure_pos(ctx: Context, ext_root: pathlib.Path, filename: str, client: "ChatCompletionsClient") -> Optional[Dict[str, Any]]:
    """
    Ensure a POS exists and matches PD hash. Regenerate if missing/hash-mismatch/stale TTL.
    Returns the POS dict or None on failure.
    """
    pd_hash = hash_pd(ext_root, filename)
    if not pd_hash:
        ctx.log(f"Skipping PD {filename}: unreadable.")
        return None

    pos = read_pos(ext_root, filename)
    need_regen = False

    if pos is None:
        need_regen = True
    else:
        # Basic field presence
        if not isinstance(pos, dict) or "h" not in pos or "f" not in pos or "v" not in pos:
            need_regen = True
        # Hash mismatch
        elif pos.get("h") != pd_hash:
            need_regen = True
        # TTL, if configured
        elif ORION_DEP_TTL_SEC > 0:
            built_ts = pos.get("_built_ts") or 0
            try:
                built_ts = float(built_ts)
            except Exception:
                built_ts = 0.0
            if built_ts <= 0 or (now_ts() - built_ts) > ORION_DEP_TTL_SEC:
                need_regen = True

    if need_regen:
        new_pos = summarize_project_description(ctx,client, ext_root, filename, pd_hash)
        if new_pos is None:
            return None
        # Attach build metadata (non-schema)
        new_pos["_built_ts"] = now_ts()
        write_pos(ext_root, filename, new_pos)
        return new_pos

    return pos

def ensure_all_pos(ctx: Context, ext_root: pathlib.Path, client: "ChatCompletionsClient") -> List[Dict[str, Any]]:
    """
    Ensure POS for all PDs (flat). Returns a list of heads for bootstrap.
    """
    heads: List[Dict[str, Any]] = []
    items = list_project_descriptions(ext_root)
    if not items:
        return heads
    # Make sure .orion exists
    ext_orion_dir(ext_root).mkdir(parents=True, exist_ok=True)
    for fn in items:
        pos = ensure_pos(ctx,ext_root, fn, client)
        has_summary = pos is not None
        ex_count = int(len(pos.get("ex", []))) if has_summary else 0
        heads.append({"filename": fn, "has_summary": has_summary, "ex_count": ex_count})
    # Optional GC: remove stale POS files without matching PD
    expected = {f"{fn}.json" for fn in items}
    try:
        for entry in os.scandir(ext_orion_dir(ext_root)):
            if not entry.is_file():
                continue
            if entry.name not in expected:
                # stale POS; safe to ignore or remove. We'll just ignore by default.
                pass
    except FileNotFoundError:
        pass
    return heads

# -----------------------------
# Chat Completions client with tool loop
# -----------------------------

def dumpHttpFile(file: str, url: str, method: str, headers: Dict[str, str], obj: Any) -> None:
    try:
        json_str = json.dumps(obj, indent=2, ensure_ascii=False)
        with open(file, "w", encoding="utf-8") as f:
            f.write(f"{method.upper()} {url}\n")
            for key, value in headers.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write(json_str)
        print(f"HTTP request successfully dumped to {file}")
    except TypeError as e:
        print(f"Error: The object could not be serialized to JSON. Details: {e}")
    except OSError as e:
        print(f"Error: Could not write to file {file}. Details: {e}")

class ChatCompletionsClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not (api_key and model):
            raise RuntimeError("OpenAI env missing. Set OPENAI_API_KEY and AI_MODEL.")
        self.base_url = "https://api.openai.com/v1"
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def call_responses(
        self,
        ctx: Context,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_schema: Dict[str, Any],
        max_completion_tokens: Optional[int] = None,
        interactive_tool_runner=None,
        message_sink=None,
        reasoning_effort: Optional[str] = "minimal",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement that uses the Responses API instead of Chat Completions,
        while preserving the same signature and semantics expected by Orion.
        """
        # Build a stable, normalized sink
        def _sink(msg: Dict[str, Any]) -> None:
            if message_sink:
                message_sink(msg)

        model = model or self.model
        # Normalize request payload for the Responses API
        url = f"{self.base_url}/responses"  # /v1/responses
        max_output_tokens = max_completion_tokens or MAX_COMPLETION_TOKENS

        # We’ll evolve a local, mutable message buffer just like before
        local_messages = list(messages)
        have_tools = bool(tools)
        max_tool_turns = 12
        turns = 0

        def _make_payload() -> Dict[str, Any]:
            # Keep the same json_schema format you already use
            payload = {
                "model": model,
                "input": local_messages,  # same role/content items you were sending in `messages`
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "OrionSchema",
                        "schema": response_schema,
                        "strict": True,
                    }
                },
                "max_output_tokens": max_output_tokens,
                "reasoning": {
                    "effort": reasoning_effort or "minimal"
                }
            }
            if have_tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            return payload

        def _extract_msg_obj(resp_obj: Dict[str, Any]) -> Dict[str, Any]:
            """
            Normalize a Responses API result into a {content: str|None, tool_calls: list} 'message'
            compatible with the rest of Orion's logic.
            """
            # Primary path: Responses returns an `output` list of segments.
            output = resp_obj.get("output")
            content_chunks: List[str] = []
            tool_calls = []

            if output and isinstance(output, list):
                for o in output:
                    if not isinstance(o, dict):
                        continue
                    _otype = o.get("type")
                    if _otype == "message":
                        _ct = o.get("content")  
                        if isinstance(_ct, str):
                                content_chunks.append(_ct)
                        elif isinstance(_ct, list):
                            for item in _ct:
                                if isinstance(item, str):
                                    content_chunks.append(item)
                                elif isinstance(item, dict) and item.get("type","") == "output_text":
                                    content_chunks.append(item.get("text",""))
                                pass
                            pass
                        pass
                    elif _otype == "tool_call":
                        tool_calls.append(o)
                    pass
                pass
            pass
            # Final normalized message object
            content = "\n".join([c for c in content_chunks if isinstance(c, str)]) if content_chunks else None
            return {"content": content, "tool_calls": tool_calls}

        while True:
            ctx.log(f"Calling POST (tools={len(tools) if tools else 0})")
            r = self.session.post(url, json=_make_payload(), timeout=240)
            if r.status_code != 200:
                raise RuntimeError(f"Responses API error {r.status_code}: {r.text[:2000]}")

            resp = r.json()

            # Normalize the Responses payload into a chat-like message object
            msg_obj = _extract_msg_obj(resp)

            tool_calls = msg_obj.get("tool_calls") or []
            if tool_calls:
                # Record assistant turn with tool_calls (same as before)
                _sink({"role": "assistant", "content": msg_obj.get("content", None), "tool_calls": tool_calls})

                if interactive_tool_runner is None:
                    raise RuntimeError("Tool requested but no interactive_tool_runner provided.")

                turns += 1
                if turns > max_tool_turns:
                    raise RuntimeError("Exceeded max tool-call turns; aborting.")

                # Execute each tool call and append tool results to the conversation
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    args_text = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        args = {}
                    # Run the tool
                    tool_output = interactive_tool_runner(name, args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_output, ensure_ascii=False)
                    }
                    local_messages.append(tool_msg)
                    _sink(tool_msg)

                # Also append the assistant turn (with tool_calls) to the history we’ll resend
                local_messages.append({
                    "role": "assistant",
                    "content": msg_obj.get("content", None),
                    "tool_calls": tool_calls
                })
                # Loop back to let the model continue after tool outputs
                continue

            # No tool calls → this should be the final, strict JSON text per your schema
            final_text = msg_obj.get("content") or ""
            _sink({"role": "assistant", "content": final_text})

            try:
                final_json = json.loads(final_text)
            except Exception as e:
                # Provide some debugging context if the model didn't honor json_schema
                raise RuntimeError(
                    f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}"
                )
            return final_json

    def call_chatcompletions(
        self,
        ctx: Context,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_schema: Dict[str, Any],
        max_completion_tokens: Optional[int] = None,
        interactive_tool_runner=None,
        message_sink=None
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "OrionSchema",
                    "schema": response_schema,
                    "strict": True
                }
            },
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload["max_completion_tokens"] = max_completion_tokens or MAX_COMPLETION_TOKENS

        url = self.chat_url()

        local_messages = list(messages)
        max_tool_turns = 12
        turns = 0

        def _sink(msg: Dict[str, Any]) -> None:
            if message_sink:
                message_sink(msg)

        while True:
            ctx.log(f"Calling POST (tools={len(tools) if tools else 0})")
            r = self.session.post(
                url,
                json={
                    "model": payload["model"],
                    "messages": local_messages,
                    "response_format": payload["response_format"],
                    **({"tools": payload["tools"], "tool_choice": payload.get("tool_choice")} if tools else {}),
                    "max_completion_tokens": payload["max_completion_tokens"]
                },
                timeout=480
            )
            if r.status_code != 200:
                raise RuntimeError(f"Chat Completions API error {r.status_code}: {r.text[:2000]}")
            resp = r.json()
            choice = (resp.get("choices") or [{}])[0]
            msg_obj = choice.get("message", {}) or {}

            tool_calls = msg_obj.get("tool_calls") or []
            if tool_calls:
                local_messages.append({
                    "role": "assistant",
                    "content": msg_obj.get("content", None),
                    "tool_calls": tool_calls
                })
                _sink({"role": "assistant", "content": msg_obj.get("content", None), "tool_calls": tool_calls})
                if interactive_tool_runner is None:
                    raise RuntimeError("Tool requested but no interactive_tool_runner provided.")
                turns += 1
                if turns > max_tool_turns:
                    raise RuntimeError("Exceeded max tool-call turns; aborting.")
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    args_text = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        args = {}
                    tool_output = interactive_tool_runner(name, args)
                    local_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_output, ensure_ascii=False)
                    })
                    _sink({"role": "tool", "tool_call_id": tc_id, "content": json.dumps(tool_output, ensure_ascii=False)})
                continue

            final_text = msg_obj.get("content") or ""
            _sink({"role": "assistant", "content": final_text})
            try:
                final_json = json.loads(final_text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}")
            return final_json

# -----------------------------
# Orion main class
# -----------------------------

def tool_definitions() -> List[Dict[str, Any]]:
    # Local repo tools + external dependency tools
    defs = [
        {
            "type": "function",
            "function": {
                "name": "list_paths",
                "description": "List repository files; optionally filter by glob.",
                "parameters": {
                    "type": "object",
                    "properties": {"glob": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_file_contents",
                "description": "Return full contents for a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_file_snippet",
                "description": "Return a line-range snippet for a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"}
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_summary",
                "description": "Return a brief machine-oriented summary for a local repo file, if available.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search files for a substring; returns paths.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user for a clarification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"}
                    },
                    "required": ["prompt"],
                    "additionalProperties": False
                }
            }
        }
    ]

    # External dependency tools (flat directory)
    defs.extend([
        {
            "type": "function",
            "function": {
                "name": "list_project_descriptions",
                "description": "List dependency Project Descriptions (filenames) from the external directory.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_project_orion_summary",
                "description": "Return the Project Orion Summary (POS) for a given PD filename; regenerates if stale.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"}
                    },
                    "required": ["filename"],
                    "additionalProperties": False
                }
            }
        }
    ])
    return defs

class Orion:
    def __init__(self, repo_root: pathlib.Path) -> None:
        self.repo_root = repo_root.resolve()
        self.storage = Storage(self.repo_root)
        self.md = self.storage.load_metadata()
        self.history = self.storage.load_history()
        self.client = ChatCompletionsClient(OPENAI_API_KEY, AI_MODEL)

        # External dependency root (flat)
        self.ext_root: Optional[pathlib.Path] = ext_dir_valid(ORION_EXTERNAL_DIR)

        # Build tools registry for interactive execution
        self.tools_registry = {
            # Local repo
            "list_paths": lambda ctx, args: tool_list_paths(ctx, self.repo_root, args),
            "get_file_contents": lambda ctx, args: tool_get_file_contents(ctx, self.repo_root, args),
            "get_file_snippet": lambda ctx, args: tool_get_file_snippet(ctx, self.repo_root, args),
            "get_summary": lambda ctx, args: tool_get_summary(ctx, self, args),
            "search_code": lambda ctx, args: tool_search_code(ctx, self.repo_root, args),
            "ask_user": lambda ctx, args: tool_ask_user(ctx, args),
            # External dependencies (flat) #TODO pass ctx
            "list_project_descriptions": lambda ctx, args: self._tool_list_pds(ctx, args),
            "get_project_orion_summary": lambda ctx, args: self._tool_get_pos(ctx, args)
        }

    # ---------- External tools (flat) ----------

    def _tool_list_pds(self, ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ext_root:
            return {"_meta_error": "ORION_EXTERNAL_DIR not set or invalid.", "_args_echo": args}
        try:
            items = list_project_descriptions(self.ext_root)
            return {"filenames": items, "_args_echo": args}
        except Exception as e:
            return {"_meta_error": f"list_pds failed: {e}", "_args_echo": args}

    def _tool_get_pos(self, ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ext_root:
            return {"_meta_error": "ORION_EXTERNAL_DIR not set or invalid.", "_args_echo": args}
        filename = str(args.get("filename") or "")
        if not filename:
            return {"_meta_error": "filename required", "_args_echo": args}
        try:
            pos = ensure_pos(ctx, self.ext_root, filename, self.client)
            if not pos:
                return {"_meta_error": f"no POS available for {filename}", "_args_echo": args}
            return {"filename": filename, "summary": pos, "_args_echo": args}
        except Exception as e:
            return {"_meta_error": f"get_pos failed for {filename}: {e}", "_args_echo": args}

    # ---------- Bootstrap helpers ----------

    def _build_bootstrap_message(self, ctx: Context) -> Dict[str, Any]:
        """
        Build a one-time bootstrap system message with the complete file list and colocated summaries (if present).
        Also includes external dependency heads (flat) if ORION_EXTERNAL_DIR is set.
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
            "note": "This is the complete list of files in the repository at this time; treat it as authoritative.",
            "complete_list": True,
            "files": items
        }

        # External dependency heads (filenames + small counts)
        if self.ext_root:
            try:
                heads = ensure_all_pos(ctx,self.ext_root, self.client)
            except Exception:
                heads = []
            payload["dependency_projects"] = heads

        return {"role": "system", "content": json.dumps(payload, ensure_ascii=False)}

    def _ensure_bootstrap_if_new_conversation(self, ctx: Context) -> None:
        """
        If there is no conversation history file, refresh summaries and persist a bootstrap message.
        Also ensures external POS are created/validated on first run.
        """
        if not self.storage.conv_file.exists():
            # Refresh local summaries first (small repo assumption)
            self._refresh_summaries(ctx)
            # Ensure external POS (if configured)
            if self.ext_root:
                ensure_all_pos(ctx, self.ext_root, self.client)
            bootstrap_msg = self._build_bootstrap_message(ctx)
            self.storage.append_raw_message(bootstrap_msg)
            self.history.append(bootstrap_msg)

    # ---------- Commands ----------

    def cmd_help(self, ctx: Context) -> None:
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
        before = len(self.md["pending_changes"])
        self.md["pending_changes"] = [c for c in self.md["pending_changes"] if c.get("id") != change_id]
        after = len(self.md["pending_changes"])
        if before == after:
            ctx.send_to_user(f"No change with id {change_id} found.")
        else:
            self.storage.save_metadata(self.md)
            ctx.send_to_user(f"Discarded change {change_id}.")

    def _refresh_summaries(self, ctx: Context, only_paths: Optional[List[str]] = None) -> None:
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

        changed = []
        skipped = 0
        created = 0
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
            need = (prev != digest)
            if need:
                res = summarize_file(ctx, self.client, self.repo_root, rel)
                if res:
                    path_to_digest[rel] = digest
                    changed.append(rel)
                    created += 1
                    self.storage.save_metadata(self.md)
                pass
            pass
        pass

        if only_paths is not None:
            ctx.log(f"Refreshed summaries for {len(changed)} file(s); skipped {skipped} large file(s).")
        else:
            ctx.log(f"Summarization complete. Updated {len(changed)} file(s); skipped {skipped} large file(s).")

    def cmd_refresh(self, ctx: Context) -> None:
        self._refresh_summaries(ctx)
        # Also refresh external dependency POS if configured
        if self.ext_root:
            ensure_all_pos(ctx, self.ext_root, self.client)
            ctx.log("Refreshed external dependency Project Orion Summaries.")

    def cmd_refresh_deps(self, ctx: Context) -> None:
        if not self.ext_root:
            ctx.log("ORION_EXTERNAL_DIR not set or invalid; nothing to refresh.")
            return
        ensure_all_pos(ctx, self.ext_root, self.client)
        ctx.log("Refreshed external dependency Project Orion Summaries.")

    def cmd_status(self, ctx: Context) -> None:
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
        n = self.md["batches_since_last_consolidation"]
        if n >= 3 and n % 3 == 0:
            ctx.log("Auto-consolidating changes...")
            self.cmd_consolidate(ctx)

    def cmd_apply(self, ctx: Context) -> None:
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

        system_text = (
            "You are Orion Apply. You will receive the accumulated change specs and the current contents of affected files. "
            "If you need more context, call tools. Then return a strict JSON object with fields: "
            "mode ('ok'|'incompatible'), explanation (string), files (array of {path,is_new,code}), issues (array of {reason,paths}). "
            "Do not include hidden reasoning. For non-applicable arrays, return empty arrays."
            "Remember to add a comment just before every change you make explaining the reasoning behind the change. Start these comments with `orion:` . If you find a comment already exists at that point update it to reflect your reasoning."
        )
        user_text = json.dumps({
            "changes": pending,
            "files": files_payload
        }, ensure_ascii=False)

        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["ok", "incompatible"]},
                "explanation": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "is_new": {"type": "boolean"},
                            "code": {"type": "string"}
                        },
                        "required": ["path", "is_new", "code"]
                    }
                },
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "reason": {"type": "string"},
                            "paths": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["reason", "paths"]
                    }
                }
            },
            "required": ["mode", "explanation", "files", "issues"]
        }

        tools = tool_definitions()

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            fn = self.tools_registry.get(name)
            if not fn:
                return {"_meta_error": f"unknown tool {name}", "_args_echo": args}
            return fn(ctx,args)

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ]
        ctx.log("Calling model to apply changes...")
        final_json = self.client.call_chatcompletions(ctx,messages, tools, response_schema, interactive_tool_runner=runner)
        ok, err = validate_apply_response(final_json)
        if not ok:
            ctx.error_message(f"Apply failed: invalid response from model: {err}")
            return

        if final_json["mode"] == "incompatible":
            ctx.send_to_user("Model reported incompatibility:")
            for issue in final_json["issues"]:
                ctx.send_to_user(f"- {issue['reason']}: {', '.join(issue['paths'])}")
            ctx.send_to_user(f"Explanation: {final_json['explanation']}")
            return

        # Write files
        written_paths: List[str] = []
        for f in final_json["files"]:
            write_file(self.repo_root, f["path"], f["code"])
            written_paths.append(f["path"])

        # Commit log entry
        self.md["plan_state"]["commit_log"].append({
            "id": short_id("commit"),
            "ts": now_ts(),
            "paths": [f["path"] for f in final_json["files"]],
            "explanation": final_json["explanation"]
        })
        self.storage.save_metadata(self.md)
        ctx.log(f"Wrote {len(final_json['files'])} files. Explanation: {final_json['explanation']}")

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
        for f in final_json["files"]:
            lines = count_lines(f["code"])
            if lines > LINE_CAP:
                path = f["path"]
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
                        make_change_item(target2, "create", "Create second part of split")
                    ]
                )
                self.md["pending_changes"].append(split_change)
                added_splits += 1
        if added_splits:
            self.storage.save_metadata(self.md)
            ctx.log(f"Added {added_splits} split follow-up change(s) due to LINE_CAP.")

    def handle_user_input(self, ctx: Context, text: str) -> None:
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
                    self.cmd_discard_change(ctx,parts[1])
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
                sys.exit(0)
            else:
                ctx.error_message(f"Unknown command: {cmd}. Type :help for help.")
            return

        # New conversation bootstrap if needed (no history file present)
        self._ensure_bootstrap_if_new_conversation(ctx)

        # Append user turn (raw)
        self.storage.append_raw_message({"role": "user", "content": text})
        self.history.append({"role": "user", "content": text})

        # System prompt
        system_text = (
            "You are Orion Conversation. Respond with a strict JSON object:\n"
            "{ assistant_message: string, changes: ChangeSpec[] }\n"
            "ChangeSpec: { id, title, description, items[] }. ChangeItemSpec: { path, change_type, summary_of_change }.\n"
            "All fields required; arrays may be empty when not applicable."
        )

        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "assistant_message": {"type": "string"},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "path": {"type": "string"},
                                        "change_type": {"type": "string", "enum": ["modify", "create", "delete", "move", "rename"]},
                                        "summary_of_change": {"type": "string"}
                                    },
                                    "required": ["path", "change_type", "summary_of_change"]
                                }
                            }
                        },
                        "required": ["id", "title", "description", "items"]
                    }
                }
            },
            "required": ["assistant_message", "changes"]
        }

        # Rebuild messages: prepend conversation system prompt, then replay full stored history verbatim
        messages = [{"role": "system", "content": system_text}]
        for h in self.history:
            msg = {"role": h["role"]}
            if "content" in h:
                msg["content"] = h["content"]
            if "tool_calls" in h:
                msg["tool_calls"] = h["tool_calls"]
            if "tool_call_id" in h:
                msg["tool_call_id"] = h["tool_call_id"]
            messages.append(msg)

        tools = tool_definitions()

        def runner(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            fn = self.tools_registry.get(name)
            if not fn:
                return {"_meta_error": f"unknown tool {name}", "_args_echo": args}
            return fn(ctx,args)

        # Sink to persist assistant/tool messages during the call
        def sink(msg: Dict[str, Any]) -> None:
            self.storage.append_raw_message(msg)
            self.history.append(msg)

        ctx.log("Calling model for conversation response...")
        final_json = self.client.call_chatcompletions(ctx,messages, tools, response_schema, interactive_tool_runner=runner, message_sink=sink)

        if "assistant_message" not in final_json or "changes" not in final_json:
            ctx.log("Model returned invalid conversation response.")
            return
        assistant_msg = final_json["assistant_message"]
        changes = validate_change_specs(final_json["changes"])
        ctx.send_to_user(assistant_msg)
        self.storage.append_history("assistant", assistant_msg, {"changes_count": len(changes)})
        self.history.append({"role": "assistant", "content": assistant_msg})

        if changes:
            self.md["pending_changes"].extend(changes)
            self.md["batches_since_last_consolidation"] += 1
            self.storage.save_metadata(self.md)
            self.auto_consolidate_if_needed(ctx)

    def run(self) -> None:
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
            self.handle_user_input(ctx,text)

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("Usage: orion.py [repo_root]")
        print("Environment: OPENAI_API_KEY, AI_MODEL")
        print("Optional External Dependencies (flat): ORION_EXTERNAL_DIR, ORION_DEP_TTL_SEC")
        return
    repo_root = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 and not sys.argv[1].startswith("--") else pathlib.Path(".").resolve()
    Orion(repo_root).run()

if __name__ == "__main__":
    main()
