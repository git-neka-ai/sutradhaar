# orion: Slim tool surface to core set and reflective registry. Remove get_file_snippet and get_summary; promote get_file_contents to be used with system_state promotions.

import inspect
import json
import pathlib
from typing import Any, Dict, List, Optional, Callable, get_type_hints

from .context import Context
from .fs import list_repo_paths, normalize_path, read_file, count_lines, read_json, read_jsonl, write_file, now_ts
import yaml
import requests
import re

# -----------------------------
# Reflection utilities and registry
# -----------------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}

_type_map = {
    str: {"type": "string"},
    int: {"type": "integer"},
    bool: {"type": "boolean"},
    float: {"type": "number"},
}


def _json_schema_for_annotation(ann: Any) -> Dict[str, Any]:
    """Map a Python annotation to a simple JSON Schema snippet."""
    # Handle typing.Optional[T] as its inner type
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", ())
    if origin is Optional and args:
        return _json_schema_for_annotation(args[0])
    # Direct map
    return _type_map.get(ann, {"type": "string"})


def _merge_schema(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow-merge override fields into base JSON Schema for a parameter."""
    if not override:
        return base
    out = dict(base)
    out.update({k: v for k, v in override.items() if v is not None})
    return out


def _build_parameters_schema(fn: Callable, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []
    # Skip first arg (ctx)
    params = list(sig.parameters.values())[1:]
    for p in params:
        name = p.name
        ann = hints.get(name, str)
        base = _json_schema_for_annotation(ann)
        ov = (overrides or {}).get(name)
        props[name] = _merge_schema(base, ov)
        if p.default is inspect._empty:
            required.append(name)
    # Inject synthetic reason_for_call for the model only (not required)
    props["reason_for_call"] = {"type": "string", "description": "Short reason the tool is needed (for model traceability)."}
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def tool(name: str, description: str, *, param_overrides: Optional[Dict[str, Dict[str, Any]]] = None):
    """Decorator to register a function as a tool with reflective schema.

    orion: param_overrides allows per-parameter JSON Schema fields like description, pattern, enum, etc.
    """
    def _wrap(fn: Callable):
        _REGISTRY[name] = {
            "fn": fn,
            "name": name,
            "description": description,
            "schema": _build_parameters_schema(fn, overrides=param_overrides),
            "param_overrides": param_overrides or {},
        }
        return fn
    return _wrap


def discover_tools() -> List[Dict[str, Any]]:
    """Return OpenAI tool specs for all registered internal tools, with synthetic reason param."""
    specs: List[Dict[str, Any]] = []
    for name, meta in _REGISTRY.items():
        specs.append({
            "type": "function",
            "name": name,
            "description": meta["description"],
            "parameters": meta["schema"],
        })
    return specs


def list_tool_names() -> List[str]:
    return list(_REGISTRY.keys())


def _coerce_value(val: Any, ann: Any) -> Any:
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", ())
    if origin is Optional and args:
        ann = args[0]
    try:
        if ann is int:
            return int(val)
        if ann is float:
            return float(val)
        if ann is bool:
            if isinstance(val, bool):
                return val
            s = str(val).strip().lower()
            return s in ("1", "true", "yes", "y")
        if ann is str:
            return str(val)
    except Exception:
        return val
    return val


def run_tool(ctx: Context, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to a registered tool, stripping reason_for_call and wrapping the raw payload."""
    meta = _REGISTRY.get(name)
    reason = str(args.get("reason_for_call") or "")
    if not meta:
        return {"reason_for_call": reason, "result": {"_meta_error": f"unknown tool {name}", "_args_echo": args}}
    fn: Callable = meta["fn"]
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    kwargs: Dict[str, Any] = {}
    # Skip first param (ctx)
    for p in list(sig.parameters.values())[1:]:
        nm = p.name
        ann = hints.get(nm, str)
        if nm in args:
            kwargs[nm] = _coerce_value(args[nm], ann)
        elif p.default is inspect._empty:
            return {"reason_for_call": reason, "result": {"_meta_error": f"missing required parameter: {nm}"}}
    try:
        raw = fn(ctx, **kwargs)
    except Exception as e:
        return {"reason_for_call": reason, "result": {"_meta_error": f"tool {name} failed: {e}"}}
    if isinstance(raw, dict):
        payload = raw
    else:
        payload = {"value": raw}
    return {"reason_for_call": reason, "result": payload}


# -----------------------------
# Internal tools (typed, raw returns)
# -----------------------------

# orion: list_paths returns a list of repo-relative paths, optionally filtered by glob.
@tool(name="list_paths", description="List repository files; optionally filter by glob.")
def list_paths(ctx: Context, glob: Optional[str] = None) -> Dict[str, Any]:
    paths = list_repo_paths(ctx.repo_root)
    if glob:
        import fnmatch
        paths = [p for p in paths if fnmatch.fnmatch(p, glob)]
    return {"paths": paths[:2000]}


# orion: get_file_contents returns full file contents and a line_count, or _meta_error.
@tool(name="get_file_contents", description="Return full contents for a file.")
def get_file_contents(ctx: Context, path: str) -> Dict[str, Any]:
    np = normalize_path(path)
    try:
        content = read_file(ctx.repo_root, np)
    except Exception:
        return {"_meta_error": f"Could not read {np}"}
    return {"path": np, "content": content, "line_count": count_lines(content)}


# orion: Rename tool from search_code to search_files to unify tool naming across code and docs; behavior unchanged.
@tool(name="search_files", description="Search files for a substring; returns paths.")
def search_files(ctx: Context, query: str, max_results: int = 100) -> Dict[str, Any]:
    q = (query or "").lower()
    if not q:
        return {"matches": []}
    matches: List[Dict[str, Any]] = []
    for p in list_repo_paths(ctx.repo_root):
        try:
            content = read_file(ctx.repo_root, p)
        except Exception:
            continue
        if q in content.lower():
            matches.append({"path": p})
        if len(matches) >= int(max_results):
            break
    return {"matches": matches}


# orion: ask_user prompts via stdin and returns the user's raw answer (possibly empty).
@tool(name="ask_user", description="Ask the user for a clarification.")
def ask_user(ctx: Context, prompt: str) -> Dict[str, Any]:
    ctx.send_to_user(f"Model asks: {prompt}")
    ctx.send_to_user("Enter a response (or leave empty to cancel): ")
    try:
        ans = input().strip()
    except EOFError:
        ans = ""
    return {"answer": ans}


# orion: Fetch an archived conversation by id and return filtered records.
@tool(name="get_archived_conversation_details", description="Get archived conversation by id; returns {id, filename, records:[{ts, role, content}]} filtered to roles user, assistant, apply.")
def get_archived_conversation_details(ctx: Context, id: str) -> Dict[str, Any]:
    try:
        md = read_json(ctx.repo_root / ".orion" / "orion-metadata.json", {})
    except Exception:
        return {"_meta_error": "could not read metadata"}
    archives = md.get("conversation_archives") or []
    rec = None
    for a in archives:
        if str(a.get("id")) == str(id):
            rec = a
            break
    if not rec:
        return {"_meta_error": "archive id not found"}
    filename = rec.get("filename") or ""
    try:
        # Resolve path safely; enforce it lives under .orion
        p = pathlib.Path(filename)
        if not p.is_absolute():
            p = (ctx.repo_root / filename).resolve()
        if ".orion" not in p.parts:
            return {"_meta_error": "invalid archive path"}
        rows = read_jsonl(p)
    except Exception:
        return {"_meta_error": "could not read archived conversation"}
    records: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("type") != "message":
            continue
        role = row.get("role")
        if role not in ("user", "assistant", "apply"):
            continue
        records.append({
            "ts": row.get("ts"),
            "role": role,
            "content": row.get("content", ""),
        })
    return {"id": rec.get("id"), "filename": filename, "records": records}


# -----------------------------
# TODO list tools (YAML-backed)
# -----------------------------

_TODO_REL_PATH = ".orion/orion-todolist.yaml"


def _todo_dir(ctx: Context) -> pathlib.Path:
    return (ctx.repo_root / ".orion").resolve()


def _load_todos(ctx: Context) -> List[Dict[str, Any]]:
    try:
        text = read_file(ctx.repo_root, _TODO_REL_PATH)
    except Exception:
        return []
    try:
        data = yaml.safe_load(text)
    except Exception:
        data = None
    if not isinstance(data, list):
        return []
    # Normalize items
    out: List[Dict[str, Any]] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        tid = it.get("id")
        txt = it.get("text")
        st = it.get("status") or "open"
        ts = it.get("ts")
        dts = it.get("done_ts") if "done_ts" in it else None
        try:
            tid = int(tid)
        except Exception:
            continue
        out.append({"id": tid, "text": str(txt or ""), "status": str(st), "ts": ts, "done_ts": dts})
    return out


def _save_todos(ctx: Context, items: List[Dict[str, Any]]) -> None:
    # orion: Ensure .orion directory exists before write.
    _todo_dir(ctx).mkdir(parents=True, exist_ok=True)
    dump = yaml.safe_dump(items, sort_keys=False, allow_unicode=True)
    write_file(ctx.repo_root, _TODO_REL_PATH, dump)


@tool(name="list_todos", description="List TODO items from .orion/orion-todolist.yaml; optionally filter by status ('open' or 'done').")
def list_todos(ctx: Context, status: Optional[str] = None) -> Dict[str, Any]:
    items = _load_todos(ctx)
    if status:
        s = str(status).strip().lower()
        items = [it for it in items if it.get("status", "open").lower() == s]
    # Return compact view
    return {"items": [{"id": it["id"], "text": it["text"], "status": it.get("status", "open")} for it in items]}


@tool(name="add_todo", description="Add a TODO with the given text; assigns next numeric id and persists.")
def add_todo(ctx: Context, text: str) -> Dict[str, Any]:
    txt = (text or "").strip()
    if not txt:
        return {"_meta_error": "text required"}
    items = _load_todos(ctx)
    next_id = 1 + max((it["id"] for it in items), default=0)
    now = now_ts()
    rec = {"id": next_id, "text": txt, "status": "open", "ts": now, "done_ts": None}
    items.append(rec)
    _save_todos(ctx, items)
    return {"id": next_id, "text": txt, "status": "open"}


@tool(name="set_todo_status", description="Update status for a TODO id to 'open' or 'done'.")
def set_todo_status(ctx: Context, id: int, status: str) -> Dict[str, Any]:
    try:
        tid = int(id)
    except Exception:
        return {"_meta_error": "invalid id"}
    st = str(status or "").strip().lower()
    if st not in ("open", "done"):
        return {"_meta_error": "invalid status"}
    items = _load_todos(ctx)
    found = False
    now = now_ts()
    for it in items:
        if it.get("id") == tid:
            it["status"] = st
            it["done_ts"] = now if st == "done" else None
            found = True
            break
    if not found:
        return {"_meta_error": "todo id not found"}
    _save_todos(ctx, items)
    return {"id": tid, "status": st}


@tool(name="remove_todo", description="Remove a TODO item by id.")
def remove_todo(ctx: Context, id: int) -> Dict[str, Any]:
    try:
        tid = int(id)
    except Exception:
        return {"_meta_error": "invalid id"}
    items = _load_todos(ctx)
    new_items = [it for it in items if it.get("id") != tid]
    if len(new_items) == len(items):
        return {"_meta_error": "todo id not found"}
    _save_todos(ctx, new_items)
    return {"id": tid, "removed": True}


# -----------------------------
# Downloads cache tools (file-backed metadata + contents)
# -----------------------------

_DOWNLOADS_REL_PATH = ".orion/downloads.yaml"
_MAX_DOWNLOAD_BYTES = 500 * 1024  # 500 KB limit
_CONTENTS_DIR_REL = ".orion/download_contents"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def _contents_dir(ctx: Context) -> pathlib.Path:
    d = (ctx.repo_root / _CONTENTS_DIR_REL).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_download_name(name: str) -> Optional[str]:
    nm = (name or "").strip()
    if not nm:
        return "name required"
    if not _NAME_PATTERN.match(nm):
        return "invalid name; must match ^[A-Za-z0-9._-]{1,120}$"
    if "/" in nm or "\\" in nm or nm in (".", ".."):
        return "invalid name; path separators not allowed"
    return None


def _content_path(ctx: Context, name: str) -> pathlib.Path:
    return _contents_dir(ctx) / name


def _load_downloads(ctx: Context) -> List[Dict[str, Any]]:
    try:
        text = read_file(ctx.repo_root, _DOWNLOADS_REL_PATH)
    except Exception:
        return []
    try:
        data = yaml.safe_load(text)
    except Exception:
        data = None
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        url = str(it.get("url") or "").strip()
        ct = str(it.get("content_type") or "")
        ts = it.get("ts")
        try:
            bcount = int(it.get("bytes") or 0)
        except Exception:
            bcount = 0
        # orion: Preserve legacy inline contents if present for one-time migration in get_download.
        legacy_contents = it.get("contents") if "contents" in it else None
        if name:
            rec = {
                "name": name,
                "url": url,
                "ts": ts,
                "content_type": ct,
                "bytes": bcount,
            }
            if legacy_contents is not None:
                rec["contents"] = str(legacy_contents)
            out.append(rec)
    return out


def _save_downloads(ctx: Context, items: List[Dict[str, Any]]) -> None:
    # orion: Ensure .orion directory exists before write; strip any inline contents from metadata.
    (ctx.repo_root / ".orion").mkdir(parents=True, exist_ok=True)
    scrubbed: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        scrubbed.append({
            "name": it.get("name"),
            "url": it.get("url"),
            "ts": it.get("ts"),
            "content_type": it.get("content_type"),
            "bytes": it.get("bytes"),
        })
    dump = yaml.safe_dump(scrubbed, sort_keys=False, allow_unicode=True)
    write_file(ctx.repo_root, _DOWNLOADS_REL_PATH, dump)


@tool(
    name="download_info",
    description="Fetch a small text URL (≤500 KB) and persist under .orion/downloads.yaml with key 'name'. Overwrites on name collision and returns the saved record.",
    param_overrides={
        "name": {
            "description": "Path-safe key used as the filename under .orion/download_contents. Must match ^[A-Za-z0-9._-]{1,120}$.",
            "pattern": "^[A-Za-z0-9._-]{1,120}$",
        }
    },
)

def download_info(ctx: Context, url: str, name: str) -> Dict[str, Any]:
    # orion: Enforce deterministic, text-only caching with size guard and overwrite-by-name semantics. Now file-backed.
    nm_err = _validate_download_name(name)
    if nm_err:
        return {"_meta_error": nm_err}
    nm = (name or "").strip()
    u = (url or "").strip()
    if not u:
        return {"_meta_error": "url required"}

    headers = {"Accept": "text/*, application/json, application/xml, application/yaml, application/*+json"}
    try:
        resp = requests.get(u, headers=headers, timeout=30, stream=True)
    except Exception as e:
        return {"_meta_error": f"request failed: {e}"}

    try:
        resp.raise_for_status()
    except Exception as e:
        return {"_meta_error": f"http error: {e}"}

    ct = str(resp.headers.get("Content-Type", ""))
    ctl = ct.lower()
    is_textual = ctl.startswith("text/") or any(x in ctl for x in ("json", "xml", "yaml", "csv", "plain"))
    if not is_textual:
        return {"_meta_error": f"non-text content-type: {ct or 'unknown'}"}

    try:
        cl = int(resp.headers.get("Content-Length", "0"))
    except Exception:
        cl = 0
    if cl > 0 and cl > _MAX_DOWNLOAD_BYTES:
        return {"_meta_error": f"content too large: {cl} bytes"}

    total = 0
    buf = bytearray()
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                return {"_meta_error": f"content too large: >{_MAX_DOWNLOAD_BYTES} bytes"}
            buf.extend(chunk)
    except Exception as e:
        return {"_meta_error": f"stream error: {e}"}

    enc = resp.encoding or "utf-8"
    try:
        text = buf.decode(enc, errors="replace")
    except Exception:
        text = buf.decode("utf-8", errors="replace")

    items = _load_downloads(ctx)
    now = now_ts()
    rec = {
        "name": nm,
        "url": u,
        "ts": now,
        "content_type": ct,
        "bytes": len(buf),
    }

    # Overwrite on name collision
    replaced = False
    for i, it in enumerate(items):
        if it.get("name") == nm:
            items[i] = rec
            replaced = True
            break
    if not replaced:
        items.append(rec)

    # Write metadata and contents file
    _save_downloads(ctx, items)
    try:
        _content_path(ctx, nm).write_text(text, encoding="utf-8")
    except Exception as e:
        return {"_meta_error": f"failed to write content file: {e}"}

    # Return full record including contents for convenience
    out = dict(rec)
    out["contents"] = text
    return {"record": out}


@tool(name="get_download", description="Retrieve a cached download by name from .orion/downloads.yaml; returns the full record including contents.")

def get_download(ctx: Context, name: str) -> Dict[str, Any]:
    nm_err = _validate_download_name(name)
    if nm_err:
        return {"_meta_error": nm_err}
    nm = (name or "").strip()
    items = _load_downloads(ctx)
    idx = None
    rec = None
    for i, it in enumerate(items):
        if it.get("name") == nm:
            rec = it
            idx = i
            break
    if rec is None:
        return {"_meta_error": "download not found"}

    p = _content_path(ctx, nm)
    text: Optional[str] = None
    if p.exists():
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = ""
    else:
        # orion: Legacy migration — write inline YAML contents to file if present, then scrub YAML on next save path.
        if "contents" in rec:
            text = str(rec.get("contents") or "")
            try:
                p.write_text(text, encoding="utf-8")
            except Exception:
                pass
            # Remove inline contents from in-memory rec and persist metadata-only YAML
            try:
                del rec["contents"]
                if idx is not None:
                    items[idx] = rec
                    _save_downloads(ctx, items)
            except Exception:
                pass
        else:
            text = ""

    out = dict(rec)
    out["contents"] = text
    return {"record": out}


@tool(name="rename_download", description="Rename a cached download (updates metadata and content file). The new name must be path-safe.")

def rename_download(ctx: Context, old_name: str, new_name: str) -> Dict[str, Any]:
    err_old = _validate_download_name(old_name)
    if err_old:
        return {"_meta_error": err_old}
    err_new = _validate_download_name(new_name)
    if err_new:
        return {"_meta_error": err_new}
    old = old_name.strip()
    new = new_name.strip()

    items = _load_downloads(ctx)
    # Ensure old exists and new does not
    old_idx = None
    for i, it in enumerate(items):
        if it.get("name") == old:
            old_idx = i
            break
    if old_idx is None:
        return {"_meta_error": "old_name not found"}
    if any(it.get("name") == new for it in items):
        return {"_meta_error": "new_name already exists"}

    # Update metadata
    items[old_idx]["name"] = new
    _save_downloads(ctx, items)

    # Rename content file if present
    old_p = _content_path(ctx, old)
    new_p = _content_path(ctx, new)
    try:
        if old_p.exists():
            old_p.replace(new_p)
    except Exception:
        # Non-fatal: metadata already updated
        pass

    return {"old_name": old, "new_name": new, "ok": True}


@tool(name="remove_download", description="Remove a cached download by name (deletes metadata and content file if present).")

def remove_download(ctx: Context, name: str) -> Dict[str, Any]:
    nm_err = _validate_download_name(name)
    if nm_err:
        return {"_meta_error": nm_err}
    nm = name.strip()
    items = _load_downloads(ctx)
    new_items = [it for it in items if it.get("name") != nm]
    if len(new_items) == len(items):
        return {"_meta_error": "download not found"}
    _save_downloads(ctx, new_items)

    # Delete content file (best effort)
    p = _content_path(ctx, nm)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

    return {"id": nm, "removed": True}
