# orion: Slim tool surface to core set and reflective registry. Remove get_file_snippet and get_summary; promote get_file_contents to be used with system_state promotions.

import inspect
import json
import pathlib
from typing import Any, Dict, List, Optional, Callable, get_type_hints

from .context import Context
from .fs import list_repo_paths, normalize_path, read_file, count_lines

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


def _build_parameters_schema(fn: Callable) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []
    # Skip first arg (ctx)
    params = list(sig.parameters.values())[1:]
    for p in params:
        name = p.name
        ann = hints.get(name, str)
        props[name] = _json_schema_for_annotation(ann)
        if p.default is inspect._empty:
            required.append(name)
    # Inject synthetic reason_for_call for the model only (not required)
    props["reason_for_call"] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def tool(name: str, description: str):
    """Decorator to register a function as a tool with reflective schema."""
    def _wrap(fn: Callable):
        _REGISTRY[name] = {
            "fn": fn,
            "name": name,
            "description": description,
            "schema": _build_parameters_schema(fn),
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


# orion: search_code performs a linear substring scan (case-insensitive) across non-ignored files.
@tool(name="search_code", description="Search files for a substring; returns paths.")
def search_code(ctx: Context, query: str, max_results: int = 100) -> Dict[str, Any]:
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
