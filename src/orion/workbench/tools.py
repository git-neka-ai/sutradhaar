# orion: Collected generic repo tools (list, read, snippet, search, ask_user) and a get_summary wrapper into one module to simplify the tool registry wiring in Orion. Added docstrings and comments on bounds, performance, and path safety. Standardized all tool returns to a uniform envelope {"reason_for_call": <incoming>, "result": <payload>} to make downstream handling consistent.

import json
import pathlib
from typing import Any, Dict, List

from .context import Context
from .fs import list_repo_paths, normalize_path, read_file, colocated_summary_path, count_lines

# orion: Helper to wrap tool payloads in the standard envelope; keeps reason_for_call explicit for auditing and prompts.
def _wrap_tool_result(args: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"reason_for_call": str(args.get("reason_for_call") or ""), "result": payload}


# orion: Add docstring and mention optional glob filtering.
def tool_list_paths(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a list of repo-relative paths, optionally filtered by a glob/query pattern."""
    query = args.get("glob") or args.get("query")
    paths = list_repo_paths(ctx.repo_root)
    if query:
        import fnmatch
        paths = [p for p in paths if fnmatch.fnmatch(p, query)]
    # orion: Return in standardized envelope; omit top-level _args_echo per new contract.
    return _wrap_tool_result(args, {"paths": paths[:2000]})


# orion: Document path normalization and return shape (includes line_count for quick context).

def tool_get_file_contents(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return full UTF-8 contents for a file plus line count, or a _meta_error on failure."""
    path = normalize_path(args.get("path", ""))
    try:
        content = read_file(ctx.repo_root, path)
    except Exception:
        return _wrap_tool_result(args, {"_meta_error": f"Could not read {path}"})
    # orion: Wrap successful payload in standard envelope.
    return _wrap_tool_result(args, {"path": path, "content": content, "line_count": count_lines(content)})


# orion: Add docstring and guard/adjust the requested range into valid bounds for the file.

def tool_get_file_snippet(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a [start_line, end_line] inclusive snippet; adjusts bounds to the file size."""
    path = normalize_path(args.get("path", ""))
    start_line = int(args.get("start_line", 1))
    end_line = int(args.get("end_line", start_line + 200))
    try:
        content = read_file(ctx.repo_root, path)
    except Exception:
        return _wrap_tool_result(args, {"_meta_error": f"Could not read {path}"})
    lines = content.splitlines()
    start_line = max(1, start_line)
    end_line = min(len(lines), end_line if end_line >= start_line else start_line)
    snippet = "\n".join(lines[start_line - 1 : end_line])
    # orion: Return standardized envelope with adjusted bounds and snippet.
    return _wrap_tool_result(args, {"path": path, "start_line": start_line, "end_line": end_line, "content": snippet})


# orion: Document that summaries are read only from colocated .orion files, not generated on demand.

def tool_get_summary(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a colocated per-file summary if present; otherwise a _meta_error."""
    # Return summary for a file from the colocated .orion path only.
    path = normalize_path(args.get("path", ""))
    try:
        sp = colocated_summary_path(ctx.repo_root, path)
        if sp.exists():
            with sp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # orion: Keep auxiliary _meta_note inside result; wrap in standard envelope.
            return _wrap_tool_result(args, {"path": path, "summary": data, "_meta_note": "summary returned (colocated)"})
        return _wrap_tool_result(args, {"_meta_error": f"no summary available for {path}"})
    except Exception as e:
        return _wrap_tool_result(args, {"_meta_error": f"summary error for {path}: {e}"})


# orion: Document linear scan behavior and case-insensitive search; cap results for performance.

def tool_search_code(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    """Search all non-ignored files for a substring; returns up to max_results path matches."""
    query = str(args.get("query", "") or "")
    if not query:
        return _wrap_tool_result(args, {"matches": []})
    matches: List[Dict[str, Any]] = []
    paths = list_repo_paths(ctx.repo_root)
    for p in paths:
        try:
            content = read_file(ctx.repo_root, p)
        except Exception:
            continue
        lower = content.lower()
        q = query.lower()
        if q in lower:
            matches.append({"path": p})
        if len(matches) >= int(args.get("max_results", 100)):
            break
    # orion: Return matches inside the standardized envelope.
    return _wrap_tool_result(args, {"matches": matches})


# orion: Document interactive prompt behavior (stdin blocking) and cancellation on empty input.

def tool_ask_user(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    """Ask a question interactively via stdin, returning the user's response (possibly empty)."""
    prompt = args.get("prompt") or args.get("question") or "Model requests input:"
    ctx.send_to_user(f"Model asks: {prompt}")
    ctx.send_to_user("Enter a response (or leave empty to cancel): ")
    try:
        ans = input().strip()
    except EOFError:
        ans = ""
    # orion: Wrap user answer in the standard envelope for consistency.
    return _wrap_tool_result(args, {"answer": ans})
