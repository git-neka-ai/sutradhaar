# orion: Moved file and project summarization models and logic into a dedicated module to reuse across Orion features and to keep orion.py focused on control flow.

import json
import pathlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from .config import LINE_CAP, SUMMARY_MAX_BYTES, ORION_DEP_TTL_SEC
from .context import Context
from .fs import _safe_abs, count_lines, sha256_bytes, colocated_summary_path
from .client import ChatCompletionsClient


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


def summarize_file(ctx: Context, client: ChatCompletionsClient, repo_root: pathlib.Path, rel_path: str) -> Optional[Dict[str, Any]]:
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
    final_json = client.call_responses(
        ctx,
        messages=messages,
        tools=None,
        response_schema=schema,
        max_completion_tokens=None,
        interactive_tool_runner=None,
        reasoning_effort="minimal",
    )

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


class ProjectOrionSummary(CustomBaseModel):
    # Minimal enforced POS schema (token-optimized)
    v: int = Field(..., description="Schema version - use 1")
    f: str = Field(..., description="Project Description filename")
    h: str = Field(..., description="sha256 digest of PD raw bytes")
    ex: List[Any] = Field(..., description="Exported surface (classes/functions/config handles)")
    u: List[str] = Field(..., description="Usage notes/hints (bullets; no code)")
    r: List[str] = Field(..., description="Risks/constraints (bullets)")


def pd_summarizer_system_text() -> str:
    return (
        "You are Orion's Project Description summarizer. Produce a compact, machine-oriented project summary JSON.\n"
        "- Do NOT include code snippets; write textual bullet hints only.\n"
        "- Capture exported surface (classes/functions/config keys) and short usage notes and risks.\n"
        "- Use the strict JSON schema provided.\n"
        "- Keep tokens minimal but sufficient for planning.\n"
    )


def summarize_project_description(
    ctx: Context,
    client: ChatCompletionsClient,
    ext_root: pathlib.Path,
    filename: str,
    pd_hash: str,
) -> Optional[Dict[str, Any]]:
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
    final_json = client.call_chatcompletions(
        ctx,
        messages=messages,
        tools=None,
        response_schema=schema,
        max_completion_tokens=None,
        interactive_tool_runner=None,
    )

    try:
        ps = ProjectOrionSummary.model_validate(final_json)
    except ValidationError as ve:
        ctx.error_message(f"Project summary validation failed for {filename}: {ve}")
        return None

    obj = ps.model_dump()
    obj["f"] = filename
    obj["h"] = pd_hash
    return obj


from .external import list_project_descriptions, read_pos, write_pos, hash_pd


def ensure_pos(ctx: Context, ext_root: pathlib.Path, filename: str, client: ChatCompletionsClient) -> Optional[Dict[str, Any]]:
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
        new_pos = summarize_project_description(ctx, client, ext_root, filename, pd_hash)
        if new_pos is None:
            return None
        # Attach build metadata (non-schema)
        from .fs import now_ts
        new_pos["_built_ts"] = now_ts()
        write_pos(ext_root, filename, new_pos)
        return new_pos

    return pos


def ensure_all_pos(ctx: Context, ext_root: pathlib.Path, client: ChatCompletionsClient) -> List[Dict[str, Any]]:
    """
    Ensure POS for all PDs (flat). Returns a list of heads for bootstrap.
    """
    heads: List[Dict[str, Any]] = []
    items = list_project_descriptions(ext_root)
    if not items:
        return heads
    # Make sure .orion exists
    from .external import ext_orion_dir
    ext_orion_dir(ext_root).mkdir(parents=True, exist_ok=True)
    for fn in items:
        pos = ensure_pos(ctx, ext_root, fn, client)
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
