# orion: Externalized system prompts to orion.resources loaded via orion.prompts.get_prompt. Also ensured imports are complete for existing logic. Added docstrings and inline comments to clarify size limits, schema enforcement, and regeneration TTLs.

import json
import pathlib
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from .config import LINE_CAP, SUMMARY_MAX_BYTES, ORION_DEP_TTL_SEC
from .context import Context
# orion: Import is_ignored_path and write_file to enforce .orionignore on reads/writes for summaries.
from .fs import _safe_abs, count_lines, sha256_bytes, colocated_summary_path, is_ignored_path, write_file
from .client import ChatCompletionsClient
from .prompts import get_prompt
# orion: Import new minimal summary models and remove legacy FileSummary usage.
from .models import CustomBaseModel, CodeSummary, InfoSummary, HtmlSummary, CssSummary


# orion: Add a small heuristic docstring for language detection.

def guess_language(path: str) -> str:
    """Guess a short language tag from a filename suffix (defaults to 'info')."""
    ext = pathlib.Path(path).suffix.lower()
    # orion: Extend heuristic to cover html/info/css and common config/text types for hybrid routing.
    return {
        ".py": "py",
        ".ts": "ts",
        ".tsx": "tsx",
        ".js": "js",
        ".jsx": "jsx",
        ".sh": "sh",
        ".bash": "sh",
        ".zsh": "sh",
        ".html": "html",
        ".htm": "html",
        ".md": "info",
        ".markdown": "info",
        ".rst": "info",
        ".txt": "info",
        ".json": "info",
        ".yaml": "info",
        ".yml": "info",
        ".toml": "info",
        ".ini": "info",
        ".css": "css",
    }.get(ext, "info")


# orion: Document summarization flow, including size cap, schema validation, and colocated output.

def summarize_file(ctx: Context, client: ChatCompletionsClient, repo_root: pathlib.Path, rel_path: str) -> Optional[Dict[str, Any]]:
    """
    Create or update a colocated JSON summary for a given source file.

    Steps:
      1) Enforce a maximum file size to bound tokens and latency.
      2) Route by language to minimal summary schemas and prompts.
      3) Validate the response and write to a/b/.orion/file.ext.json using compact separators.

    Returns:
        Parsed summary object on success; None on failure or skip.
    """
    abs_path = _safe_abs(repo_root, rel_path)

    # orion: Respect .orionignore — skip summarization entirely if the source path is ignored.
    if is_ignored_path(repo_root, rel_path):
        ctx.log(f"Skipping summary for ignored path: {rel_path}")
        return None

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
    language_tag = guess_language(rel_path)
    info = {
        "path": rel_path,
        "language": language_tag,
        "line_count": count_lines(text),
        "size_bytes": len(data),
        "sha256": digest,
    }

    # orion: Route by language to minimal summary schemas (CodeSummary/HtmlSummary/InfoSummary/CssSummary) and prompts; emit minimal objects only (no headers/meta).
    if language_tag == "html":
        system_txt = get_prompt("prompt_summarizer_html_system.txt")
        schema = HtmlSummary.model_json_schema()
        model_cls = HtmlSummary
    elif language_tag == "css":
        system_txt = get_prompt("prompt_summarizer_css_system.txt")
        schema = CssSummary.model_json_schema()
        model_cls = CssSummary
    elif language_tag == "info":
        system_txt = get_prompt("prompt_summarizer_info_system.txt")
        schema = InfoSummary.model_json_schema()
        model_cls = InfoSummary
    else:
        # Default: treat as code
        system_txt = get_prompt("prompt_summarizer_code_system.txt", line_cap=LINE_CAP)
        schema = CodeSummary.model_json_schema()
        model_cls = CodeSummary

    user_txt = json.dumps({"info": info, "content": text}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt}
    ]

    ctx.log(f"Summarizing file: {rel_path} (size: {len(data)} bytes, lines: {info['line_count']}, lang: {language_tag})")
    final_json = client.call_responses(
        ctx,
        messages=messages,
        tools=None,
        response_schema=schema,
        max_completion_tokens=None,
        interactive_tool_runner=None,
        call_type="file_summary",
    )

    try:
        validated = model_cls.model_validate(final_json)
    except ValidationError as ve:
        ctx.error_message(f"Summary validation failed for {rel_path}: {ve}")
        return None

    obj = validated.model_dump()

    sp = colocated_summary_path(repo_root, rel_path)

    # orion: Avoid persisting under ignored summary paths (e.g., ignored directories); honor .orionignore for targets too.
    if is_ignored_path(repo_root, sp):
        ctx.log(f"Skipping summary write for ignored target path: {sp.relative_to(repo_root).as_posix()}")
        return obj

    # orion: Route writes through fs.write_file to enforce .orionignore and create parents; compact separators minimize on-disk size.
    rel_summary_path = sp.relative_to(repo_root).as_posix()
    write_file(
        repo_root,
        rel_summary_path,
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n",
    )

    return obj


class ProjectOrionSummary(CustomBaseModel):
    """Compact project-level summary for external PD files, with strict fields and minimized keys."""
    # Minimal enforced POS schema (token-optimized)
    v: int = Field(..., description="Schema version - use 1")
    f: str = Field(..., description="Project Description filename")
    h: str = Field(..., description="sha256 digest of PD raw bytes")
    ex: List[str] = Field(..., description="Exported surface (classes/functions/config handles)")
    u: List[str] = Field(..., description="Usage notes/hints (bullets; no code)")
    r: List[str] = Field(..., description="Risks/constraints (bullets)")


# orion: Document the PD summarization flow and migrate to the Responses API to unify endpoints with file summarizer.

def summarize_project_description(
    ctx: Context,
    client: ChatCompletionsClient,
    ext_root: pathlib.Path,
    filename: str,
    pd_hash: str,
) -> Optional[Dict[str, Any]]:
    """
    Summarize a Project Description (PD) file into a compact POS structure.

    Args:
        ctx: Context for logging.
        client: OpenAI client for model calls.
        ext_root: External PD directory.
        filename: PD filename (flat directory).
        pd_hash: Precomputed sha256 of the PD raw bytes.

    Returns:
        The validated POS dict or None on failure.
    """
    pd_path = (ext_root / filename).resolve()
    try:
        data = pd_path.read_bytes()
    except Exception as e:
        ctx.error_message(f"Failed to read PD {filename}: {e}")
        return None

    text = data.decode("utf-8", errors="replace")
    info = {"filename": filename, "size_bytes": len(data), "sha256": pd_hash}

    # orion: Load PD summarizer system prompt from resources.
    system_txt = get_prompt("prompt_summarizer_pd_system.txt")
    user_txt = json.dumps({"info": info, "content": text}, ensure_ascii=False)
    schema = ProjectOrionSummary.model_json_schema()

    messages = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt}
    ]

    ctx.log(f"Summarizing Project Description: {filename} (size: {len(data)} bytes)")
    # orion: Use Responses API with minimal reasoning to standardize all LLM invocations.
    final_json = client.call_responses(
        ctx,
        messages=messages,
        tools=None,
        response_schema=schema,
        max_completion_tokens=None,
        interactive_tool_runner=None,
        call_type="project_summary",
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


# orion: Document regeneration policy: on missing/corrupt, hash mismatch, or TTL expiry.

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


# orion: Document that heads are returned for bootstrap and that stale POS are tolerated (optional GC comment).

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
