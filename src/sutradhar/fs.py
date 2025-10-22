# orion: Split filesystem helpers, time/id utilities, JSON helpers, hashing, and git-aware file listing from editor.py into a dedicated module to decouple IO concerns from Orion logic. Additionally, add .orionignore support to prevent listing and access to ignored files, ensuring privacy and control over Orion's file operations.

import hashlib
import json
import os
import pathlib
import subprocess
import time
import uuid
# orion: Extend typing imports to support types used in .orionignore caching and matcher utilities.
from typing import Any, List, Tuple, Optional, Callable, Dict


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


def append_jsonl(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: pathlib.Path) -> List[Any]:
    if not path.exists():
        return []
    lines: List[Any] = []
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

# orion: Add helpers and cache for .orionignore support. We parse patterns once per repo_root and reuse until the file changes.
_ORIONIGNORE_CACHE: Dict[pathlib.Path, Tuple[Optional[float], List[Tuple[bool, str]]]] = {}

# orion: Return the path to the .orionignore file for a given repository root.
def _orionignore_path(repo_root: pathlib.Path) -> pathlib.Path:
    return (repo_root / ".orionignore").resolve()

# orion: Parse .orionignore lines into a list of (negated, glob_pattern) tuples using Path.match-compatible semantics.
def _parse_orionignore_patterns(text: str) -> List[Tuple[bool, str]]:
    patterns: List[Tuple[bool, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = False
        if line.startswith("!"):
            negated = True
            line = line[1:].strip()
            if not line:
                continue
        rooted = line.startswith("/")
        if rooted:
            line = line.lstrip("/")
        dir_only = line.endswith("/")
        if dir_only:
            line = line.rstrip("/")
        # Build a Path.match-compatible glob
        pat = line
        if dir_only:
            # Match anything inside the directory
            pat = f"{pat}/**"
        if rooted:
            glob_pat = pat
        else:
            # Allow match anywhere in the tree
            glob_pat = f"**/{pat}" if pat else "**"
        patterns.append((negated, glob_pat))
    return patterns

# orion: Load and cache .orionignore patterns, invalidating cache on file mtime changes.
def _get_orionignore_patterns(repo_root: pathlib.Path) -> List[Tuple[bool, str]]:
    ig_path = _orionignore_path(repo_root)
    try:
        mtime = ig_path.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    cached = _ORIONIGNORE_CACHE.get(repo_root)
    if cached and cached[0] == mtime:
        return cached[1]
    if mtime is None:
        patterns: List[Tuple[bool, str]] = []
    else:
        try:
            text = ig_path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        patterns = _parse_orionignore_patterns(text)
    _ORIONIGNORE_CACHE[repo_root] = (mtime, patterns)
    return patterns

# orion: Determine if a given relative POSIX path is ignored by .orionignore rules (last match wins; negation unignores).
def _is_ignored_rel(repo_root: pathlib.Path, rel_posix: str) -> bool:
    rules = _get_orionignore_patterns(repo_root)
    if not rules:
        return False
    p = pathlib.PurePosixPath(rel_posix)
    ignored = False
    for negated, pat in rules:
        if p.match(pat):
            ignored = not negated
    return ignored

# orion: Normalize and check if a path (absolute or relative) is ignored according to .orionignore.
def is_ignored_path(repo_root: pathlib.Path, path: pathlib.Path | str) -> bool:
    abs_path = (repo_root / str(path)).resolve() if not isinstance(path, pathlib.Path) else path.resolve()
    try:
        rel = abs_path.relative_to(repo_root)
    except Exception:
        # Outside repo; treat as not ignored here, _safe_abs will guard.
        return False
    rel_posix = rel.as_posix()
    return _is_ignored_rel(repo_root, rel_posix)

# orion: Guard function to enforce ignore rules before file IO operations.
def _ensure_not_ignored(repo_root: pathlib.Path, abs_path: pathlib.Path) -> None:
    rel = abs_path.relative_to(repo_root).as_posix()
    if _is_ignored_rel(repo_root, rel):
        raise PermissionError(f"Access to '{rel}' is blocked by .orionignore")


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
    # orion: Enforce .orionignore; block reads to ignored paths to respect user privacy and intent.
    _ensure_not_ignored(repo_root, abs_path)
    with abs_path.open("r", encoding="utf-8") as f:
        return f.read()


def write_file(repo_root: pathlib.Path, path: str, content: str) -> None:
    abs_path = _safe_abs(repo_root, path)
    # orion: Enforce .orionignore; block writes to ignored paths to avoid unintended modifications.
    _ensure_not_ignored(repo_root, abs_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with abs_path.open("w", encoding="utf-8") as f:
        f.write(content)


def read_bytes(repo_root: pathlib.Path, path: str) -> bytes:
    abs_path = _safe_abs(repo_root, path)
    # orion: Enforce .orionignore; block binary reads to ignored paths for consistency with text reads.
    _ensure_not_ignored(repo_root, abs_path)
    with abs_path.open("rb") as f:
        return f.read()


def list_repo_paths(repo_root: pathlib.Path) -> List[str]:
    paths: List[str] = []
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
            # orion: Exclude files matched by .orionignore when walking the filesystem.
            if _is_ignored_rel(repo_root, normalize_path(rel)):
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
    rc, out, _ = run_git(["ls-files", "-z"], repo_root)
    if rc != 0:
        return []
    files = [p for p in out.split("\x00") if p]
    return sorted(files)


def git_list_untracked_unignored(repo_root: pathlib.Path) -> List[str]:
    rc, out, _ = run_git(["ls-files", "-o", "--exclude-standard", "-z"], repo_root)
    if rc != 0:
        return []
    files = [p for p in out.split("\x00") if p]
    return sorted(files)


def list_all_nonignored_files(repo_root: pathlib.Path) -> List[str]:
    # Prefer Git if available
    rc, out, _ = run_git(["rev-parse", "--is-inside-work-tree"], repo_root)

    def is_internal(p: str) -> bool:
        # Skip runtime metadata and any file inside a .orion directory
        if p.endswith("orion-metadata.json") or p.endswith("orion-conversation.jsonl"):
            return True
        return ".orion" in pathlib.Path(p).parts

    if rc == 0 and out.strip() == "true":
        tracked = set(git_list_tracked(repo_root))
        untracked = set(git_list_untracked_unignored(repo_root))
        everything = [normalize_path(p) for p in sorted((tracked | untracked)) if not is_internal(p)]
        # orion: Filter Git-derived file list by .orionignore to provide a consistent view across Git and non-Git environments.
        everything = [p for p in everything if not _is_ignored_rel(repo_root, p)]
        return everything
    # Fallback: full walk without .git and skipping .orion
    return list_repo_paths(repo_root)
