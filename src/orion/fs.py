# orion: Split filesystem helpers, time/id utilities, JSON helpers, hashing, and git-aware file listing from editor.py into a dedicated module to decouple IO concerns from Orion logic. Additionally, add .orionignore support to prevent listing and access to ignored files, ensuring privacy and control over Orion's file operations. Added docstrings across public helpers and detailed comments around ignore parsing and safety checks.

import hashlib
import json
import os
import pathlib
import subprocess
import time
import uuid
# orion: Extend typing imports to support types used in .orionignore caching and matcher utilities.
from typing import Any, List, Tuple, Optional, Callable, Dict


# orion: Document that now_ts is used for timestamping logs and file rotations.
def now_ts() -> float:
    """Return the current UNIX timestamp in seconds (float)."""
    return time.time()


# orion: Document the short ID format used for plan/commit identifiers.
def short_id(prefix: str) -> str:
    """Return a short unique identifier with the given prefix (e.g., prefix-1a2b3c4d)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# orion: Note: normalized to POSIX for consistent wire format and matching.
def normalize_path(p: str) -> str:
    """Normalize a filesystem path to POSIX-style string (forward slashes)."""
    return str(pathlib.Path(p).as_posix())


# orion: Document tolerant JSON read with default fallback to keep callers simple.
def read_json(path: pathlib.Path, default: Any) -> Any:
    """Read JSON from path; return default if file is missing or invalid."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# orion: Document atomic write via temp file replace.
def write_json(path: pathlib.Path, obj: Any) -> None:
    """Atomically write a JSON object to path (UTF-8, pretty-printed)."""
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# orion: JSONL helpers intentionally avoid locking; acceptable for single-process usage.
def append_jsonl(path: pathlib.Path, obj: Any) -> None:
    """Append a single JSON object as one line to a JSONL file (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# orion: Tolerant line-by-line parser; invalid lines are skipped quietly to keep history robust.
def read_jsonl(path: pathlib.Path) -> List[Any]:
    """Read a JSONL file into a list of parsed objects; returns [] if missing."""
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


# orion: Document behavior for text with/without trailing newline.
def count_lines(s: str) -> int:
    """Return the number of lines in a string, handling trailing newline gracefully."""
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


# orion: Utility for consistent hashing (used in summaries and PD hashing).
def sha256_bytes(data: bytes) -> str:
    """Compute a hex sha256 digest for the provided bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

# orion: Add helpers and cache for .orionignore support. We parse patterns once per repo_root and reuse until the file changes.
_ORIONIGNORE_CACHE: Dict[pathlib.Path, Tuple[Optional[float], List[Tuple[bool, str]]]] = {}

# orion: Return the path to the .orionignore file for a given repository root.
def _orionignore_path(repo_root: pathlib.Path) -> pathlib.Path:
    """Return the absolute .orionignore path inside repo_root."""
    return (repo_root / ".orionignore").resolve()

# orion: Parse .orionignore lines into a list of (negated, glob_pattern) tuples using Path.match-compatible semantics.

def _parse_orionignore_patterns(text: str) -> List[Tuple[bool, str]]:
    """
    Parse .orionignore contents into (negated, glob) rules.

    Rules:
      - Empty lines and comments (#) are ignored.
      - Lines starting with '!' negate the ignore (unignore).
      - Leading '/' anchors to repo root; omitted '/' matches anywhere (via **/).
      - Trailing '/' targets directories (we expand to dir/**).
    """
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
    """Return cached .orionignore rules for repo_root, refreshing when the file changes."""
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
    """Return True if rel_posix should be ignored per .orionignore matching (last rule wins)."""
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
    """Convenience wrapper to test ignore rules for an absolute or repo-relative path."""
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
    """Raise PermissionError if abs_path (within repo_root) is blocked by .orionignore."""
    rel = abs_path.relative_to(repo_root).as_posix()
    if _is_ignored_rel(repo_root, rel):
        raise PermissionError(f"Access to '{rel}' is blocked by .orionignore")


# orion: Document that _safe_abs defends against path traversal outside repo_root.

def _safe_abs(repo_root: pathlib.Path, rel: str) -> pathlib.Path:
    """Resolve a repo-relative path and reject escapes outside repo_root."""
    abs_path = (repo_root / rel).resolve()
    # ensure inside repo_root
    try:
        abs_path.relative_to(repo_root)
    except Exception:
        raise ValueError(f"Path escapes repo root: {rel}")
    return abs_path


# orion: Text read enforces ignore rules and returns the full file content.

def read_file(repo_root: pathlib.Path, path: str) -> str:
    """Read a UTF-8 text file relative to repo_root, enforcing .orionignore rules."""
    abs_path = _safe_abs(repo_root, path)
    # orion: Enforce .orionignore; block reads to ignored paths to respect user privacy and intent.
    _ensure_not_ignored(repo_root, abs_path)
    with abs_path.open("r", encoding="utf-8") as f:
        return f.read()


# orion: Text write enforces ignore rules and creates parent directories.

def write_file(repo_root: pathlib.Path, path: str, content: str) -> None:
    """Write text content to a repo-relative file, enforcing .orionignore rules."""
    abs_path = _safe_abs(repo_root, path)
    # orion: Enforce .orionignore; block writes to ignored paths to avoid unintended modifications.
    _ensure_not_ignored(repo_root, abs_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with abs_path.open("w", encoding="utf-8") as f:
        f.write(content)


# orion: Binary read with the same ignore policy as text read.

def read_bytes(repo_root: pathlib.Path, path: str) -> bytes:
    """Read a binary file relative to repo_root, enforcing .orionignore rules."""
    abs_path = _safe_abs(repo_root, path)
    # orion: Enforce .orionignore; block binary reads to ignored paths for consistency with text reads.
    _ensure_not_ignored(repo_root, abs_path)
    with abs_path.open("rb") as f:
        return f.read()


# orion: Enumerate repo files while pruning git/.orion and internal metadata; also honors .orionignore rules.

def list_repo_paths(repo_root: pathlib.Path) -> List[str]:
    """Walk the working tree and return non-ignored repo-relative file paths (POSIX)."""
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


# orion: Clarify the colocated path scheme for per-file summaries.

def colocated_summary_path(repo_root: pathlib.Path, rel_path: str) -> pathlib.Path:
    """
    For a source file at a/b/c.ext, return the path a/b/.orion/c.ext.json under repo_root.
    """
    rel = pathlib.Path(rel_path)
    summary_dir = rel.parent / ".orion"
    summary_name = rel.name + ".json"
    return (repo_root / summary_dir / summary_name).resolve()

# -----------------------------
# Git helpers for file discovery
# -----------------------------

# orion: Thin wrapper for running git, returning (rc, stdout, stderr) as text.

def run_git(args: List[str], cwd: pathlib.Path) -> Tuple[int, str, str]:
    """Run a git command in cwd and return (returncode, stdout, stderr)."""
    proc = subprocess.Popen(["git"] + args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


# orion: List tracked files using git ls-files -z. Returns [] if git is unavailable.

def git_list_tracked(repo_root: pathlib.Path) -> List[str]:
    """Return a sorted list of tracked files (as reported by git ls-files)."""
    rc, out, _ = run_git(["ls-files", "-z"], repo_root)
    if rc != 0:
        return []
    files = [p for p in out.split("\x00") if p]
    return sorted(files)


# orion: List untracked but unignored files per git's standard ignore rules.

def git_list_untracked_unignored(repo_root: pathlib.Path) -> List[str]:
    """Return a sorted list of untracked, unignored files (git -o --exclude-standard)."""
    rc, out, _ = run_git(["ls-files", "-o", "--exclude-standard", "-z"], repo_root)
    if rc != 0:
        return []
    files = [p for p in out.split("\x00") if p]
    return sorted(files)


# orion: Prefer git for speed and parity with dev workflows; otherwise fall back to a plain walk. Always enforce .orionignore.

def list_all_nonignored_files(repo_root: pathlib.Path) -> List[str]:
    """Return the union of tracked and untracked-unignored files, filtered by .orionignore and Orion internals."""
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
