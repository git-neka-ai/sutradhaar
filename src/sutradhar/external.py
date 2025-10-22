# orion: Extracted external Project Description (PD) and Project Orion Summary (POS) helpers to isolate external directory management and hashing concerns. Added docstrings and comments to clarify assumptions and error handling.

import os
import pathlib
from typing import Any, Dict, List, Optional

from .fs import sha256_bytes, write_json


# orion: Document return contract and validation of the provided path.
def ext_dir_valid(root: Optional[str]) -> Optional[pathlib.Path]:
    """
    Validate and resolve an external PD directory path.

    Args:
        root: String path to the external (flat) PD directory.

    Returns:
        A resolved pathlib.Path if valid and a directory; otherwise None.
    """
    if not root:
        return None
    p = pathlib.Path(root).resolve()
    if not p.exists() or not p.is_dir():
        return None
    return p


# orion: Simple helper; document the convention for colocating generated summaries.
def ext_orion_dir(ext_root: pathlib.Path) -> pathlib.Path:
    """Return the .orion directory path inside the external PD root."""
    return (ext_root / ".orion").resolve()


# orion: Existing docstring kept; add inline comment describing flat-folder assumption.
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


# orion: Document filename->POS path mapping for the flat structure.
def pos_path_for_filename(ext_root: pathlib.Path, filename: str) -> pathlib.Path:
    """Map a PD filename to its POS path under ext_root/.orion (with .json extension)."""
    return (ext_orion_dir(ext_root) / f"{filename}.json").resolve()


# orion: Clarify that read_pos is best-effort and intentionally swallows errors to keep the pipeline resilient.
def read_pos(ext_root: pathlib.Path, filename: str) -> Optional[Dict[str, Any]]:
    """
    Read a Project Orion Summary (POS) JSON for the given PD filename.

    Returns:
        The POS dict if available and valid JSON; otherwise None.
    """
    p = pos_path_for_filename(ext_root, filename)
    if not p.exists():
        return None
    try:
        import json
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # orion: Corrupt or partially written POS is treated as missing; regeneration will be attempted by callers.
        return None


# orion: Simple wrapper around write_json, documented for symmetry with read_pos.
def write_pos(ext_root: pathlib.Path, filename: str, obj: Dict[str, Any]) -> None:
    """Write a POS JSON for the PD filename under ext_root/.orion."""
    p = pos_path_for_filename(ext_root, filename)
    write_json(p, obj)


# orion: Add docstring and comments about path resolution and failure modes.
def hash_pd(ext_root: pathlib.Path, filename: str) -> Optional[str]:
    """
    Compute the sha256 hash of a PD file's raw bytes.

    Returns:
        Hex digest string if file exists and is readable; otherwise None.
    """
    pd_path = (ext_root / filename).resolve()
    if not pd_path.exists() or not pd_path.is_file():
        return None
    try:
        data = pd_path.read_bytes()
    except Exception:
        # orion: Best-effort; if unreadable, propagate a None signal so callers can skip/regenerate.
        return None
    return sha256_bytes(data)
