# orion: Extracted external Project Description (PD) and Project Orion Summary (POS) helpers to isolate external directory management and hashing concerns.

import os
import pathlib
from typing import Any, Dict, List, Optional

from .fs import sha256_bytes, write_json


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
        import json
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
