# orion: Lightweight YAML settings loader for Orion.

from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml


def load_settings(repo_root: pathlib.Path) -> Dict[str, Any]:
    """
    Load Orion settings from <repo>/.orion/settings.yaml or settings.yml.

    Returns an empty dict {} when the settings file is missing, unreadable, or
    does not contain a mapping. The function never raises.
    """
    try:
        orion_dir = pathlib.Path(repo_root) / ".orion"
        candidates = [orion_dir / "settings.yaml", orion_dir / "settings.yml"]
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    text = p.read_text(encoding="utf-8")
                    data = yaml.safe_load(text)
                    if isinstance(data, dict):
                        return data
                    # orion: Non-mapping YAML is treated as empty settings.
                    return {}
            except Exception:
                # orion: Swallow parse/IO errors and continue to next candidate.
                continue
        return {}
    except Exception:
        return {}
