# orion: Introduced a small helper to load prompt templates from orion.resources via importlib.resources and optionally format them with dynamic values. This centralizes prompt management and enables reuse across modules. Added docstring clarifying brace handling and formatting behavior.

from importlib import resources


def get_prompt(name: str, **kwargs) -> str:
    """
    Load a text prompt from the orion.resources package.

    If kwargs are provided, apply str.format(**kwargs) to the content so prompts can
    contain placeholders (e.g., {line_cap}). If no kwargs are provided, return the
    raw text without attempting formatting to avoid accidental brace handling in
    prompts that show JSON examples.
    """
    data = resources.files("orion.workbench.resources").joinpath(name).read_text(encoding="utf-8")
    if kwargs:
        return data.format(**kwargs)
    return data
