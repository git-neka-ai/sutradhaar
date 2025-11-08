# orion: Add centralized Pydantic v2 models used by conversation and apply flows; this removes ad-hoc
# JSON Schemas and ensures a single source of truth. Includes path normalization validators to preserve
# existing behavior and Config(extra='forbid') to keep responses strict.

from __future__ import annotations

from enum import Enum
# orion: Add Optional typing import to support optional CodeSummary field in HtmlSummary.
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .fs import normalize_path

class CustomBaseModel(BaseModel):
    """Pydantic base model configured to forbid unknown fields for strict validation."""
    model_config = ConfigDict(extra="forbid")


class FileSummary(CustomBaseModel):
    """Compact per-file summary optimized for LLM consumption (token minimized)."""
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


# orion: Add new minimal, language-aware summary models (no headers/meta) for hybrid routing.
class CodeSummary(CustomBaseModel):
    ex: List[str] = Field(default_factory=list, description="Exports/symbols")
    im: List[str] = Field(default_factory=list, description="Imports/dependencies")
    fx: List[str] = Field(default_factory=list, description="Functions")
    cl: List[str] = Field(default_factory=list, description="Classes")
    cfg: List[str] = Field(default_factory=list, description="Configs/environment variables used")
    r: List[str] = Field(default_factory=list, description="Risks/constraints")
    sm: List[str] = Field(default_factory=list, description="Notes on safe-to-modify areas")


class InfoSummary(CustomBaseModel):
    s: str = Field(..., description="Short synopsis (2–4 sentences)")


class HtmlSummary(CustomBaseModel):
    info: InfoSummary = Field(..., description="Synopsis of the HTML page")
    code: Optional[CodeSummary] = Field(default=None, description="Summary of inline <script> JavaScript only")


class CssSummary(CustomBaseModel):
    sel: List[str] = Field(default_factory=list, description="Deduplicated CSS selectors")


class ChangeType(str, Enum):
    modify = "modify"
    create = "create"
    delete = "delete"
    move = "move"
    rename = "rename"


class ChangeItem(CustomBaseModel):

    path: str = Field(..., description="File path affected by the change")
    change_type: ChangeType = Field(..., description="Type of change applied to the file")
    summary_of_change: str = Field(..., description="Summary of the change")

    # orion: Normalize file paths on ingest so downstream logic always sees normalized paths.
    @field_validator("path")
    @classmethod
    def _normalize_path(cls, v: str) -> str:
        return normalize_path(v)


class ChangeSpec(CustomBaseModel):

    id: str = Field(..., description="Unique identifier for the change spec")
    title: str = Field(..., description="Short title for the change spec")
    description: str = Field(..., description="Detailed description of the change spec")
    items: List[ChangeItem] = Field(..., description="List of individual change items")


class ConversationResponse(CustomBaseModel):

    assistant_message: str = Field(..., description="Assistant's message content")
    changes: List[ChangeSpec] = Field(..., description="List of proposed change specifications")


# orion: Rename FilePatch to FileContents and field 'code' to 'contents' to reflect whole-file replacement semantics.
class FileContents(CustomBaseModel):

    path: str = Field(..., description="File path to write")
    is_new: bool = Field(..., description="Whether the file is newly created")
    contents: str = Field(..., description="File contents")

    # orion: Normalize output file paths before writing to disk to match prior behavior.
    @field_validator("path")
    @classmethod
    def _normalize_path(cls, v: str) -> str:
        return normalize_path(v)


class Issue(CustomBaseModel):
    reason: str = Field(..., description="Description of the issue encountered")
    paths: List[str] = Field(..., description="List of file paths related to the issue")

    # orion: Normalize any paths referenced in issues for consistency.
    @field_validator("paths")
    @classmethod
    def _normalize_paths(cls, v: List[str]) -> List[str]:
        return [normalize_path(p) for p in v]


class ApplyResponse(CustomBaseModel):

    mode: Literal["ok", "incompatible"] = Field(..., description="Status of the apply operation")
    explanation: str = Field(..., description="Explanation of the apply status")
    # orion: Update ApplyResponse.files to use FileContents after type rename.
    files: List[FileContents] = Field(..., description="List of file contents to apply")
    issues: List[Issue] = Field(..., description="List of issues encountered during apply")
