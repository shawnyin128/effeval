"""Prompt sets of the speed evaluator: sample validation, built-in manifests and user sets.

A prompt set is a JSONL file of samples (format in `effeval.evaluators.speed`). Built-in sets are manifests in
`core/data/prompt_sets/*.yaml`: an inline JSONL file shipped in that package, or a pinned download with a
converter from the source format. User sets are `<name>.jsonl` files in the directory named by the
`EFFEVAL_PROMPT_SETS` environment variable. Pure Python: the server loads this to list tasks.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from effeval.core.types import StrictModel

ENV_VAR = "EFFEVAL_PROMPT_SETS"
PACKAGE = "effeval.core.data.prompt_sets"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_KEYS = frozenset({"id", "category", "turns", "system", "max_new_tokens"})
_REQUIRED = ("id", "category", "turns")


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    turns: tuple[str, ...]
    system: str | None = None
    max_new_tokens: int | None = None


def _text(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field!r} must be a non-empty string")
    return value


def check_record(record: Any, where: str) -> Prompt:
    """The `Prompt` of one decoded sample object. Raises ValueError naming `where` and the offending key."""
    if not isinstance(record, dict):
        raise ValueError(f"{where}: a sample must be a JSON object, got {type(record).__name__}")
    unknown = sorted(set(record) - _KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; allowed: {sorted(_KEYS)}")
    missing = [key for key in _REQUIRED if key not in record]
    if missing:
        raise ValueError(f"{where}: missing required keys {missing}")
    turns = record["turns"]
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{where}: 'turns' must be a non-empty list of strings")
    system = record.get("system")
    if "system" in record:
        system = _text(system, "system", where)
    max_new_tokens = record.get("max_new_tokens")
    if "max_new_tokens" in record and (
        isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1
    ):
        raise ValueError(f"{where}: 'max_new_tokens' must be an integer >= 1")
    return Prompt(
        id=_text(record["id"], "id", where),
        category=_text(record["category"], "category", where),
        turns=tuple(_text(turn, "turns[]", where) for turn in turns),
        system=system,
        max_new_tokens=max_new_tokens,
    )


def _checked(records: list[tuple[str, Any]]) -> list[Prompt]:
    prompts: list[Prompt] = []
    first: dict[str, str] = {}
    for where, record in records:
        prompt = check_record(record, where)
        if prompt.id in first:
            raise ValueError(f"{where}: duplicate id {prompt.id!r}, first used at {first[prompt.id]}")
        first[prompt.id] = where
        prompts.append(prompt)
    if not prompts:
        raise ValueError("a prompt set needs at least one sample")
    return prompts


def parse_prompt_set(text: str, source: str) -> list[Prompt]:
    """The samples of JSONL `text`, blank lines skipped; errors name `source` and the line number.

    Raises ValueError on invalid JSON, a non-object line, a missing, unknown or ill-typed key, a duplicate id
    or no samples.
    """
    records = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        where = f"{source}:{number}"
        try:
            records.append((where, json.loads(line)))
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: invalid JSON ({error.msg})") from error
    return _checked(records)


class Download(StrictModel):
    url: str
    commit: str = Field(description="Revision of the source the url points at; documentation only.")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="Checksum of the downloaded file.")
    cache: str = Field(description="Path of the verified file under `$XDG_CACHE_HOME/effeval/`.")
    records_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Checksum of the canonical JSONL of the converter's records; pins its output.",
    )


class PromptSetManifest(StrictModel):
    name: str
    version: str = Field(description="Version of the manifest and its converter; bump when either changes.")
    description: str
    samples: int = Field(ge=1, description="Sample count, for progress estimates.")
    categories: int = Field(default=1, ge=1, description="Category count, for progress estimates.")
    file: str | None = Field(
        default=None, description="Inline JSONL in the sample format, in the data package."
    )
    download: Download | None = None
    converter: str | None = Field(default=None, description="Module of `core/prompt_converters`.")

    @model_validator(mode="after")
    def _one_source(self) -> PromptSetManifest:
        if (self.file is None) == (self.download is None):
            raise ValueError(f"prompt set {self.name!r}: give exactly one of `file` and `download`")
        if (self.converter is None) != (self.download is None):
            raise ValueError(f"prompt set {self.name!r}: a converter goes with a download, and only with it")
        if not _NAME.match(self.name):
            raise ValueError(f"prompt set name {self.name!r} must match {_NAME.pattern}")
        return self


@dataclass(frozen=True)
class PromptSetInfo:
    """What the task catalog needs of a set; `version` identifies its content for run identity."""

    name: str
    origin: Literal["builtin", "user"]
    version: str
    description: str
    samples: int
    categories: int


@dataclass(frozen=True)
class PromptSet:
    info: PromptSetInfo
    prompts: list[Prompt]
    sha256: str
    """SHA-256 of the samples: the file's bytes, or the canonical JSONL of converted records."""
    source: dict[str, Any]
    """Where the samples came from, for the protocol."""


def builtin_manifests() -> dict[str, PromptSetManifest]:
    """The manifests shipped in `core/data/prompt_sets`, by set name."""
    found: dict[str, PromptSetManifest] = {}
    for entry in sorted(resources.files(PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".yaml"):
            manifest = PromptSetManifest.model_validate(yaml.safe_load(entry.read_text(encoding="utf-8")))
            found[manifest.name] = manifest
    return found


def user_set_dir() -> Path | None:
    """The directory named by `EFFEVAL_PROMPT_SETS` if it exists, else None."""
    value = os.environ.get(ENV_VAR)
    return Path(value) if value and Path(value).is_dir() else None


def user_set_files() -> dict[str, Path]:
    """`<name>.jsonl` files of the user directory by set name. Raises ValueError on a name that is not valid
    or is a built-in set's."""
    directory = user_set_dir()
    if directory is None:
        return {}
    builtin = builtin_manifests()
    found = {}
    for path in sorted(directory.glob("*.jsonl")):
        if not _NAME.match(path.stem):
            raise ValueError(f"{path}: a prompt set name must match {_NAME.pattern}")
        if path.stem in builtin:
            raise ValueError(f"{path}: {path.stem!r} is a built-in prompt set")
        found[path.stem] = path
    return found


def user_set_stamp() -> tuple[tuple[str, int, int], ...]:
    """(name, size, mtime_ns) of every user file; changes whenever a set is added, removed or edited."""
    return tuple(
        (name, (st := path.stat()).st_size, st.st_mtime_ns) for name, path in user_set_files().items()
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_jsonl(prompts: list[Prompt]) -> bytes:
    lines = []
    for p in prompts:
        record = {"id": p.id, "category": p.category, "turns": list(p.turns)}
        if p.system is not None:
            record["system"] = p.system
        if p.max_new_tokens is not None:
            record["max_new_tokens"] = p.max_new_tokens
        lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _builtin_file(manifest: PromptSetManifest) -> bytes:
    assert manifest.file is not None
    return resources.files(PACKAGE).joinpath(manifest.file).read_bytes()


def _builtin_info(manifest: PromptSetManifest) -> PromptSetInfo:
    if manifest.download:
        pinned = f"{manifest.download.sha256}+records:{manifest.download.records_sha256}"
    else:
        pinned = _sha256(_builtin_file(manifest))
    return PromptSetInfo(
        name=manifest.name,
        origin="builtin",
        version=f"{manifest.version}+sha256:{pinned}",
        description=manifest.description,
        samples=manifest.samples,
        categories=manifest.categories,
    )


def _user_info(name: str, path: Path) -> PromptSetInfo:
    raw = path.read_bytes()
    return PromptSetInfo(
        name=name,
        origin="user",
        version=f"sha256:{_sha256(raw)}",
        description=f"Prompt set {path.name} from the directory {ENV_VAR} names.",
        samples=max(1, sum(1 for line in raw.split(b"\n") if line.strip())),
        categories=1,
    )


def builtin_set_infos() -> dict[str, PromptSetInfo]:
    """The built-in prompt sets by name."""
    return {name: _builtin_info(m) for name, m in builtin_manifests().items()}


def user_set_infos() -> dict[str, PromptSetInfo]:
    """The user prompt sets by name; reads the files to hash them."""
    return {name: _user_info(name, path) for name, path in user_set_files().items()}


def cache_path(download: Download) -> Path:
    """`$XDG_CACHE_HOME/effeval/<download.cache>`; `~/.cache` if XDG is unset."""
    root = os.environ.get("XDG_CACHE_HOME")
    return (Path(root) if root else Path.home() / ".cache") / "effeval" / download.cache


def fetch(path: Path, url: str, sha256: str) -> Path:
    """The verified file `path`, downloaded from `url` when absent.

    Raises ValueError when the file's or the download's sha256 differs from `sha256`; a bad download leaves
    no file.
    """
    if path.exists():
        found = _sha256(path.read_bytes())
        if found != sha256:
            raise ValueError(f"{path} has sha256 {found}, expected {sha256}")
        return path
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    found = _sha256(data)
    if found != sha256:
        raise ValueError(f"download of {url} has sha256 {found}, expected {sha256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)
    return path


def _load_builtin(manifest: PromptSetManifest) -> PromptSet:
    info = _builtin_info(manifest)
    if manifest.download is None:
        raw = _builtin_file(manifest)
        prompts = parse_prompt_set(raw.decode("utf-8"), manifest.file or manifest.name)
        return PromptSet(info, prompts, _sha256(raw), {"kind": "builtin_file", "file": manifest.file})
    download = manifest.download
    raw = fetch(cache_path(download), download.url, download.sha256).read_bytes()
    converter = importlib.import_module(f"effeval.core.prompt_converters.{manifest.converter}")
    records = [(f"{manifest.name} converted sample {i}", r) for i, r in enumerate(converter.convert(raw), 1)]
    prompts = _checked(records)
    converted = _sha256(_canonical_jsonl(prompts))
    if converted != download.records_sha256:
        raise ValueError(
            f"the {manifest.converter!r} converter produced records with sha256 {converted} from the pinned "
            f"{manifest.name} download, but the manifest pins {download.records_sha256}: the converter "
            "changed; review it, then bump the manifest version and update records_sha256"
        )
    source = {"kind": "download", "url": download.url, "commit": download.commit, "sha256": download.sha256}
    return PromptSet(info, prompts, converted, source)


def load_prompt_set(name: str) -> PromptSet:
    """The built-in or user set `name`, validated.

    Raises LookupError for an unknown name and ValueError for an invalid file, a checksum mismatch, a bad
    conversion or a converter whose output differs from the manifest's pinned `records_sha256`.
    """
    manifests = builtin_manifests()
    if name in manifests:
        return _load_builtin(manifests[name])
    files = user_set_files()
    if name not in files:
        raise LookupError(f"unknown prompt set {name!r}; known: {sorted({*manifests, *files})}")
    path = files[name]
    raw = path.read_bytes()
    prompts = parse_prompt_set(raw.decode("utf-8"), path.name)
    return PromptSet(_user_info(name, path), prompts, _sha256(raw), {"kind": "user_file", "file": path.name})
