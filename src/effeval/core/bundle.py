"""Result bundle: the single exchange format for CLI output, push, paste import and MCP write-back."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ValidationError

from effeval.core.canonical import canonical_json, sha256_hex
from effeval.core.types import Record


class Bundle(BaseModel):
    bundle_version: Literal[1] = 1
    records: list[Record]
    checksum: str


class BundleError(ValueError):
    """Raised for malformed bundles or checksum mismatches."""


def _checksum(records: list[Record]) -> str:
    payload = [r.model_dump(mode="json") for r in records]
    return "sha256:" + sha256_hex(canonical_json(payload))


def make_bundle(records: list[Record]) -> Bundle:
    return Bundle(records=records, checksum=_checksum(records))


def dumps(bundle: Bundle) -> str:
    return json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2)


def loads(text: str) -> Bundle:
    """Parse and verify a bundle. Raises BundleError when the JSON, schema or checksum is wrong."""
    try:
        bundle = Bundle.model_validate_json(text)
    except ValidationError as exc:
        raise BundleError(f"not a valid effeval result bundle: {exc.error_count()} schema error(s)") from exc
    expected = _checksum(bundle.records)
    if bundle.checksum != expected:
        raise BundleError("bundle checksum mismatch: the text was modified after it was produced")
    return bundle


PACKAGE_BUNDLE = "bundle.json"
SHA_KEY = "per_sample_sha256"


def sample_member(job_id: str) -> str:
    return f"samples/{job_id}.jsonl.gz"


def file_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_package(path: Path, bundle: Bundle) -> None:
    """Write a results package: the bundle plus every per-sample file its records point to on this machine.

    Each record's `per_sample_sha256` is the checklist the server verifies the files against.
    """
    with tarfile.open(path, "w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        add(PACKAGE_BUNDLE, dumps(bundle).encode("utf-8"))
        for record in bundle.records:
            uri = record.artifacts.get("per_sample", "")
            if uri.startswith("file://"):
                local = Path(unquote(urlparse(uri).path))
                if local.exists():
                    add(sample_member(record.job_id), local.read_bytes())


def read_package(payload: bytes) -> tuple[Bundle, dict[str, bytes]]:
    """Return (bundle, job_id -> per-sample file bytes) from a results package without extracting it.

    Only `bundle.json` and `samples/<job_id>.jsonl.gz` of the bundle's jobs are read. Raises BundleError when
    the archive or the bundle is malformed.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:

            def member(name: str) -> bytes | None:
                try:
                    handle = tar.extractfile(name)
                except KeyError:
                    return None
                return None if handle is None else handle.read()

            text = member(PACKAGE_BUNDLE)
            if text is None:
                raise BundleError(f"not a results package: {PACKAGE_BUNDLE} is missing")
            bundle = loads(text.decode("utf-8"))
            files = {}
            for record in bundle.records:
                data = member(sample_member(record.job_id))
                if data is not None:
                    files[record.job_id] = data
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise BundleError(f"not a results package: {exc}") from exc
    return bundle, files
