"""JSONL and lightweight stage manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class JsonlReadError(ValueError):
    """Raised for malformed JSONL with path and line-number context."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = Path(path)
        self.line_number = line_number
        super().__init__(f"{self.path}:{self.line_number}: {message}")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield object records from a JSONL file, rejecting malformed lines with context."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlReadError(
                    path, line_number, f"malformed JSON: {exc.msg} at char {exc.pos}"
                ) from exc
            if not isinstance(record, dict):
                raise JsonlReadError(path, line_number, "expected a JSON object record")
            yield record


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, record: Mapping[str, Any], *, fsync: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if fsync:
            f.flush()
            os.fsync(f.fileno())


def file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_manifest_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.manifest.json")


def _stringify_paths(paths: Mapping[str, Path | str | None]) -> dict[str, str | None]:
    return {name: str(path) if path is not None else None for name, path in paths.items()}


def _hash_paths(paths: Mapping[str, Path | str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, path in paths.items():
        if path is None:
            continue
        digest = file_sha256(Path(path))
        if digest is not None:
            out[name] = digest
    return out


def write_stage_manifest(
    *,
    stage: str,
    manifest_path: Path,
    schema_version: str,
    seed: int | None,
    counts: Mapping[str, Any],
    inputs: Mapping[str, Path | str | None] | None = None,
    outputs: Mapping[str, Path | str | None] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = inputs or {}
    outputs = outputs or {}
    manifest: dict[str, Any] = {
        "stage": stage,
        "schema_version": schema_version,
        "seed": seed,
        "counts": dict(counts),
        "inputs": _stringify_paths(inputs),
        "outputs": _stringify_paths(outputs),
        "file_hashes": {
            "inputs": _hash_paths(inputs),
            "outputs": _hash_paths(outputs),
        },
    }
    if extra:
        manifest.update(extra)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
