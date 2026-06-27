"""Shared JSONL helper behavior."""

from __future__ import annotations

import json

import pytest

from doc_extract import jsonl


def test_jsonl_write_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "records.jsonl"

    assert jsonl.write_jsonl(path, [{"id": "a"}]) == 1
    jsonl.append_jsonl(path, {"id": "b"})

    assert jsonl.load_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_jsonl_read_errors_include_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\n{bad\n', encoding="utf-8")

    with pytest.raises(jsonl.JsonlReadError) as exc:
        list(jsonl.iter_jsonl(path))

    assert str(exc.value).startswith(f"{path}:2:")


def test_stage_manifest_records_paths_counts_and_hashes(tmp_path):
    out = tmp_path / "out.jsonl"
    jsonl.write_jsonl(out, [{"id": "a"}])

    manifest = jsonl.write_stage_manifest(
        stage="example",
        manifest_path=tmp_path / "out.jsonl.manifest.json",
        schema_version="1.0",
        seed=42,
        counts={"written": 1},
        outputs={"jsonl": out},
    )

    saved = json.loads((tmp_path / "out.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert manifest["stage"] == "example"
    assert manifest["counts"] == {"written": 1}
    assert manifest["outputs"] == {"jsonl": str(out)}
    assert manifest["file_hashes"]["outputs"]["jsonl"] == jsonl.file_sha256(out)
