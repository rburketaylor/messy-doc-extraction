"""Invariant: corruption preserves the gold's value tokens (dates + amounts)."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime

import pytest

from doc_extract import canon, corrupt, generate
from doc_extract.jsonl import sidecar_manifest_path

_DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]{3} \d{4}|\d{1,2}/\d{1,2}/\d{4}"
    r"|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2}-[A-Za-z]{3}-\d{4}"
)
_MONEY_RE = re.compile(r"(?:C\$|A\$|[¥$€£₹]|[A-Z]{3})\s?[\d,]+\.\d{2}")


def _date_multiset(text):
    return sorted(canon.normalize_date(d) for d in _DATE_RE.findall(text))


def _amount_multiset(text):
    return sorted(canon.normalize_amount(a) for a in _MONEY_RE.findall(text))


@pytest.fixture(scope="module")
def clean_records(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "clean.jsonl"
    generate.generate(15, 42, p)
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def test_normalizer_roundtrips():
    iso = "2026-06-25"
    base = datetime.strptime(iso, "%Y-%m-%d")
    for fmt in ("%d %b %Y", "%d/%m/%Y", "%B %d, %Y", "%d-%b-%Y"):
        assert canon.normalize_date(base.strftime(fmt)) == iso
    assert canon.normalize_amount("$1,234.56") == "1234.56"
    assert canon.normalize_amount("USD 1234.56") == "1234.56"
    assert canon.normalize_amount("C$1,000.00") == "1000.00"
    assert canon.normalize_quantity("2.00") == "2"
    assert canon.normalize_unit("each") == "EA"


@pytest.mark.parametrize("transform", corrupt.TRANSFORMS)
def test_per_corruption_type_preserves_values(clean_records, transform):
    for rec in clean_records:
        rng = random.Random(hash((rec["id"], transform.__name__)) % (2**31))
        if transform is corrupt._corrupt_amounts:
            dirty = transform(rec["clean_text"], rng, rec["clean_json"]["currency"])
        else:
            dirty = transform(rec["clean_text"], rng)
        assert _date_multiset(dirty) == _date_multiset(rec["clean_text"]), \
            f"{transform.__name__} changed date multiset: id={rec['id']}"
        assert _amount_multiset(dirty) == _amount_multiset(rec["clean_text"]), \
            f"{transform.__name__} changed amount multiset: id={rec['id']}"


def test_value_multiset_preserved(clean_records):
    for rec in clean_records:
        for seed in range(6):
            dirty = corrupt.apply_corruptions(rec["clean_text"], rec["clean_json"],
                                              random.Random(seed))
            assert corrupt.is_label_preserving(rec["clean_text"], rec["clean_json"], dirty), \
                f"value multiset changed: id={rec['id']} seed={seed}"


def test_label_breaking_detected(clean_records):
    """Value LOSS and value INJECTION must fail is_label_preserving (meaningful invariant)."""
    rec = clean_records[0]
    clean = rec["clean_text"]
    a_date = _DATE_RE.search(clean)
    assert a_date is not None
    dropped = clean.replace(a_date.group(0), "", 1)
    assert not corrupt.is_label_preserving(clean, rec["clean_json"], dropped)
    injected = clean + "\n$9.99"
    assert not corrupt.is_label_preserving(clean, rec["clean_json"], injected)


def test_dirty_jsonl_carries_gold(clean_records, tmp_path):
    clean_path = tmp_path / "clean.jsonl"
    dirty_path = tmp_path / "dirty.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r) + "\n")
    n = corrupt.corrupt_file(clean_path, dirty_path, 42)
    assert n == len(clean_records)
    manifest = json.loads(sidecar_manifest_path(dirty_path).read_text(encoding="utf-8"))
    assert manifest["stage"] == "corrupt"
    assert manifest["counts"] == {"written": len(clean_records)}
    assert set(manifest["file_hashes"]) == {"inputs", "outputs"}
    rows = [json.loads(line) for line in dirty_path.open(encoding="utf-8")]
    assert len(rows) == len(clean_records)
    for r in rows:
        assert r["dirty_text"]
        assert r["clean_json"]
