"""Generated invoices render internally consistent rounded totals."""

from __future__ import annotations

import json
from decimal import Decimal

from doc_extract import generate
from doc_extract.jsonl import sidecar_manifest_path


def _amount(value):
    return Decimal(value or "0.00")


def test_seed_42_generated_totals_match_displayed_rounded_fields(tmp_path):
    out_path = tmp_path / "clean.jsonl"
    generate.generate(100, 42, out_path)

    rows = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    manifest = json.loads(sidecar_manifest_path(out_path).read_text(encoding="utf-8"))
    assert manifest["stage"] == "generate"
    assert manifest["seed"] == 42
    assert manifest["counts"] == {"requested": 100, "written": 100}
    assert "clean_jsonl" in manifest["file_hashes"]["outputs"]
    assert rows
    for row in rows:
        inv = row["clean_json"]
        expected = (
            _amount(inv["subtotal"])
            + _amount(inv["tax_total"])
            + _amount(inv["shipping_total"])
            - _amount(inv["discount_total"])
        )
        assert _amount(inv["grand_total"]) == expected
