"""Phase 5: deterministic split reproducibility, canonicalization, and invalid filtering."""

from __future__ import annotations

import copy
import json

from doc_extract import prepare

_VALID = {
    "vendor_name": "Acme", "invoice_number": "INV-1", "invoice_date": "2026-06-25",
    "currency": "USD", "grand_total": "100.00",
    "line_items": [{"description": "w", "quantity": "1", "unit_price": "100.00",
                    "amount": "100.00"}],
}


def _row(i, output):
    return {"id": f"doc-{i:03d}", "model": "deepseek-v4-flash",
            "input_text": f"text {i}", "output": output}


def _write_labeled(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _ids(path):
    return sorted(json.loads(line)["id"] for line in path.open(encoding="utf-8"))


def test_filter_and_split(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    rows = [_row(i, dict(_VALID, invoice_number=f"INV-{i}")) for i in range(10)]
    rows.append(_row(99, dict(_VALID, invoice_date="25/06/2026")))   # invalid shape
    rows.append(_row(98, dict(_VALID, currency="XYZ")))              # invalid enum
    rows.append(_row(97, dict(_VALID, discount_total="-1.00")))      # uncanonicalizable amount
    rows.append(_row(96, dict(_VALID, grand_total="N/A")))           # unparseable amount
    word_qty = copy.deepcopy(_VALID)
    word_qty["line_items"][0]["quantity"] = "two"
    rows.append(_row(95, word_qty))                                  # unparseable quantity
    rows.append(_row(94, dict(_VALID, invoice_date="2026-02-31")))   # impossible date
    _write_labeled(labeled, rows)

    m = prepare.prepare(labeled, tmp_path / "out1", seed=42, split=0.8)
    assert m["stage"] == "prepare"
    assert m["n_filtered"] == 6
    assert m["counts"]["n_filtered"] == 6
    assert m["inputs"]["labeled_jsonl"] == str(labeled)
    assert m["n_train"] + m["n_test"] == 10
    train1, test1 = tmp_path / "out1" / "train.jsonl", tmp_path / "out1" / "test.jsonl"
    train_ids, test_ids = _ids(train1), _ids(test1)
    assert set(train_ids).isdisjoint(set(test_ids))     # held out
    assert "doc-099" not in train_ids and "doc-099" not in test_ids  # invalid excluded


def test_split_reproducible(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    rows = [_row(i, dict(_VALID, invoice_number=f"INV-{i}")) for i in range(10)]
    _write_labeled(labeled, rows)

    prepare.prepare(labeled, tmp_path / "a", seed=42, split=0.8)
    prepare.prepare(labeled, tmp_path / "b", seed=42, split=0.8)
    assert _ids(tmp_path / "a" / "train.jsonl") == _ids(tmp_path / "b" / "train.jsonl")
    assert _ids(tmp_path / "a" / "test.jsonl") == _ids(tmp_path / "b" / "test.jsonl")


def test_completions_are_strict_json(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    _write_labeled(labeled, [_row(0, _VALID)])
    prepare.prepare(labeled, tmp_path / "o", seed=42, split=1.0)
    rec = json.loads(next((tmp_path / "o" / "train.jsonl").open(encoding="utf-8")))
    assert set(rec) >= {"id", "prompt", "completion"}
    parsed = json.loads(rec["completion"])  # strict JSON, parses
    assert parsed["vendor_name"] == "Acme"
    assert not rec["completion"].lstrip().startswith("```")  # no markdown fence


def test_completion_uses_canonicalized_label(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    messy = copy.deepcopy(_VALID)
    messy.update({
        "vendor_name": "  Acme   Corp  ",
        "subtotal": "USD 1,000",
        "grand_total": "$1,000.0",
    })
    messy["line_items"][0].update({
        "description": "  Widget   labor ",
        "quantity": "2.00",
        "unit": "each",
        "unit_price": "$500",
        "amount": "1,000.0",
    })
    _write_labeled(labeled, [_row(0, messy)])

    m = prepare.prepare(labeled, tmp_path / "o", seed=42, split=1.0)
    assert m["n_filtered"] == 0
    rec = json.loads(next((tmp_path / "o" / "train.jsonl").open(encoding="utf-8")))
    parsed = json.loads(rec["completion"])

    assert parsed["vendor_name"] == "Acme Corp"
    assert parsed["subtotal"] == "1000.00"
    assert parsed["grand_total"] == "1000.00"
    assert parsed["line_items"][0]["description"] == "Widget labor"
    assert parsed["line_items"][0]["quantity"] == "2"
    assert parsed["line_items"][0]["unit"] == "EA"
    assert parsed["line_items"][0]["unit_price"] == "500.00"
    assert parsed["line_items"][0]["amount"] == "1000.00"
