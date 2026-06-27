"""Active-learning workflow tests; no API or model loading."""

from __future__ import annotations

import json

from doc_extract import active
from doc_extract.jsonl import load_jsonl, write_jsonl


def _invoice(n: int | str = 0) -> dict:
    return {
        "vendor_name": "Acme",
        "buyer_name": None,
        "invoice_number": f"INV-{n}",
        "invoice_date": "2026-06-25",
        "currency": "USD",
        "purchase_order_number": None,
        "subtotal": "100.00",
        "tax_total": None,
        "shipping_total": None,
        "discount_total": None,
        "grand_total": "100.00",
        "line_items": [
            {
                "description": "Widget",
                "quantity": "1",
                "unit": "EA",
                "unit_price": "100.00",
                "amount": "100.00",
            }
        ],
    }


def _dirty_row(i: int) -> dict:
    return {"id": f"doc-{i:03d}", "dirty_text": f"invoice text {i}", "clean_json": _invoice(i)}


def _raw(obj: dict) -> str:
    return json.dumps(obj)


def _wrong(inv: dict) -> dict:
    out = dict(inv)
    out["vendor_name"] = "Wrong Vendor"
    return out


def test_split_gold_is_deterministic_and_disjoint(tmp_path):
    dirty = tmp_path / "dirty.jsonl"
    write_jsonl(dirty, [_dirty_row(i) for i in range(10)])

    m1 = active.split_gold(
        in_path=dirty,
        train_out=tmp_path / "a" / "train_pool.jsonl",
        test_out=tmp_path / "a" / "test_gold.jsonl",
        seed=42,
        split=0.7,
    )
    active.split_gold(
        in_path=dirty,
        train_out=tmp_path / "b" / "train_pool.jsonl",
        test_out=tmp_path / "b" / "test_gold.jsonl",
        seed=42,
        split=0.7,
    )

    train_ids = {r["id"] for r in load_jsonl(tmp_path / "a" / "train_pool.jsonl")}
    test_ids = {r["id"] for r in load_jsonl(tmp_path / "a" / "test_gold.jsonl")}
    train_ids_b = {r["id"] for r in load_jsonl(tmp_path / "b" / "train_pool.jsonl")}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids == train_ids_b
    assert m1["counts"] == {"n_input": 10, "n_train": 7, "n_test": 3}


def test_mine_failures_ranks_parse_schema_before_low_f1_and_caps(tmp_path):
    pool = [_dirty_row(i) for i in range(3)]
    write_jsonl(tmp_path / "pool.jsonl", pool)

    schema_bad = dict(pool[1]["clean_json"])
    schema_bad["unexpected"] = "x"
    general_predictions = [
        {"id": "doc-000", "raw_output": _raw(_wrong(pool[0]["clean_json"]))},
        {"id": "doc-001", "raw_output": _raw(schema_bad)},
        {"id": "doc-002", "raw_output": "{bad json"},
    ]
    extract_predictions = [
        {"id": row["id"], "raw_output": _raw(row["clean_json"])}
        for row in pool
    ]
    write_jsonl(tmp_path / "general.jsonl", general_predictions)
    write_jsonl(tmp_path / "extract.jsonl", extract_predictions)

    active.mine_failures(
        train_pool_path=tmp_path / "pool.jsonl",
        general_predictions_path=tmp_path / "general.jsonl",
        extract_predictions_path=tmp_path / "extract.jsonl",
        out_path=tmp_path / "hard.jsonl",
        max_labels=2,
    )

    hard = load_jsonl(tmp_path / "hard.jsonl")
    assert [r["id"] for r in hard] == ["doc-002", "doc-001"]
    assert hard[0]["failures"]["base_general"]["failure_type"] == "parse"
    assert hard[1]["failures"]["base_general"]["failure_type"] == "schema"


def test_mine_failures_prioritizes_shared_failures(tmp_path):
    pool = [_dirty_row(0), _dirty_row(1)]
    write_jsonl(tmp_path / "pool.jsonl", pool)
    write_jsonl(
        tmp_path / "general.jsonl",
        [
            {"id": "doc-000", "raw_output": _raw(_wrong(pool[0]["clean_json"]))},
            {"id": "doc-001", "raw_output": "{bad json"},
        ],
    )
    write_jsonl(
        tmp_path / "extract.jsonl",
        [
            {"id": "doc-000", "raw_output": _raw(_wrong(pool[0]["clean_json"]))},
            {"id": "doc-001", "raw_output": _raw(pool[1]["clean_json"])},
        ],
    )

    active.mine_failures(
        train_pool_path=tmp_path / "pool.jsonl",
        general_predictions_path=tmp_path / "general.jsonl",
        extract_predictions_path=tmp_path / "extract.jsonl",
        out_path=tmp_path / "hard.jsonl",
        max_labels=1,
    )

    hard = load_jsonl(tmp_path / "hard.jsonl")
    assert [r["id"] for r in hard] == ["doc-000"]
    assert hard[0]["shared_failure"] is True


def test_label_hard_accepts_only_schema_valid_truth_matching_labels(tmp_path, monkeypatch):
    rows = [_dirty_row(i) for i in range(3)]
    write_jsonl(tmp_path / "hard.jsonl", rows)

    def fake_extract_invoice_json(**kwargs):
        doc_id = kwargs["doc_id"]
        if doc_id == "doc-000":
            return rows[0]["clean_json"]
        if doc_id == "doc-001":
            return rows[0]["clean_json"]
        return {"bad": "shape"}

    monkeypatch.setattr(active.teacher_labeler, "extract_invoice_json", fake_extract_invoice_json)

    counts = active.label_hard_batch(
        client=object(),
        in_path=tmp_path / "hard.jsonl",
        out_path=tmp_path / "accepted.jsonl",
        quarantine_path=tmp_path / "quarantine.jsonl",
        model="teacher",
        max_tokens=128,
        max_labels=3,
    )

    accepted = load_jsonl(tmp_path / "accepted.jsonl")
    quarantined = load_jsonl(tmp_path / "quarantine.jsonl")
    assert counts["labeled"] == 1
    assert counts["truth_rejected"] == 1
    assert counts["invalid_label"] == 1
    assert [r["id"] for r in accepted] == ["doc-000"]
    assert {r["id"] for r in quarantined} == {"doc-001", "doc-002"}


def test_prepare_active_sft_builds_hard_easy_mix_and_excludes_test_ids(tmp_path):
    train_pool = [_dirty_row(i) for i in range(6)]
    test_gold = [_dirty_row(i) for i in range(6, 8)]
    hard_labels = [
        {
            "id": row["id"],
            "input_text": row["dirty_text"],
            "output": row["clean_json"],
        }
        for row in train_pool[:4]
    ]
    write_jsonl(tmp_path / "train_pool.jsonl", train_pool)
    write_jsonl(tmp_path / "test_gold.jsonl", test_gold)
    write_jsonl(tmp_path / "hard_labeled.jsonl", hard_labels)

    manifest = active.prepare_active_sft(
        train_pool_path=tmp_path / "train_pool.jsonl",
        hard_labels_path=tmp_path / "hard_labeled.jsonl",
        test_gold_path=tmp_path / "test_gold.jsonl",
        out_dir=tmp_path / "sft",
        seed=42,
        hard_per_easy=2,
    )

    train_rows = load_jsonl(tmp_path / "sft" / "train.jsonl")
    assert manifest["counts"]["n_hard"] == 4
    assert manifest["counts"]["n_easy"] == 2
    assert manifest["counts"]["n_train"] == 6
    assert {r["source"] for r in train_rows} == {"hard", "easy"}
    assert {r["id"] for r in train_rows}.isdisjoint({"doc-006", "doc-007"})


def test_compare_runs_writes_all_four_runs_and_overfit_flag(tmp_path, monkeypatch):
    test_row = _dirty_row(0)
    hard_row = _dirty_row(1)
    write_jsonl(tmp_path / "test_gold.jsonl", [test_row])
    write_jsonl(tmp_path / "hard.jsonl", [hard_row])

    specs = active.model_run_specs()
    exact_test = _raw(test_row["clean_json"])
    exact_hard = _raw(hard_row["clean_json"])
    bad_test = _raw(_wrong(test_row["clean_json"]))
    bad_hard = _raw(_wrong(hard_row["clean_json"]))

    def fake_generate(model, processor, prompts, max_new_tokens):
        if model == specs["base_general"].model_path:
            return [bad_test, bad_hard]
        if model == specs["ft_general"].model_path:
            return [exact_test, exact_hard]
        if model == specs["base_extract"].model_path:
            return [exact_test, bad_hard]
        if model == specs["ft_extract"].model_path:
            return [bad_test, exact_hard]
        raise AssertionError(model)

    monkeypatch.setattr(active.evaluate, "load_model", lambda spec: (spec, object()))
    monkeypatch.setattr(active.evaluate, "generate", fake_generate)

    result = active.compare_runs(
        test_file=tmp_path / "test_gold.jsonl",
        hard_file=tmp_path / "hard.jsonl",
        out_path=tmp_path / "comparison.json",
        max_new_tokens=12,
    )

    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert set(result["runs"]) == set(active.DEFAULT_COMPARISON_RUNS)
    assert set(saved["runs"]) == set(active.DEFAULT_COMPARISON_RUNS)
    assert result["deltas"]["ft_general_vs_base_general"]["delta_micro_f1"] > 0
    assert result["overfit_flags"]["ft_extract"] is True
