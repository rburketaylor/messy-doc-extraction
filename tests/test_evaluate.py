"""Fast evaluation harness tests; no model loading."""

from __future__ import annotations

import copy
import json

from doc_extract import evaluate
from doc_extract.jsonl import sidecar_manifest_path

_GOLD = {
    "vendor_name": "Acme",
    "buyer_name": None,
    "invoice_number": "INV-1",
    "invoice_date": "2026-06-25",
    "currency": "USD",
    "purchase_order_number": None,
    "subtotal": "300.00",
    "tax_total": None,
    "shipping_total": None,
    "discount_total": None,
    "grand_total": "300.00",
    "line_items": [
        {
            "description": "Widget A",
            "quantity": "1",
            "unit": "EA",
            "unit_price": "100.00",
            "amount": "100.00",
        },
        {
            "description": "Widget B",
            "quantity": "2",
            "unit": "EA",
            "unit_price": "100.00",
            "amount": "200.00",
        },
    ],
}


def _raw(obj):
    return json.dumps(obj)


def test_json_fence_stripping_allows_exact_record_scoring():
    pred = "```json\n" + _raw(_GOLD) + "\n```"

    metrics = evaluate.compute_metrics([pred], [_GOLD])

    assert metrics["parse_rate"] == 1.0
    assert metrics["schema_pass_rate"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["record_exact"] == 1.0


def test_parse_and_schema_failures_are_hard_gates():
    malformed = "{not json"
    extra = copy.deepcopy(_GOLD)
    extra["unexpected"] = "x"

    metrics = evaluate.compute_metrics([malformed, _raw(extra)], [_GOLD, _GOLD])

    assert metrics["parse_rate"] == 0.5
    assert metrics["schema_pass_rate"] == 0.0
    assert metrics["micro_f1"] == 0.0
    assert metrics["record_exact"] == 0.0


def test_nullable_null_gold_does_not_require_prediction():
    pred = copy.deepcopy(_GOLD)
    for key in ("buyer_name", "purchase_order_number", "tax_total", "shipping_total"):
        pred.pop(key)
    pred.pop("discount_total")

    metrics = evaluate.compute_metrics([_raw(pred)], [_GOLD])

    assert metrics["schema_pass_rate"] == 1.0
    assert metrics["record_exact"] == 1.0


def test_nullable_null_gold_penalizes_spurious_non_null_prediction():
    pred = copy.deepcopy(_GOLD)
    pred["buyer_name"] = "Not on invoice"

    metrics = evaluate.compute_metrics([_raw(pred)], [_GOLD])

    assert metrics["schema_pass_rate"] == 1.0
    assert metrics["record_exact"] == 0.0
    assert metrics["micro_f1"] < 1.0


def test_line_items_match_independent_of_order():
    pred = copy.deepcopy(_GOLD)
    pred["line_items"] = list(reversed(pred["line_items"]))

    metrics = evaluate.compute_metrics([_raw(pred)], [_GOLD])

    assert metrics["record_exact"] == 1.0
    assert evaluate.per_record_exact([_raw(pred)], [_GOLD]) == [1.0]


def test_per_record_exact_scores_non_exact_records():
    pred = copy.deepcopy(_GOLD)
    pred["grand_total"] = "301.00"

    assert evaluate.per_record_exact([_raw(_GOLD), _raw(pred)], [_GOLD, _GOLD]) == [1.0, 0.0]


def test_bootstrap_outputs_are_deterministic_for_seed():
    base_counts = [(1, 1, 0), (2, 0, 1), (0, 0, 2)]
    ft_counts = [(2, 0, 0), (2, 0, 0), (1, 0, 1)]
    deltas = [0.0, 1.0, -1.0]

    assert evaluate.paired_bootstrap_ci_corpus_f1(
        base_counts, ft_counts, n_boot=25, ci=0.8, seed=123
    ) == evaluate.paired_bootstrap_ci_corpus_f1(
        base_counts, ft_counts, n_boot=25, ci=0.8, seed=123
    )
    assert evaluate.paired_bootstrap_ci(deltas, n_boot=25, ci=0.8, seed=123) == (
        evaluate.paired_bootstrap_ci(deltas, n_boot=25, ci=0.8, seed=123)
    )


def test_evaluate_main_writes_metrics_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate.config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(evaluate, "load_records", lambda test_file: (["prompt"], [_GOLD]))
    monkeypatch.setattr(evaluate, "load_model", lambda spec: (spec, object()))
    monkeypatch.setattr(
        evaluate,
        "generate",
        lambda model, processor, prompts, max_new_tokens: [_raw(_GOLD)],
    )
    out_path = tmp_path / "metrics.json"

    evaluate.main([
        "--test-file", str(tmp_path / "test.jsonl"),
        "--base", "base-model",
        "--ft", str(tmp_path / "merged"),
        "--out", str(out_path),
        "--max-new-tokens", "12",
    ])

    metrics = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = json.loads(sidecar_manifest_path(out_path).read_text(encoding="utf-8"))
    assert metrics["learning_proven"] is False
    assert manifest["stage"] == "evaluate"
    assert manifest["counts"] == {"n_records": 1}
    assert manifest["base_model"] == "base-model"
    assert manifest["max_new_tokens"] == 12
