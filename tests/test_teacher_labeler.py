"""Teacher labeling resumability and failure semantics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from doc_extract import teacher_labeler
from doc_extract.jsonl import sidecar_manifest_path

_VALID = {
    "vendor_name": "Acme",
    "buyer_name": None,
    "invoice_number": "INV-1",
    "invoice_date": "2026-06-25",
    "currency": "USD",
    "purchase_order_number": None,
    "subtotal": None,
    "tax_total": None,
    "shipping_total": None,
    "discount_total": None,
    "grand_total": "100.00",
    "line_items": [{
        "description": "w",
        "quantity": "1",
        "unit": None,
        "unit_price": "100.00",
        "amount": "100.00",
    }],
}


def _write_dirty(path, ids):
    with path.open("w", encoding="utf-8") as f:
        for doc_id in ids:
            f.write(json.dumps({"id": doc_id, "dirty_text": f"text {doc_id}"}) + "\n")


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def test_transport_failure_preserves_partial_output_and_leaves_failed_id_unseen(
    tmp_path, monkeypatch
):
    in_path = tmp_path / "dirty.jsonl"
    out_path = tmp_path / "labeled.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    _write_dirty(in_path, ["doc-0", "doc-1"])

    def fake_extract_invoice_json(**kwargs):
        if kwargs["doc_id"] == "doc-1":
            raise teacher_labeler.RetryableTransportError("timeout")
        return _VALID

    monkeypatch.setattr(teacher_labeler, "extract_invoice_json", fake_extract_invoice_json)

    counts = teacher_labeler.label_batch(
        client=object(),
        in_path=in_path,
        out_path=out_path,
        quarantine_path=quarantine_path,
        model="teacher",
        max_tokens=128,
    )

    assert counts["labeled"] == 1
    assert counts["transport_failed"] == 1
    manifest = json.loads(sidecar_manifest_path(out_path).read_text(encoding="utf-8"))
    assert manifest["stage"] == "label"
    assert manifest["counts"]["transport_failed"] == 1
    assert manifest["outputs"]["labeled_jsonl"] == str(out_path)
    assert [r["id"] for r in _read_jsonl(out_path)] == ["doc-0"]
    assert teacher_labeler._load_seen_ids(out_path, quarantine_path) == {"doc-0"}


def test_main_exits_nonzero_when_transport_failures_remain(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(teacher_labeler.config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(teacher_labeler, "_make_client", lambda: object())
    monkeypatch.setattr(
        teacher_labeler,
        "label_batch",
        lambda **kwargs: {
            "processed": 2,
            "labeled": 1,
            "quarantined": 0,
            "transport_failed": 1,
            "skipped": 0,
        },
    )

    with pytest.raises(SystemExit) as exc:
        teacher_labeler.main([
            "--in", str(tmp_path / "dirty.jsonl"),
            "--out", str(tmp_path / "labeled.jsonl"),
            "--quarantine", str(tmp_path / "quarantine.jsonl"),
        ])

    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["transport_failed"] == 1


def test_semantic_failure_is_quarantined_without_transport_failure(tmp_path, monkeypatch):
    in_path = tmp_path / "dirty.jsonl"
    out_path = tmp_path / "labeled.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    _write_dirty(in_path, ["doc-0"])

    def fake_extract_invoice_json(**kwargs):
        teacher_labeler._append_jsonl(kwargs["quarantine_path"], {
            "id": kwargs["doc_id"],
            "error": "bad json",
        })
        raise teacher_labeler.SemanticExtractionError("bad json")

    monkeypatch.setattr(teacher_labeler, "extract_invoice_json", fake_extract_invoice_json)

    counts = teacher_labeler.label_batch(
        client=object(),
        in_path=in_path,
        out_path=out_path,
        quarantine_path=quarantine_path,
        model="teacher",
        max_tokens=128,
    )

    assert counts["quarantined"] == 1
    assert counts["transport_failed"] == 0
    assert _read_jsonl(out_path) == []
    assert [r["id"] for r in _read_jsonl(quarantine_path)] == ["doc-0"]
    assert teacher_labeler._load_seen_ids(out_path, quarantine_path) == {"doc-0"}


def test_extract_repairs_after_first_semantic_failure(tmp_path, monkeypatch):
    calls = []

    def fake_call_teacher_retry(client, messages, *, model, max_tokens):
        calls.append(messages)
        if len(calls) == 1:
            return "{not json"
        return json.dumps(_VALID)

    monkeypatch.setattr(teacher_labeler, "_call_teacher_retry", fake_call_teacher_retry)

    out = teacher_labeler.extract_invoice_json(
        client=object(),
        doc_id="doc-0",
        dirty_text="invoice text",
        quarantine_path=tmp_path / "quarantine.jsonl",
        model="teacher",
        max_tokens=128,
    )

    assert out["vendor_name"] == "Acme"
    assert len(calls) == 2
    repair_prompt = calls[1][1]["content"]
    assert "previous model output:" in repair_prompt
    assert "{not json" in repair_prompt
    assert "validation error to fix:" in repair_prompt
    assert not (tmp_path / "quarantine.jsonl").exists()


def test_extract_quarantine_payload_contains_failure_context(tmp_path, monkeypatch):
    outputs = iter(["{not json", "{also bad"])

    def fake_call_teacher_retry(client, messages, *, model, max_tokens):
        return next(outputs)

    monkeypatch.setattr(teacher_labeler, "_call_teacher_retry", fake_call_teacher_retry)
    quarantine_path = tmp_path / "quarantine.jsonl"

    with pytest.raises(teacher_labeler.SemanticExtractionError):
        teacher_labeler.extract_invoice_json(
            client=object(),
            doc_id="doc-0",
            dirty_text="invoice text",
            quarantine_path=quarantine_path,
            model="teacher",
            max_tokens=128,
        )

    rows = _read_jsonl(quarantine_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "doc-0"
    assert row["model"] == "teacher"
    assert row["input_text"] == "invoice text"
    assert row["first_output"] == "{not json"
    assert row["second_output"] == "{also bad"
    assert "Malformed JSON" in row["first_error"]
    assert "Malformed JSON" in row["second_error"]
    assert row["ts"]


def test_call_teacher_rejects_non_stop_finish_reason():
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content=json.dumps(_VALID)),
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    with pytest.raises(teacher_labeler.SemanticExtractionError, match="finish_reason=length"):
        teacher_labeler._call_teacher(client, [], model="teacher", max_tokens=128)


def test_seen_id_scan_skips_malformed_prior_jsonl(tmp_path):
    path = tmp_path / "labeled.jsonl"
    path.write_text('{"id": "doc-0"}\n{bad json\n{"id": "doc-1"}\n', encoding="utf-8")

    assert teacher_labeler._load_seen_ids(path) == {"doc-0", "doc-1"}


def test_transport_error_propagates_without_quarantine(tmp_path, monkeypatch):
    def fake_call_teacher_retry(client, messages, *, model, max_tokens):
        raise teacher_labeler.RetryableTransportError("timeout")

    monkeypatch.setattr(teacher_labeler, "_call_teacher_retry", fake_call_teacher_retry)
    quarantine_path = tmp_path / "quarantine.jsonl"

    with pytest.raises(teacher_labeler.RetryableTransportError):
        teacher_labeler.extract_invoice_json(
            client=object(),
            doc_id="doc-0",
            dirty_text="invoice text",
            quarantine_path=quarantine_path,
            model="teacher",
            max_tokens=128,
        )

    assert not quarantine_path.exists()
