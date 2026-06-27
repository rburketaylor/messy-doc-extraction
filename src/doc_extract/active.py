"""Active-learning workflow for the LiquidAI base-vs-extract study."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doc_extract import config, evaluate, teacher_labeler
from doc_extract.jsonl import (
    append_jsonl,
    load_jsonl,
    sidecar_manifest_path,
    write_jsonl,
    write_stage_manifest,
)
from doc_extract.prepare import format_extraction_prompt
from doc_extract.schema import SCHEMA_VERSION
from doc_extract.validation import InvoiceValidationError, validate_and_canonicalize_invoice


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    model_id: str
    adapter_dir: Path
    merged_dir: Path


@dataclass(frozen=True)
class ModelRunSpec:
    name: str
    family: str
    model_path: str
    checkpoint_kind: str


DEFAULT_COMPARISON_RUNS = ("base_general", "ft_general", "base_extract", "ft_extract")


def model_family_specs() -> dict[str, ModelFamilySpec]:
    return {
        "general": ModelFamilySpec(
            name="general",
            model_id=config.GENERAL_MODEL_ID,
            adapter_dir=config.GENERAL_ADAPTER_DIR,
            merged_dir=config.GENERAL_MERGED_DIR,
        ),
        "extract": ModelFamilySpec(
            name="extract",
            model_id=config.EXTRACT_MODEL_ID,
            adapter_dir=config.EXTRACT_ADAPTER_DIR,
            merged_dir=config.EXTRACT_MERGED_DIR,
        ),
    }


def model_run_specs() -> dict[str, ModelRunSpec]:
    families = model_family_specs()
    return {
        "base_general": ModelRunSpec(
            name="base_general",
            family="general",
            model_path=families["general"].model_id,
            checkpoint_kind="base",
        ),
        "ft_general": ModelRunSpec(
            name="ft_general",
            family="general",
            model_path=str(families["general"].merged_dir),
            checkpoint_kind="fine_tuned",
        ),
        "base_extract": ModelRunSpec(
            name="base_extract",
            family="extract",
            model_path=families["extract"].model_id,
            checkpoint_kind="base",
        ),
        "ft_extract": ModelRunSpec(
            name="ft_extract",
            family="extract",
            model_path=str(families["extract"].merged_dir),
            checkpoint_kind="fine_tuned",
        ),
    }


def split_gold(
    *, in_path: Path, train_out: Path, test_out: Path, seed: int, split: float
) -> dict[str, Any]:
    rows = sorted(load_jsonl(in_path), key=lambda r: str(r["id"]))
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_train = int(round(len(idx) * split))
    train_idx = set(idx[:n_train])
    train_rows = [rows[i] for i in range(len(rows)) if i in train_idx]
    test_rows = [rows[i] for i in range(len(rows)) if i not in train_idx]

    train_ids = {str(r["id"]) for r in train_rows}
    test_ids = {str(r["id"]) for r in test_rows}
    if train_ids & test_ids:
        raise ValueError("train/test split produced overlapping ids")

    write_jsonl(train_out, train_rows)
    write_jsonl(test_out, test_rows)
    counts = {"n_input": len(rows), "n_train": len(train_rows), "n_test": len(test_rows)}
    return write_stage_manifest(
        stage="split-gold",
        manifest_path=sidecar_manifest_path(train_out),
        schema_version=SCHEMA_VERSION,
        seed=seed,
        counts=counts,
        inputs={"dirty_jsonl": in_path},
        outputs={"train_pool_jsonl": train_out, "test_gold_jsonl": test_out},
        extra={"train_split": split, **counts},
    )


def _canonical_invoice(payload: Any) -> dict[str, Any]:
    return validate_and_canonicalize_invoice(payload)


def _load_prompts_golds_records(path: Path) -> tuple[list[str], list[dict[str, Any]], list[dict]]:
    rows = load_jsonl(path)
    prompts: list[str] = []
    golds: list[dict[str, Any]] = []
    for rec in rows:
        if "prompt" in rec and "completion" in rec:
            prompts.append(str(rec["prompt"]))
            golds.append(_canonical_invoice(json.loads(rec["completion"])))
            continue
        prompts.append(format_extraction_prompt(str(rec["dirty_text"])))
        golds.append(_canonical_invoice(rec["clean_json"]))
    return prompts, golds, rows


def _empty_cuda_cache() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def predict(
    *, input_path: Path, out_path: Path, run_name: str, max_new_tokens: int,
    model_path: str | None = None,
) -> dict[str, Any]:
    specs = model_run_specs()
    if run_name not in specs:
        raise ValueError(f"unknown model run: {run_name}")
    spec = specs[run_name]
    resolved_model = model_path or spec.model_path
    prompts, _golds, records = _load_prompts_golds_records(input_path)

    model = processor = None
    try:
        model, processor = evaluate.load_model(resolved_model)
        predictions = evaluate.generate(model, processor, prompts, max_new_tokens)
    finally:
        del model
        del processor
        _empty_cuda_cache()

    rows = [
        {
            "id": str(rec["id"]),
            "run": run_name,
            "family": spec.family,
            "model": str(resolved_model),
            "raw_output": raw,
        }
        for rec, raw in zip(records, predictions, strict=False)
    ]
    write_jsonl(out_path, rows)
    counts = {"n_records": len(rows)}
    return write_stage_manifest(
        stage="predict",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=config.SEED,
        counts=counts,
        inputs={"input_jsonl": input_path},
        outputs={"predictions_jsonl": out_path},
        extra={
            "run": run_name,
            "family": spec.family,
            "model": str(resolved_model),
            "max_new_tokens": max_new_tokens,
        },
    )


def _prediction_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in load_jsonl(path):
        raw = rec.get("raw_output", rec.get("prediction", rec.get("output", "")))
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        out[str(rec["id"])] = raw
    return out


def score_prediction(raw_output: str | None, gold: dict[str, Any]) -> dict[str, Any]:
    raw = raw_output or ""
    metrics = evaluate.compute_metrics([raw], [gold])
    if metrics["parse_rate"] < 1.0:
        failure_type = "parse"
        failure_rank = 0
    elif metrics["schema_pass_rate"] < 1.0:
        failure_type = "schema"
        failure_rank = 1
    elif metrics["record_exact"] < 1.0:
        failure_type = "low_f1"
        failure_rank = 2
    else:
        failure_type = "pass"
        failure_rank = 3
    return {
        **metrics,
        "failure_type": failure_type,
        "failure_rank": failure_rank,
        "failed": failure_rank < 3,
    }


def mine_failures(
    *,
    train_pool_path: Path,
    general_predictions_path: Path,
    extract_predictions_path: Path,
    out_path: Path,
    max_labels: int,
) -> dict[str, Any]:
    pool = load_jsonl(train_pool_path)
    general_preds = _prediction_map(general_predictions_path)
    extract_preds = _prediction_map(extract_predictions_path)
    candidates = []
    for rec in pool:
        doc_id = str(rec["id"])
        gold = _canonical_invoice(rec["clean_json"])
        general_score = score_prediction(general_preds.get(doc_id), gold)
        extract_score = score_prediction(extract_preds.get(doc_id), gold)
        if not (general_score["failed"] or extract_score["failed"]):
            continue
        shared = bool(general_score["failed"] and extract_score["failed"])
        best_rank = min(general_score["failure_rank"], extract_score["failure_rank"])
        mean_f1 = (general_score["micro_f1"] + extract_score["micro_f1"]) / 2
        candidates.append((
            (0 if shared else 1, best_rank, mean_f1, doc_id),
            {
                **rec,
                "failures": {
                    "base_general": general_score,
                    "base_extract": extract_score,
                },
                "shared_failure": shared,
            },
        ))

    selected = []
    for priority, (_sort_key, rec) in enumerate(sorted(candidates, key=lambda x: x[0])):
        if priority >= max_labels:
            break
        selected.append({**rec, "priority": priority})
    write_jsonl(out_path, selected)

    counts = {
        "n_train_pool": len(pool),
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "max_labels": max_labels,
    }
    return write_stage_manifest(
        stage="mine-failures",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=config.SEED,
        counts=counts,
        inputs={
            "train_pool_jsonl": train_pool_path,
            "general_predictions_jsonl": general_predictions_path,
            "extract_predictions_jsonl": extract_predictions_path,
        },
        outputs={"hard_cases_jsonl": out_path},
        extra=counts,
    )


def _truth_matches(label: dict[str, Any], truth: dict[str, Any]) -> bool:
    return _canonical_invoice(label) == _canonical_invoice(truth)


def label_hard_batch(
    *,
    client: Any,
    in_path: Path,
    out_path: Path,
    quarantine_path: Path,
    model: str,
    max_tokens: int,
    max_labels: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    records = load_jsonl(in_path)
    if max_labels is not None:
        records = records[:max_labels]
    seen = teacher_labeler._load_seen_ids(out_path, quarantine_path)
    processed = labeled = truth_rejected = invalid_label = 0
    semantic_failed = transport_failed = skipped = 0
    total_usage = teacher_labeler.new_usage_totals()
    labeled_usage = teacher_labeler.new_usage_totals()
    rejected_usage = teacher_labeler.new_usage_totals()
    for rec in records:
        processed += 1
        doc_id = str(rec["id"])
        if doc_id in seen:
            skipped += 1
            continue
        dirty_text = str(rec["dirty_text"])
        rec_usage: list[dict[str, int]] = []
        try:
            payload = teacher_labeler.extract_invoice_json(
                client=client,
                doc_id=doc_id,
                dirty_text=dirty_text,
                quarantine_path=quarantine_path,
                model=model,
                max_tokens=max_tokens,
                usage_log=rec_usage,
            )
            try:
                truth_matches = _truth_matches(payload, rec["clean_json"])
            except InvoiceValidationError as exc:
                append_jsonl(
                    quarantine_path,
                    {
                        "id": doc_id,
                        "model": model,
                        "input_text": dirty_text,
                        "error": "invalid_label",
                        "validation_error": str(exc),
                        "output": payload,
                        "gold": rec["clean_json"],
                        "ts": datetime.now(UTC).isoformat(),
                    },
                    fsync=True,
                )
                seen.add(doc_id)
                invalid_label += 1
                teacher_labeler.accumulate_usage(rejected_usage, rec_usage)
                teacher_labeler.accumulate_usage(total_usage, rec_usage)
                continue
            if not truth_matches:
                append_jsonl(
                    quarantine_path,
                    {
                        "id": doc_id,
                        "model": model,
                        "input_text": dirty_text,
                        "error": "truth_mismatch",
                        "output": payload,
                        "gold": rec["clean_json"],
                        "ts": datetime.now(UTC).isoformat(),
                    },
                    fsync=True,
                )
                seen.add(doc_id)
                truth_rejected += 1
                teacher_labeler.accumulate_usage(rejected_usage, rec_usage)
                teacher_labeler.accumulate_usage(total_usage, rec_usage)
                continue
            append_jsonl(
                out_path,
                {
                    "id": doc_id,
                    "model": model,
                    "input_text": dirty_text,
                    "output": payload,
                    "source": "hard",
                    "ts": datetime.now(UTC).isoformat(),
                },
                fsync=True,
            )
            seen.add(doc_id)
            labeled += 1
            teacher_labeler.accumulate_usage(labeled_usage, rec_usage)
            teacher_labeler.accumulate_usage(total_usage, rec_usage)
        except teacher_labeler.SemanticExtractionError:
            seen.add(doc_id)
            semantic_failed += 1
            teacher_labeler.accumulate_usage(rejected_usage, rec_usage)
            teacher_labeler.accumulate_usage(total_usage, rec_usage)
        except teacher_labeler.RetryableTransportError:
            transport_failed += 1
            teacher_labeler.accumulate_usage(total_usage, rec_usage)
    counts = {
        "processed": processed,
        "labeled": labeled,
        "truth_rejected": truth_rejected,
        "invalid_label": invalid_label,
        "semantic_failed": semantic_failed,
        "transport_failed": transport_failed,
        "skipped": skipped,
        "token_usage": {
            "total": total_usage,
            "labeled": labeled_usage,
            "rejected": rejected_usage,
        },
    }
    write_stage_manifest(
        stage="label-hard",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=seed,
        counts=counts,
        inputs={"hard_cases_jsonl": in_path},
        outputs={"hard_labeled_jsonl": out_path, "quarantine_jsonl": quarantine_path},
        extra={"model": model, "max_tokens": max_tokens, "max_labels": max_labels},
    )
    return counts


def _to_sft_record(doc_id: str, input_text: str, output: dict[str, Any]) -> dict[str, str]:
    return {
        "id": doc_id,
        "prompt": format_extraction_prompt(input_text),
        "completion": json.dumps(output, ensure_ascii=False),
    }


def _load_test_ids(test_gold_path: Path | None) -> set[str]:
    if test_gold_path is None or not Path(test_gold_path).exists():
        return set()
    return {str(rec["id"]) for rec in load_jsonl(test_gold_path)}


def prepare_active_sft(
    *,
    train_pool_path: Path,
    hard_labels_path: Path,
    out_dir: Path,
    seed: int,
    hard_per_easy: int,
    test_gold_path: Path | None = None,
) -> dict[str, Any]:
    if hard_per_easy <= 0:
        raise ValueError("hard_per_easy must be positive")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pool = load_jsonl(train_pool_path)
    pool_by_id = {str(rec["id"]): rec for rec in train_pool}
    test_ids = _load_test_ids(test_gold_path)

    hard_examples = []
    filtered_hard = 0
    for label in load_jsonl(hard_labels_path):
        doc_id = str(label["id"])
        if doc_id in test_ids:
            raise ValueError(f"test id appeared in hard labels: {doc_id}")
        source = pool_by_id.get(doc_id)
        if source is None:
            filtered_hard += 1
            continue
        try:
            canonical = _canonical_invoice(label["output"])
            if not _truth_matches(canonical, source["clean_json"]):
                filtered_hard += 1
                continue
        except InvoiceValidationError:
            filtered_hard += 1
            continue
        hard_examples.append(_to_sft_record(doc_id, str(source["dirty_text"]), canonical))

    hard_ids = {rec["id"] for rec in hard_examples}
    easy_candidates = [
        rec for rec in train_pool
        if str(rec["id"]) not in hard_ids and str(rec["id"]) not in test_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(easy_candidates)
    n_easy = min(len(easy_candidates), len(hard_examples) // hard_per_easy)
    easy_examples = [
        _to_sft_record(
            str(rec["id"]),
            str(rec["dirty_text"]),
            _canonical_invoice(rec["clean_json"]),
        )
        for rec in easy_candidates[:n_easy]
    ]

    sft_rows = [{**rec, "source": "hard"} for rec in hard_examples]
    sft_rows += [{**rec, "source": "easy"} for rec in easy_examples]
    rng.shuffle(sft_rows)
    train_path = out_dir / "train.jsonl"
    write_jsonl(train_path, sft_rows)

    outputs: dict[str, Path | str | None] = {"train_jsonl": train_path}
    n_test_reference = 0
    if test_gold_path is not None and Path(test_gold_path).exists():
        _prompts, _golds, test_rows = _load_prompts_golds_records(test_gold_path)
        test_sft = [
            _to_sft_record(
                str(rec["id"]),
                str(rec["dirty_text"]),
                _canonical_invoice(rec["clean_json"]),
            )
            for rec in test_rows
        ]
        n_test_reference = len(test_sft)
        test_path = out_dir / "test.jsonl"
        write_jsonl(test_path, test_sft)
        outputs["test_jsonl"] = test_path

    train_ids = {str(rec["id"]) for rec in sft_rows}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"test ids appeared in active train data: {sorted(overlap)[:3]}")

    counts = {
        "n_hard": len(hard_examples),
        "n_easy": len(easy_examples),
        "n_train": len(sft_rows),
        "n_filtered_hard": filtered_hard,
        "n_test_reference": n_test_reference,
    }
    return write_stage_manifest(
        stage="prepare-active",
        manifest_path=out_dir / "manifest.json",
        schema_version=SCHEMA_VERSION,
        seed=seed,
        counts=counts,
        inputs={
            "train_pool_jsonl": train_pool_path,
            "hard_labels_jsonl": hard_labels_path,
            "test_gold_jsonl": test_gold_path,
        },
        outputs=outputs,
        extra={"hard_per_easy": hard_per_easy, **counts},
    )


def _delta_metrics(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    fields = ("parse_rate", "schema_pass_rate", "micro_f1", "macro_f1", "record_exact")
    return {f"delta_{field}": float(candidate[field] - base[field]) for field in fields}


def _paired_comparison(
    *, base_predictions: list[str], candidate_predictions: list[str], golds: list[dict[str, Any]]
) -> dict[str, Any]:
    base_metrics = evaluate.compute_metrics(base_predictions, golds)
    candidate_metrics = evaluate.compute_metrics(candidate_predictions, golds)
    base_counts = evaluate.per_record_counts(base_predictions, golds)
    candidate_counts = evaluate.per_record_counts(candidate_predictions, golds)
    base_exact = evaluate.per_record_exact(base_predictions, golds)
    candidate_exact = evaluate.per_record_exact(candidate_predictions, golds)
    exact_deltas = [c - b for c, b in zip(candidate_exact, base_exact, strict=False)]
    return {
        **_delta_metrics(base_metrics, candidate_metrics),
        "paired_bootstrap_ci_micro_f1": evaluate.paired_bootstrap_ci_corpus_f1(
            base_counts,
            candidate_counts,
            config.N_BOOTSTRAP,
            config.BOOTSTRAP_CI,
            config.SEED,
        ),
        "paired_bootstrap_ci_record_exact": evaluate.paired_bootstrap_ci(
            exact_deltas,
            config.N_BOOTSTRAP,
            config.BOOTSTRAP_CI,
            config.SEED,
        ),
    }


def compare_runs(
    *,
    test_file: Path,
    out_path: Path,
    hard_file: Path | None = None,
    max_new_tokens: int = 1024,
    run_names: tuple[str, ...] = DEFAULT_COMPARISON_RUNS,
) -> dict[str, Any]:
    specs = model_run_specs()
    unknown = [name for name in run_names if name not in specs]
    if unknown:
        raise ValueError(f"unknown comparison runs: {', '.join(unknown)}")

    test_prompts, test_golds, _test_records = _load_prompts_golds_records(test_file)
    hard_prompts: list[str] = []
    hard_golds: list[dict[str, Any]] = []
    if hard_file is not None and Path(hard_file).exists():
        hard_prompts, hard_golds, _hard_records = _load_prompts_golds_records(hard_file)
    prompts = test_prompts + hard_prompts

    all_predictions: dict[str, list[str]] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    hard_metrics: dict[str, dict[str, Any]] = {}
    for run_name in run_names:
        spec = specs[run_name]
        model = processor = None
        try:
            model, processor = evaluate.load_model(spec.model_path)
            predictions = evaluate.generate(model, processor, prompts, max_new_tokens)
        finally:
            del model
            del processor
            _empty_cuda_cache()
        all_predictions[run_name] = predictions
        test_predictions = predictions[:len(test_prompts)]
        hard_predictions = predictions[len(test_prompts):]
        test_metrics[run_name] = evaluate.compute_metrics(test_predictions, test_golds)
        if hard_golds:
            hard_metrics[run_name] = evaluate.compute_metrics(hard_predictions, hard_golds)

    deltas: dict[str, dict[str, Any]] = {}
    pairs = {
        "ft_general_vs_base_general": ("base_general", "ft_general"),
        "base_extract_vs_base_general": ("base_general", "base_extract"),
        "ft_extract_vs_base_extract": ("base_extract", "ft_extract"),
    }
    for label, (base_name, candidate_name) in pairs.items():
        if base_name not in all_predictions or candidate_name not in all_predictions:
            continue
        deltas[label] = _paired_comparison(
            base_predictions=all_predictions[base_name][:len(test_prompts)],
            candidate_predictions=all_predictions[candidate_name][:len(test_prompts)],
            golds=test_golds,
        )

    overfit_flags: dict[str, bool] = {}
    if {"base_extract", "ft_extract"} <= set(hard_metrics):
        train_improved = (
            hard_metrics["ft_extract"]["micro_f1"] > hard_metrics["base_extract"]["micro_f1"]
        )
        test_worsened = (
            test_metrics["ft_extract"]["micro_f1"] < test_metrics["base_extract"]["micro_f1"]
            or test_metrics["ft_extract"]["record_exact"]
            < test_metrics["base_extract"]["record_exact"]
        )
        overfit_flags["ft_extract"] = bool(train_improved and test_worsened)

    result = {
        "headline_dataset": "test_gold",
        "runs": test_metrics,
        "deltas": deltas,
        "train_hard_diagnostic": {
            "diagnostic_only": True,
            "runs": hard_metrics,
        },
        "overfit_flags": overfit_flags,
        "model_runs": {
            name: {
                "family": specs[name].family,
                "model": specs[name].model_path,
                "checkpoint_kind": specs[name].checkpoint_kind,
            }
            for name in run_names
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_stage_manifest(
        stage="compare",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=config.SEED,
        counts={"n_test": len(test_golds), "n_train_hard": len(hard_golds)},
        inputs={"test_gold_jsonl": test_file, "hard_cases_jsonl": hard_file},
        outputs={"comparison_metrics_json": out_path},
        extra={"runs": list(run_names), "max_new_tokens": max_new_tokens},
    )
    return result
