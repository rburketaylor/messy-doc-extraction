"""Phase 7: field-level evaluation harness — 3-layer gate + paired bootstrap.

Layers: parse (json.loads) -> schema (INVOICE_JSON_SCHEMA) -> canonicalized-leaf micro-F1 (canon).
Line items matched optimally (scipy linear_sum_assignment) with path-aware per-leaf scoring.
Runs base vs fine-tuned; paired bootstrap CI on micro_f1 AND record_exact proves learning.
Reuses Phase 3 canon.py (single source of truth for normalization).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from scipy.optimize import linear_sum_assignment

from doc_extract import canon, config
from doc_extract.jsonl import load_jsonl, sidecar_manifest_path, write_stage_manifest
from doc_extract.prompting import format_prompt_for_generation
from doc_extract.schema import FIELD_TYPE_REGISTRY, INVOICE_JSON_SCHEMA, SCHEMA_VERSION

_VALIDATOR = Draft202012Validator(INVOICE_JSON_SCHEMA)
_FENCE_RE = re.compile(r"^\s*```(?:json)?|```\s*$", re.MULTILINE)
_MISMATCH = object()


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _flatten(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _flatten(v, f"{path}[{i}]")
    else:
        out.append((path, obj))
    return out


def _leaf_kind(path):
    return FIELD_TYPE_REGISTRY.get(re.sub(r"\[\d+\]", "[]", path), "string")


def _field_key(path):
    return re.sub(r"\[\d+\]", "[]", path)


def _canon_leaves(leaves, currency):
    norm = []
    for p, v in leaves:
        try:
            norm.append((p, canon.normalize_value(v, _leaf_kind(p), currency)))
        except Exception:
            norm.append((p, _MISMATCH))
    return norm


def _score_line_items(pred_items, gold_items, currency):
    pl = [dict(_canon_leaves(_flatten(it, "line_items[]"), currency)) for it in (pred_items or [])]
    gl = [dict(_canon_leaves(_flatten(it, "line_items[]"), currency)) for it in (gold_items or [])]
    tp = fp = fn = 0
    per = {}

    def bump(key, dt, dp, dn):
        t, f, n = per.get(key, (0, 0, 0))
        per[key] = (t + dt, f + dp, n + dn)

    def pair_cost(a, b):
        keys = set(a) | set(b)
        mism = 0
        for k in keys:
            if k in a and k in b:
                if a[k] != b[k]:
                    mism += 1
            else:
                mism += 1
        return mism

    if pl and gl:
        cost = np.array([[pair_cost(pa, gb) for gb in gl] for pa in pl])
        rows, cols = linear_sum_assignment(cost)
        matched_a, matched_b = set(rows.tolist()), set(cols.tolist())
        for a, b in zip(rows, cols, strict=False):
            pa, gb = pl[a], gl[b]
            for k in set(pa) | set(gb):
                fk = _field_key(k)
                if _leaf_kind(k).endswith("nullable") and gb.get(k) is None:
                    # nullable-null gold = non-target; only penalize a spurious non-null pred
                    if k in pa and pa[k] is not None and pa[k] is not _MISMATCH:
                        fp += 1
                        bump(fk, 0, 1, 0)
                    continue
                if k in pa and k in gb:
                    if pa[k] == gb[k]:
                        tp += 1
                        bump(fk, 1, 0, 0)
                    else:
                        fn += 1
                        fp += 1
                        bump(fk, 0, 0, 1)
                        bump(fk, 0, 1, 0)
                elif k in gb:
                    fn += 1
                    bump(fk, 0, 0, 1)
                else:
                    if pa[k] is _MISMATCH:
                        continue
                    fp += 1
                    bump(fk, 0, 1, 0)
        for b in range(len(gl)):
            if b not in matched_b:
                for k, v in gl[b].items():
                    if _leaf_kind(k).endswith("nullable") and v is None:
                        continue
                    fn += 1
                    bump(_field_key(k), 0, 0, 1)
        for a in range(len(pl)):
            if a not in matched_a:
                for k, v in pl[a].items():
                    if v is _MISMATCH:
                        continue
                    fp += 1
                    bump(_field_key(k), 0, 1, 0)
    else:
        for gb in gl:
            for k, v in gb.items():
                if _leaf_kind(k).endswith("nullable") and v is None:
                    continue
                fn += 1
                bump(_field_key(k), 0, 0, 1)
        for pa in pl:
            for k, v in pa.items():
                if v is _MISMATCH:
                    continue
                fp += 1
                bump(_field_key(k), 0, 1, 0)
    return tp, fp, fn, per


def _score_record(pred, gold):
    cur = gold.get("currency")
    gmap = dict(_canon_leaves(_flatten(gold, ""), cur))
    pmap = dict(_canon_leaves(_flatten(pred, ""), cur))
    tp = fp = fn = 0
    per = {}

    def bump(key, dt, dp, dn):
        t, f, n = per.get(key, (0, 0, 0))
        per[key] = (t + dt, f + dp, n + dn)

    for p, gv in gmap.items():
        if p.startswith("line_items"):
            continue
        kind = _leaf_kind(p)
        key = _field_key(p)
        pv = pmap.get(p)
        if kind.endswith("nullable") and gv is None:
            # nullable-null gold = non-target; only penalize a spurious non-null prediction
            if pv is not None and pv is not _MISMATCH:
                fp += 1
                bump(key, 0, 1, 0)
            continue
        if pv == gv:
            tp += 1
            bump(key, 1, 0, 0)
        else:
            fn += 1
            bump(key, 0, 0, 1)
            if pv is not None and pv is not _MISMATCH:
                fp += 1
                bump(key, 0, 1, 0)
    for p, pv in pmap.items():
        if p.startswith("line_items"):
            continue
        if p not in gmap and pv is not None and pv is not _MISMATCH:
            fp += 1
            bump(_field_key(p), 0, 1, 0)
    ltp, lfp, lfn, lper = _score_line_items(
        pred.get("line_items", []), gold.get("line_items", []), cur)
    for k, (a, b, c) in lper.items():
        bump(k, a, b, c)
    return tp + ltp, fp + lfp, fn + lfn, per


def compute_metrics(predictions, golds):
    n = len(golds)
    parsed = schema_ok = 0
    tp = fp = fn = 0
    perfect = 0
    per = {}

    def bump(key, dt, dp, dn):
        t, f, nn = per.get(key, (0, 0, 0))
        per[key] = (t + dt, f + dp, nn + dn)

    for raw, gold in zip(predictions, golds, strict=False):
        ok = True
        pred = None
        try:
            pred = json.loads(strip_fences(raw))
            parsed += 1
        except Exception:
            ok = False
        if ok:
            try:
                _VALIDATOR.validate(pred)
                schema_ok += 1
            except Exception:
                ok = False
        if not ok:
            # HARD GATE: ALL gold non-null leaves (scalars + line items) -> FN, path-bucketed.
            for p, v in _canon_leaves(_flatten(gold, ""), gold.get("currency")):
                if _leaf_kind(p).endswith("nullable") and v is None:
                    continue
                fn += 1
                bump(_field_key(p), 0, 0, 1)
            continue
        rtp, rfp, rfn, rper = _score_record(pred, gold)
        tp += rtp
        fp += rfp
        fn += rfn
        for k, (a, b, c) in rper.items():
            bump(k, a, b, c)
        if rfp == 0 and rfn == 0:
            perfect += 1
    denom = 2 * tp + fp + fn
    micro = (2 * tp) / denom if denom else 1.0
    f1s = []
    for a, b, c in per.values():
        d = 2 * a + b + c
        f1s.append((2 * a) / d if d else 1.0)
    macro = float(np.mean(f1s)) if f1s else 1.0
    return {"parse_rate": parsed / n if n else 0.0, "schema_pass_rate": schema_ok / n if n else 0.0,
            "micro_f1": micro, "macro_f1": macro, "record_exact": perfect / n if n else 0.0, "n": n}


def _gold_nonnull_leaf_count(gold):
    return sum(1 for p, v in _canon_leaves(_flatten(gold, ""), gold.get("currency"))
               if not (_leaf_kind(p).endswith("nullable") and v is None))


def per_record_counts(predictions, golds):
    """Per-record (tp, fp, fn) for corpus-micro-F1 bootstrap; parse/schema fail -> (0,0,all-FN)."""
    out = []
    for raw, gold in zip(predictions, golds, strict=False):
        ok = True
        pred = None
        try:
            pred = json.loads(strip_fences(raw))
        except Exception:
            ok = False
        if ok:
            try:
                _VALIDATOR.validate(pred)
            except Exception:
                ok = False
        if not ok:
            out.append((0, 0, _gold_nonnull_leaf_count(gold)))
            continue
        tp, fp, fn, _ = _score_record(pred, gold)
        out.append((tp, fp, fn))
    return out


def per_record_exact(predictions, golds):
    out = []
    for raw, gold in zip(predictions, golds, strict=False):
        ok = True
        pred = None
        try:
            pred = json.loads(strip_fences(raw))
        except Exception:
            ok = False
        if ok:
            try:
                _VALIDATOR.validate(pred)
            except Exception:
                ok = False
        if not ok:
            out.append(0.0)
            continue
        tp, fp, fn, _ = _score_record(pred, gold)
        out.append(1.0 if (fp == 0 and fn == 0) else 0.0)
    return out


def _corpus_f1(counts):
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    d = 2 * tp + fp + fn
    return (2 * tp) / d if d else 1.0


def paired_bootstrap_ci(deltas, n_boot, ci, seed):
    """Bootstrap the mean of per-record deltas (used for record_exact)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return [0.0, 0.0]
    boots = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(boots, (1 - ci) / 2 * 100)),
            float(np.percentile(boots, (1 + ci) / 2 * 100))]


def paired_bootstrap_ci_corpus_f1(base_counts, ft_counts, n_boot, ci, seed):
    """Bootstrap CORPUS micro-F1 delta (resample records, recompute corpus TP/FP/FN) so the CI
    aligns with the headline delta_micro_f1 rather than a mean-of-per-record statistic."""
    rng = np.random.default_rng(seed)
    n = len(base_counts)
    if n == 0:
        return [0.0, 0.0]
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bf = _corpus_f1([base_counts[i] for i in idx])
        ff = _corpus_f1([ft_counts[i] for i in idx])
        deltas.append(ff - bf)
    return [float(np.percentile(deltas, (1 - ci) / 2 * 100)),
            float(np.percentile(deltas, (1 + ci) / 2 * 100))]


def load_model(spec):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(spec, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        spec, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    return model, processor


def generate(model, processor, prompts, max_new_tokens):
    import torch
    outs = []
    for prompt in prompts:
        text = format_prompt_for_generation(processor, prompt)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outs.append(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0])
    return outs


def load_records(test_file):
    prompts, golds = [], []
    for rec in load_jsonl(test_file):
        prompts.append(rec["prompt"])
        golds.append(json.loads(rec["completion"]))
    return prompts, golds


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate base vs fine-tuned extraction")
    p.add_argument("--test-file", type=Path, default=config.SFT_DIR / "test.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--ft", default=config.MERGED_DIR)
    p.add_argument("--out", type=Path, default=config.METRICS_PATH)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args(argv)
    config.ensure_dirs()

    prompts, golds = load_records(args.test_file)
    base_mp = load_model(args.base)
    base_preds = generate(base_mp[0], base_mp[1], prompts, args.max_new_tokens)
    ft_mp = load_model(args.ft)
    ft_preds = generate(ft_mp[0], ft_mp[1], prompts, args.max_new_tokens)

    base_metrics = compute_metrics(base_preds, golds)
    ft_metrics = compute_metrics(ft_preds, golds)
    base_counts = per_record_counts(base_preds, golds)
    ft_counts = per_record_counts(ft_preds, golds)
    base_exact = per_record_exact(base_preds, golds)
    ft_exact = per_record_exact(ft_preds, golds)
    ci_f1 = paired_bootstrap_ci_corpus_f1(base_counts, ft_counts,
                                          config.N_BOOTSTRAP, config.BOOTSTRAP_CI, config.SEED)
    ci_re = paired_bootstrap_ci([f - b for f, b in zip(ft_exact, base_exact, strict=False)],
                                config.N_BOOTSTRAP, config.BOOTSTRAP_CI, config.SEED)
    result = {
        "base": base_metrics, "finetuned": ft_metrics,
        "delta_micro_f1": ft_metrics["micro_f1"] - base_metrics["micro_f1"],
        "delta_record_exact": ft_metrics["record_exact"] - base_metrics["record_exact"],
        "paired_bootstrap_ci_micro_f1": ci_f1,
        "paired_bootstrap_ci_record_exact": ci_re,
        "learning_proven": ci_f1[0] > 0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_stage_manifest(
        stage="evaluate",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=config.SEED,
        counts={"n_records": len(golds)},
        inputs={"test_jsonl": args.test_file},
        outputs={"metrics_json": out_path},
        extra={
            "base_model": str(args.base),
            "finetuned_model": str(args.ft),
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
