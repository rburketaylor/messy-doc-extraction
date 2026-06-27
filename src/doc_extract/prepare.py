"""Phase 5: turn teacher labels into SFT-ready data.

Load labeled.jsonl, re-validate each teacher output against the Invoice schema, filter invalids,
deterministically split train/test (seeded), format to {id, prompt, completion} (strict JSON), and
persist JSONL + an HF datasets snapshot + a split manifest.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset

from doc_extract import config
from doc_extract.jsonl import load_jsonl, write_jsonl, write_stage_manifest
from doc_extract.schema import SCHEMA_VERSION
from doc_extract.validation import InvoiceValidationError, validate_and_canonicalize_invoice

_INSTRUCTION = (
    "Extract the invoice fields from the document below and return ONLY a JSON object matching "
    "the invoice schema. Use null for fields not present. Do not include markdown or commentary."
)


def _load_labeled(in_path: Path) -> list[dict[str, Any]]:
    return load_jsonl(in_path)


def _filter_valid(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid, n_filtered = [], 0
    for r in rows:
        out = r.get("output")
        try:
            canonical = validate_and_canonicalize_invoice(out)
        except InvoiceValidationError:
            n_filtered += 1
            continue
        valid.append({**r, "output": canonical})
    return valid, n_filtered


def _split(valid: list[dict[str, Any]], seed: int, split: float) -> tuple[list, list]:
    ordered = sorted(valid, key=lambda r: str(r["id"]))
    idx = list(range(len(ordered)))
    random.Random(seed).shuffle(idx)
    n_train = int(round(len(idx) * split))
    train_idx, test_idx = set(idx[:n_train]), set(idx[n_train:])
    train = [ordered[i] for i in range(len(ordered)) if i in train_idx]
    test = [ordered[i] for i in range(len(ordered)) if i in test_idx]
    return train, test


def _to_sft(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        prompt = _INSTRUCTION + "\n\nDOCUMENT:\n" + r["input_text"]
        completion = json.dumps(r["output"], ensure_ascii=False)  # strict JSON, no fences
        out.append({"id": str(r["id"]), "prompt": prompt, "completion": completion})
    return out


def _to_hf_dataset(rows: list[dict[str, str]]) -> Dataset:
    # Explicit columns so an empty split (e.g. split=1.0) still snapshots as a valid 0-row
    # dataset. Dataset.from_list([]) is column-less and save_to_disk then fails schema inference.
    cols = ("id", "prompt", "completion")
    return Dataset.from_dict({c: [r[c] for r in rows] for c in cols})


def prepare(in_path: Path, out_dir: Path, seed: int, split: float) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_labeled(in_path)
    valid, n_filtered = _filter_valid(rows)
    train_rows, test_rows = _split(valid, seed, split)
    train_sft, test_sft = _to_sft(train_rows), _to_sft(test_rows)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    write_jsonl(train_path, train_sft)
    write_jsonl(test_path, test_sft)

    _to_hf_dataset(train_sft).save_to_disk(str(out_dir / "train_hf"))
    _to_hf_dataset(test_sft).save_to_disk(str(out_dir / "test_hf"))

    counts = {
        "n_loaded": len(rows),
        "n_filtered": n_filtered,
        "n_train": len(train_sft),
        "n_test": len(test_sft),
    }
    manifest = write_stage_manifest(
        stage="prepare",
        manifest_path=out_dir / "manifest.json",
        schema_version=SCHEMA_VERSION,
        seed=seed,
        counts=counts,
        inputs={"labeled_jsonl": in_path},
        outputs={
            "train_jsonl": train_path,
            "test_jsonl": test_path,
            "train_hf": out_dir / "train_hf",
            "test_hf": out_dir / "test_hf",
        },
        extra={"train_split": split, **counts},
    )
    return manifest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Prepare SFT dataset from teacher labels")
    p.add_argument("--in", dest="inp", type=Path, default=config.LABELED_JSONL)
    p.add_argument("--out-dir", type=Path, default=config.SFT_DIR)
    p.add_argument("--seed", type=int, default=config.DATA_SEED)
    p.add_argument("--split", type=float, default=config.TRAIN_SPLIT)
    args = p.parse_args(argv)
    config.ensure_dirs()
    manifest = prepare(args.inp, args.out_dir, args.seed, args.split)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
