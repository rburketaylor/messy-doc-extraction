"""Top-level `doc-extract` command with subcommands for each pipeline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_extract import (
    active,
    config,
    corrupt,
    doctor,
    evaluate,
    generate,
    prepare,
    run_all,
    teacher_labeler,
    train,
)


def _path(value: Path | str) -> str:
    return str(value)


def _run_generate(args: argparse.Namespace) -> int:
    generate.main([
        "--n-docs", str(args.n_docs),
        "--seed", str(args.seed),
        "--out", _path(args.out),
    ])
    return 0


def _run_corrupt(args: argparse.Namespace) -> int:
    corrupt.main(["--in", _path(args.inp), "--out", _path(args.out), "--seed", str(args.seed)])
    return 0


def _run_label(args: argparse.Namespace) -> int:
    teacher_labeler.main([
        "--in", _path(args.inp),
        "--out", _path(args.out),
        "--quarantine", _path(args.quarantine),
        "--model", args.model,
        "--seed", str(args.seed),
        "--max-tokens", str(args.max_tokens),
    ])
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    prepare.main([
        "--in", _path(args.inp),
        "--out-dir", _path(args.out_dir),
        "--seed", str(args.seed),
        "--split", str(args.split),
    ])
    return 0


def _run_train(args: argparse.Namespace) -> int:
    argv = [
        "--train-file", _path(args.train_file),
        "--base", args.base,
        "--revision", args.revision,
        "--adapter-dir", _path(args.adapter_dir),
        "--merged-dir", _path(args.merged_dir),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--max-length", str(args.max_length),
    ]
    if args.skip_merge:
        argv.append("--skip-merge")
    train.main(argv)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    evaluate.main([
        "--test-file", _path(args.test_file),
        "--base", args.base,
        "--ft", _path(args.ft),
        "--out", _path(args.out),
        "--max-new-tokens", str(args.max_new_tokens),
    ])
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    evaluate.main([
        "--test-file", _path(args.test_file),
        "--base", args.base,
        "--ft", args.base,
        "--out", _path(args.out),
        "--max-new-tokens", str(args.max_new_tokens),
    ])
    return 0


def _run_split_gold(args: argparse.Namespace) -> int:
    manifest = active.split_gold(
        in_path=args.inp,
        train_out=args.train_out,
        test_out=args.test_out,
        seed=args.seed,
        split=args.split,
    )
    print(json.dumps(manifest))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    manifest = active.predict(
        input_path=args.inp,
        out_path=args.out,
        run_name=args.run,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(manifest))
    return 0


def _run_mine_failures(args: argparse.Namespace) -> int:
    manifest = active.mine_failures(
        train_pool_path=args.train_pool,
        general_predictions_path=args.general_predictions,
        extract_predictions_path=args.extract_predictions,
        out_path=args.out,
        max_labels=args.max_labels,
    )
    print(json.dumps(manifest))
    return 0


def _run_label_hard(args: argparse.Namespace) -> int:
    client = teacher_labeler._make_client()
    counts = active.label_hard_batch(
        client=client,
        in_path=args.inp,
        out_path=args.out,
        quarantine_path=args.quarantine,
        model=args.model,
        max_tokens=args.max_tokens,
        max_labels=args.max_labels,
        seed=args.seed,
    )
    print(json.dumps(counts))
    return 1 if counts["transport_failed"] > 0 else 0


def _run_prepare_active(args: argparse.Namespace) -> int:
    manifest = active.prepare_active_sft(
        train_pool_path=args.train_pool,
        hard_labels_path=args.hard_labels,
        test_gold_path=args.test_gold,
        out_dir=args.out_dir,
        seed=args.seed,
        hard_per_easy=args.hard_per_easy,
    )
    print(json.dumps(manifest))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    result = active.compare_runs(
        test_file=args.test_file,
        hard_file=args.hard_file,
        out_path=args.out,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2))
    return 0


def _run_all(args: argparse.Namespace) -> int:
    run_all.run_all(n_docs=args.n_docs, seed=args.seed)
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    return doctor.cli([
        *(["--require-api"] if args.require_api else []),
        *(["--require-gpu"] if args.require_gpu else []),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-extract", description="Invoice extraction pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate clean synthetic invoices")
    p.add_argument("--n-docs", type=int, default=config.DEFAULT_N_DOCS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", type=Path, default=config.CLEAN_JSONL)
    p.set_defaults(func=_run_generate)

    p = sub.add_parser("corrupt", help="corrupt clean invoices into dirty text")
    p.add_argument("--in", dest="inp", type=Path, default=config.CLEAN_JSONL)
    p.add_argument("--out", type=Path, default=config.DIRTY_JSONL)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.set_defaults(func=_run_corrupt)

    p = sub.add_parser("label", help="teacher-label dirty invoices")
    p.add_argument("--in", dest="inp", type=Path, default=config.DIRTY_JSONL)
    p.add_argument("--out", type=Path, default=config.LABELED_JSONL)
    p.add_argument("--quarantine", type=Path, default=config.QUARANTINE_JSONL)
    p.add_argument("--model", default=config.TEACHER_MODEL_ID)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-tokens", type=int, default=config.TEACHER_MAX_TOKENS)
    p.set_defaults(func=_run_label)

    p = sub.add_parser("prepare", help="prepare SFT train/test data")
    p.add_argument("--in", dest="inp", type=Path, default=config.LABELED_JSONL)
    p.add_argument("--out-dir", type=Path, default=config.SFT_DIR)
    p.add_argument("--seed", type=int, default=config.DATA_SEED)
    p.add_argument("--split", type=float, default=config.TRAIN_SPLIT)
    p.set_defaults(func=_run_prepare)

    p = sub.add_parser("train", help="QLoRA SFT training")
    p.add_argument("--train-file", type=Path, default=config.SFT_DIR / "train.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--revision", default=config.STUDENT_REVISION)
    p.add_argument("--adapter-dir", type=Path, default=config.ADAPTER_DIR)
    p.add_argument("--merged-dir", type=Path, default=config.MERGED_DIR)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--skip-merge", action="store_true")
    p.set_defaults(func=_run_train)

    p = sub.add_parser("evaluate", help="evaluate base vs fine-tuned model")
    p.add_argument("--test-file", type=Path, default=config.SFT_DIR / "test.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--ft", default=config.MERGED_DIR)
    p.add_argument("--out", type=Path, default=config.METRICS_PATH)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.set_defaults(func=_run_evaluate)

    p = sub.add_parser("baseline", help="evaluate the base model against itself")
    p.add_argument("--test-file", type=Path, default=config.SFT_DIR / "test.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--out", type=Path, default=config.METRICS_PATH)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.set_defaults(func=_run_baseline)

    p = sub.add_parser("split-gold", help="split dirty docs into train pool and frozen gold test")
    p.add_argument("--in", dest="inp", type=Path, default=config.DIRTY_JSONL)
    p.add_argument("--train-out", type=Path, default=config.TRAIN_POOL_JSONL)
    p.add_argument("--test-out", type=Path, default=config.TEST_GOLD_JSONL)
    p.add_argument("--seed", type=int, default=config.DATA_SEED)
    p.add_argument("--split", type=float, default=config.GOLD_TRAIN_SPLIT)
    p.set_defaults(func=_run_split_gold)

    p = sub.add_parser("predict", help="write raw model predictions for an active-learning run")
    p.add_argument("--in", dest="inp", type=Path, default=config.TRAIN_POOL_JSONL)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", choices=sorted(active.model_run_specs()), required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.set_defaults(func=_run_predict)

    p = sub.add_parser("mine-failures", help="rank hard train-pool records from base predictions")
    p.add_argument("--train-pool", type=Path, default=config.TRAIN_POOL_JSONL)
    p.add_argument(
        "--general-predictions",
        type=Path,
        default=config.PREDICTIONS_DIR / "base_general_train_pool.jsonl",
    )
    p.add_argument(
        "--extract-predictions",
        type=Path,
        default=config.PREDICTIONS_DIR / "base_extract_train_pool.jsonl",
    )
    p.add_argument("--out", type=Path, default=config.HARD_CASES_JSONL)
    p.add_argument("--max-labels", type=int, default=config.MAX_TEACHER_LABELS)
    p.set_defaults(func=_run_mine_failures)

    p = sub.add_parser("label-hard", help="DeepSeek-label mined hard cases with truth checks")
    p.add_argument("--in", dest="inp", type=Path, default=config.HARD_CASES_JSONL)
    p.add_argument("--out", type=Path, default=config.HARD_LABELED_JSONL)
    p.add_argument("--quarantine", type=Path, default=config.HARD_QUARANTINE_JSONL)
    p.add_argument("--model", default=config.TEACHER_MODEL_ID)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-tokens", type=int, default=config.TEACHER_MAX_TOKENS)
    p.add_argument("--max-labels", type=int, default=config.MAX_TEACHER_LABELS)
    p.set_defaults(func=_run_label_hard)

    p = sub.add_parser("prepare-active", help="build active SFT data from hard and easy labels")
    p.add_argument("--train-pool", type=Path, default=config.TRAIN_POOL_JSONL)
    p.add_argument("--hard-labels", type=Path, default=config.HARD_LABELED_JSONL)
    p.add_argument("--test-gold", type=Path, default=config.TEST_GOLD_JSONL)
    p.add_argument("--out-dir", type=Path, default=config.ACTIVE_SFT_DIR)
    p.add_argument("--seed", type=int, default=config.DATA_SEED)
    p.add_argument("--hard-per-easy", type=int, default=config.HARD_PER_EASY)
    p.set_defaults(func=_run_prepare_active)

    p = sub.add_parser("compare", help="evaluate the four-run LiquidAI comparison matrix")
    p.add_argument("--test-file", type=Path, default=config.TEST_GOLD_JSONL)
    p.add_argument("--hard-file", type=Path, default=config.HARD_CASES_JSONL)
    p.add_argument("--out", type=Path, default=config.COMPARISON_METRICS_PATH)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.set_defaults(func=_run_compare)

    p = sub.add_parser("run-all", help="run the full API/GPU learning loop")
    p.add_argument("--n-docs", type=int, default=config.DEFAULT_N_DOCS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.set_defaults(func=_run_all)

    p = sub.add_parser("doctor", help="check local prerequisites")
    p.add_argument("--require-api", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    p.set_defaults(func=_run_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
