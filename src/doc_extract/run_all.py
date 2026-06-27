"""Phase 8: programmatic end-to-end runner — mirrors the Makefile with shared seed/config."""

from __future__ import annotations

from doc_extract import config, corrupt, evaluate, generate, prepare, teacher_labeler, train


def run_all(n_docs: int = config.DEFAULT_N_DOCS, seed: int = config.SEED) -> None:
    config.ensure_dirs()
    generate.main(["--n-docs", str(n_docs), "--seed", str(seed), "--out", str(config.CLEAN_JSONL)])
    corrupt.main(["--in", str(config.CLEAN_JSONL), "--out", str(config.DIRTY_JSONL),
                  "--seed", str(seed)])
    teacher_labeler.main(["--in", str(config.DIRTY_JSONL), "--out", str(config.LABELED_JSONL),
                          "--quarantine", str(config.QUARANTINE_JSONL)])
    prepare.main(["--in", str(config.LABELED_JSONL), "--out-dir", str(config.SFT_DIR),
                  "--seed", str(seed), "--split", str(config.TRAIN_SPLIT)])
    train.main([])
    evaluate.main([])


if __name__ == "__main__":
    run_all()
