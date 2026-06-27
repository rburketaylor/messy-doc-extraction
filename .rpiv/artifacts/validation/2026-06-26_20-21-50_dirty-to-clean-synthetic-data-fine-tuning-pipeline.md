---
template_version: 1
date: 2026-06-26T20:21:50-0300
author: Burke T
commit: no-commit
branch: main
repository: messy-doc-extraction
topic: "Validation of Dirty-to-clean synthetic data fine-tuning pipeline"
status: ready
verdict: pass
parent: ".rpiv/artifacts/plans/2026-06-25_21-35-07_dirty-data-extraction-finetune.md"
tags: [validation, plan, greenfield, fine-tuning, qlora, lfm2.5-vl, deepseek, structured-extraction, synthetic-data, evaluation, reproducibility, ruff]
last_updated: 2026-06-26T20:21:50-0300
---

## Validation Report: Dirty-to-clean synthetic data fine-tuning pipeline

> Re-validation run after the prior `fail` verdict's sole blocker (59 Ruff violations) was fixed.
> All offline automated gates now pass. The dependency-version deviation flagged previously is
> **accepted by the developer** (the plan's pinned versions are outdated) and is no longer an
> action item. Live API/GPU checks remain manual (unchanged from the prior run).

### Implementation Status

- ✓ Phase 1: Project scaffold + schema/config foundation — Fully implemented
- ✓ Phase 2: Clean synthetic invoice generator — Fully implemented
- ✓ Phase 3: Canonicalization + label-preserving corruption — Fully implemented
- ✓ Phase 4: Teacher labeling (DeepSeek) — Fully implemented for offline/verifiable behavior; live API smoke remains manual
- ✓ Phase 5: Dataset preparation — Fully implemented
- ✓ Phase 6: Training (QLoRA SFT) — Recipe implemented and offline checks pass; live GPU training remains manual
- ✓ Phase 7: Evaluation harness — Pure-logic harness implemented and checked; live base-vs-finetuned eval remains manual
- ✓ Phase 8: Orchestration + reproducibility + reflection — Fully implemented; end-to-end live run remains manual

Git evidence: the repository still has no commits (`git log` empty) and implementation files are
untracked, so commit-to-commit diff evidence is unavailable. Validation is based on working-tree
file inspection plus the plan's automated checks (re-run in full this session).

### Automated Verification Results

- ✓ **Ruff gate (prior blocker, now resolved):** `ruff check .` — `All checks passed!` (0 errors;
  previously 59: 29 E702, 11 E501, 7 UP045, 5 B905, 4 E741, 2 UP017, 1 I001). The 10 truly
  safe fixes (`I001`/`UP045`/`UP017`) plus 3 induced unused-`Optional` imports were applied via
  `ruff check --fix .`; the 49 remaining (`E702`/`E501`/`E741`/`B905`) were fixed by hand.
- ✓ Phase 1 — `pip install -e .[dev]` exits 0; schema encodes `pattern`+`enum`; round-trips and
  rejects bad date/currency; import smoke (`STUDENT_MODEL_ID`, `DEEPSEEK_BASE_URL`, 16-entry
  registry); `README.md` resolves.
- ✓ Phase 2 — `generate --n-docs 20` writes exactly 20 lines; all `clean_json` validate against
  `Invoice`; re-running `--seed 42` is byte-identical (determinism).
- ✓ Phase 3 — `pytest tests/test_corrupt_invariant.py` green; `corrupt` CLI writes 20 lines; the
  label-preserving invariant holds on real output (`is_label_preserving` true for all 20 records);
  canonicalizers invert (`25/06/2026`→`2026-06-25`, `USD 1,234.56`→`1234.56`).
- ✓ Phase 4 — offline teacher checks pass: client wiring + JSON-mode prompt content, validator
  builds against the locked schema, semantic-vs-transport exception split (`SemanticExtractionError`
  vs `RetryableTransportError`), and `_load_seen_ids` resumability across two JSONL files. No live
  API call.
- ✓ Phase 5 — `pytest tests/test_prepare_split.py` green; CLI produces `n_filtered==2`,
  `n_train+n_test==10`; determinism and strict-JSON completions verified by the tests.
- ✓ Phase 6 — offline training imports resolve (`torch`, `SFTConfig`/`SFTTrainer`, PEFT,
  `AutoModelForImageTextToText`, `BitsAndBytesConfig`, `set_seed`); config wiring; recipe source
  assertions for `set_seed`, `data_seed`, `use_cache = False`, `use_reentrant`,
  `get_peft_model`, `completion_only_loss=True`. No live model download.
- ✓ Phase 7 — pure-logic eval verified: perfect-match → `parse_rate`/`record_exact`/`micro_f1`
  all 1.0; format-only amount drift (`$100.00` vs `100.00`) still scores 1.0 (canonicalization);
  hard gate `['not json', good]` → `parse_rate==0.5`, `micro_f1<1.0` with line-item gold leaves
  counted as FN. `scipy` resolves.
- ✓ Phase 8 — `from doc_extract import run_all` imports; `make -n all` dry-runs the target chain;
  README H1 `# doc-extract` present; README contains only the placeholder `DEEPSEEK_API_KEY=sk-...`
  setup instruction (no real secret).
- ✓ No regressions: `pytest -q` — 12 passed; all 10 `doc_extract` modules import cleanly; schema
  constraints (`pattern`/`enum`, 16 leaves) intact. The Ruff edits were semantics-preserving
  (split semicolons onto newlines, `zip(..., strict=False)` on equal-length iterables, `l`→`line`
  renames, line wrapping).

### Code Review Findings

#### Matches Plan:

- `src/doc_extract/schema.py:25`, `src/doc_extract/schema.py:37`, `src/doc_extract/schema.py:59` —
  Pydantic `LineItem`/`Invoice`, `INVOICE_JSON_SCHEMA`, and the 16-entry `FIELD_TYPE_REGISTRY`
  remain the single source of truth.
- `src/doc_extract/generate.py:44`, `src/doc_extract/generate.py:175`, `src/doc_extract/generate.py:193`
  — seeded Faker generation, four template families, validated clean JSON/text.
- `src/doc_extract/canon.py:21`, `src/doc_extract/canon.py:77`, `src/doc_extract/corrupt.py:69`,
  `src/doc_extract/corrupt.py:145` — shared canonicalizers and the five label-preserving transforms.
- `src/doc_extract/teacher_labeler.py:40`, `src/doc_extract/teacher_labeler.py:127`,
  `src/doc_extract/teacher_labeler.py:154`, `src/doc_extract/teacher_labeler.py:155`,
  `src/doc_extract/teacher_labeler.py:208` — DeepSeek/OpenAI client wiring, schema validation, JSON
  mode, thinking disabled, resumable batch labeling.
- `src/doc_extract/prepare.py:38`, `src/doc_extract/prepare.py:51`, `src/doc_extract/prepare.py:62`,
  `src/doc_extract/prepare.py:93` — invalid filtering, deterministic split, strict-JSON
  prompt/completion records, HF snapshots.
- `src/doc_extract/train.py:28`, `src/doc_extract/train.py:48`, `src/doc_extract/train.py:56`,
  `src/doc_extract/train.py:63`, `src/doc_extract/train.py:79` — 4-bit NF4, deterministic seed,
  LoRA `all-linear`, `SFTConfig`, `completion_only_loss=True`.
- `src/doc_extract/evaluate.py:23`, `src/doc_extract/evaluate.py:85`, `src/doc_extract/evaluate.py:170`,
  `src/doc_extract/evaluate.py:282`, `src/doc_extract/evaluate.py:367` — parse→schema→canonicalized-leaf
  eval, Hungarian line-item matching, corpus bootstrap CI, `learning_proven`.
- `Makefile:1`, `Makefile:20`, `Makefile:32`, `src/doc_extract/run_all.py:9`, `docs/REFLECTION.md:29`
  — orchestration targets, DeepSeek key check, baseline target, programmatic runner, reflection checklist.

#### Deviations from Plan:

- `pyproject.toml:13` / `requirements.txt:1-15` — dependency ranges are newer than the plan's
  Phase-1 pins (`transformers>=5`, `trl>=1`, `datasets>=5`, `openai>=2`, `faker>=29` vs the plan's
  older ranges). **Accepted by the developer** — the plan's pinned versions are outdated and the
  newer ranges are confirmed working offline (imports, `SFTConfig`/`SFTTrainer`/`OpenAI()`
  compatibility). No action required; optionally `/skill:revise` the plan's pins to match reality
  (cosmetic, non-blocking).

#### Pattern Conformance:

- ✓ All `src/doc_extract/*.py` modules follow the `python -m doc_extract.<stage>` CLI pattern with
  shared `config`/`schema` imports; tests target the risk-bearing invariants (corruption label
  preservation, deterministic split). No changes to module boundaries were made this session.
- Acceptable variation: the Ruff reformatting expanded several semicolon one-liners in
  `evaluate.py` to multi-line statements; logic is identical (`bump()` helper bodies, counter
  increments, `zip(..., strict=False)` where iterables are equal-length by construction).

#### Potential Issues:

- `src/doc_extract/config.py:18` leaves `STUDENT_REVISION = "main"`; no runtime manifest yet
  captures the resolved Hugging Face commit SHA during train/eval. This is optional reproducibility
  hardening (Phase D in the session handoff), **not a blocker** for the offline pass — flagging for
  the live-training follow-up.

### Manual Testing Required:

1. DeepSeek teacher labeling:
   - [ ] With `DEEPSEEK_API_KEY` set, run `python -m doc_extract.teacher_labeler --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl`.
   - [ ] Inspect several `labeled.jsonl` outputs against `Invoice` and the generated gold; confirm quarantine rate is small and transport failures remain retryable.
2. GPU training:
   - [ ] After `data/sft/train.jsonl` exists, run `python -m doc_extract.train` on the RTX 3090; confirm no OOM, adapter output, decreasing loss.
   - [ ] Verify `--skip-merge` skips merge and the default writes `artifacts/checkpoints/merged`.
3. Live evaluation:
   - [ ] Run `python -m doc_extract.evaluate` after training; confirm `artifacts/metrics.json` includes base/finetuned metrics and `learning_proven` reflects the bootstrap CI.
4. End-to-end orchestration:
   - [ ] Run `make all` and `make baseline` with API key + GPU available.
   - [ ] Fill in `docs/REFLECTION.md` with actual per-stage observations after the live run.

### Recommendations:

- Ready to commit — all offline automated gates pass and the Ruff blocker is resolved. Use `/skill:commit` to group the changes into atomic commits (the repo currently has none).
- Then run the live manual gates above (DeepSeek labeling → GPU training → live eval → `make all`) to complete Phase C verification.
