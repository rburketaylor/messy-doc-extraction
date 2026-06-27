---
date: 2026-06-26T17:45:48-0300
author: Burke T
commit: no-commit
branch: main
repository: messy-doc-extraction
topic: "Ruff fixes + verification/testing next steps"
tags: [validation, linting, ruff, fine-tuning, qlora, testing, handoff]
status: complete
last_updated: 2026-06-26T17:45:48-0300
last_updated_by: Burke T
type: implementation_strategy
---

# Handoff: Fix Ruff issues + verification/testing next steps

## Task(s)

This session ran `/skill:validate` against the 8-phase fine-tuning pipeline plan. Outcome:

- **Validation (COMPLETE):** wrote `.rpiv/artifacts/validation/2026-06-26_15-31-17_dirty-to-clean-synthetic-data-fine-tuning-pipeline.md`. **Verdict: `fail`.**
- **Fix Ruff violations (PLANNED — not started):** `ruff check .` fails with **59 errors** under the repo's own Ruff config. This is the only real blocker now (see Other Notes for the dependency-version red herring).
- **Verification & testing next steps (PLANNED — not started):** once Ruff is clean, re-run validate to flip the offline verdict to `pass`, then perform the live/manual verification gates (DeepSeek labeling, GPU training, live eval, end-to-end).

The pipeline itself is fully implemented across all 8 phases (greenfield `src/doc_extract/` package). Only the lint failure plus un-run live checks remain.

## Critical References

- `.rpiv/artifacts/plans/2026-06-25_21-35-07_dirty-data-extraction-finetune.md` — the 8-phase implementation plan (source of all success criteria).
- `.rpiv/artifacts/validation/2026-06-26_15-31-17_dirty-to-clean-synthetic-data-fine-tuning-pipeline.md` — the validation report this session produced (contains the full findings, the Manual Testing Required checklist, and Recommendations).
- `pyproject.toml:35-43` — the active Ruff config that defines the gate (`select = ["E", "F", "I", "UP", "B"]`, `line-length = 100`, `target-version = "py311"`).

## Recent changes

This session made **no code changes** — validate is read-only. The only file written is the validation artifact. The pipeline code was implemented in a prior session and is currently **untracked** (the repo has no commits yet).

## Learnings

### Ruff failure breakdown (59 errors, run in a Python 3.11 venv)
Counts by rule, with first-occurrence locations:

- `E702` = 29 — multiple statements on one line (semicolon). Concentrated in `src/doc_extract/evaluate.py` (e.g. `evaluate.py:70:42`, `evaluate.py:164`, `evaluate.py:184`). Pattern is the `bump(...)` helper calls and `parsed += 1; ok = ...` one-liners. Fix: split onto separate lines.
- `E501` = 11 — line too long (>100). e.g. `teacher_labeler.py:123:101` (104 chars), `schema.py:28:101`, `evaluate.py:164:101`. Fix: wrap or shorten.
- `UP045` = 7 — use `X | None` instead of `Optional[X]`. e.g. `schema.py:32:11`. **Auto-fixable** with `ruff check --fix .`. Files use `from __future__ import annotations`, so the rewrites are safe.
- `B905` = 5 — `zip()` without explicit `strict=`. In `evaluate.py` (e.g. `evaluate.py:87:21`, `evaluate.py:224`, and inside `_score_line_items` / `compute_metrics` loops). Fix: add `strict=False` (the two iterables are always equal-length predictions/golds by construction) or restructure.
- `E741` = 4 — ambiguous loop variable name `l`. In `src/doc_extract/corrupt.py:127:25` and the eval/scan loops that iterate JSONL lines. Fix: rename `l` → `line` (consistent with the rest of the codebase, which already uses `line` in many places).
- `UP017` = 2 — use `datetime.UTC` alias. e.g. `teacher_labeler.py:203:36`. **Auto-fixable.**
- `I001` = 1 — import block un-sorted in `run_all.py:3:1`. **Auto-fixable.**

**10 of 59 are auto-fixable** (`I001` + `UP045` + `UP017`); the rest (`E702`, `E501`, `E741`, `B905` = 49) need manual edits.

### Environment quirks (important)
- System Python is **3.14.6** and is **externally-managed (PEP 668)** — `pip install -e .[dev]` is **blocked** there. This is an environment constraint, not a build failure.
- `python3.12` shim is broken (pyenv "command not found"). A working **Python 3.11** interpreter exists at `/home/burket/.local/bin/python3.11`.
- This session created a temp venv at **`/tmp/doc_extract_validate_venv`** (Python 3.11) where `pip install -e .[dev]`, `pytest -q` (12 passed), `ruff`, and all plan automated checks run. **This venv may not survive a new session** — recreate it if `/tmp` is cleared: `/home/burket/.local/bin/python3.11 -m venv /tmp/doc_extract_validate_venv && . /tmp/doc_extract_validate_venv/bin/activate && pip install -e '.[dev]'`. Or use the project's own `make venv`.
- The repo has **no git commits** (`git log` empty); all implementation files are untracked. Commit-to-commit diff evidence was unavailable; validate relied on file inspection + the plan's commands.

### What already PASSES (do not break these)
All plan automated verification passed in the venv: install, `pytest -q` (12 tests), schema/generator/corrupt/teacher-offline/prepare/train-recipe/eval-logic/run_all/Makefile-dry-run. The Ruff fixes must keep `pytest` green.

## Artifacts

- `.rpiv/artifacts/validation/2026-06-26_15-31-17_dirty-to-clean-synthetic-data-fine-tuning-pipeline.md` — validation report (read first; verdict `fail`).
- `.rpiv/artifacts/plans/2026-06-25_21-35-07_dirty-data-extraction-finetune.md` — 8-phase plan with per-phase Automated/Manual verification criteria.
- `.rpiv/artifacts/research/2026-06-25_20-26-01_dirty-data-extraction-finetune.md` — parent research (model ids, TRL+PEFT recipe, DeepSeek recipe, eval contract).
- `/tmp/doc_extract_ruff.log` — full Ruff output from this session (re-generate with `ruff check .` if /tmp is gone).

## Action Items & Next Steps

### Phase A — Fix Ruff (primary blocker)
1. Recreate the venv if needed (see Learnings), activate it.
2. Run `ruff check .` to reproduce the 59 errors.
3. Run `ruff check --fix .` to auto-fix the 10 (`I001`, `UP045`, `UP017`).
4. Manually fix the remaining 49:
   - `E702` (29): split semicolon statements in `evaluate.py`.
   - `E501` (11): wrap long lines across all modules.
   - `E741` (4): rename `l` → `line` (esp. `corrupt.py:127`).
   - `B905` (5): add `strict=False` to `zip(...)` in `evaluate.py`.
5. Confirm `ruff check .` → `All checks passed!`
6. Confirm `pytest -q` still → 12 passed (no regressions).

### Phase B — Re-validate
7. Re-run `/skill:validate` against the plan. With Ruff clean and dependency versions **intentionally ignored** (per user — they're outdated/wrong), the offline verdict should flip to **`pass`**.

### Phase C — Verification & testing next steps (from the validation report's Manual Testing Required)
8. **DeepSeek teacher labeling** — set `DEEPSEEK_API_KEY`, run `python -m doc_extract.teacher_labeler --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl`; verify quarantine rate is small, transport failures stay retryable, outputs validate against `Invoice` and recover gold.
9. **GPU training** — ensure `data/sft/train.jsonl` exists, run `python -m doc_extract.train` on the RTX 3090; verify no OOM, adapter written, loss decreases; verify `--skip-merge` and that merge writes `artifacts/checkpoints/merged`.
10. **Live evaluation** — run `python -m doc_extract.evaluate`; verify `artifacts/metrics.json` has base/finetuned metrics and `learning_proven` reflects the bootstrap CI.
11. **End-to-end** — `make all` and `make baseline` (needs API key + GPU).
12. **Reflection** — fill in `docs/REFLECTION.md` with real per-stage observations after the live run (it's currently a scaffold with an acceptance-criteria checklist at `docs/REFLECTION.md:29`).

### Phase D — Optional hardening (from validation Potential Issues)
13. Capture the resolved Hugging Face student-model commit SHA into a training/eval manifest (config currently leaves `STUDENT_REVISION = "main"` at `src/doc_extract/config.py:18`) to strengthen reproducibility.
14. `/skill:commit` — group the validated changes into atomic commits (only after verdict is `pass`).

## Other Notes

- **Dependency versions are NOT a real blocker.** The validation report listed `transformers>=5`, `trl>=1`, `datasets>=5`, `openai>=2`, `faker>=29` as a deviation from the plan's older ranges. The user confirmed the plan's pinned versions are **outdated and wrong** — ignore this finding. Offline imports and `SFTConfig`/`SFTTrainer`/`OpenAI()` compatibility are confirmed working under the newer ranges. (If anything, consider `/skill:revise` to update the plan's pins to match reality, but that's cosmetic — do not block on it.)
- The pipeline's source layout: `src/doc_extract/` with `{__init__, schema, config, generate, canon, corrupt, teacher_labeler, prepare, train, evaluate, run_all}.py`; tests in `tests/{test_corrupt_invariant,test_prepare_split}.py`; glue in `Makefile`; docs in `README.md` + `docs/REFLECTION.md`.
- Tests only cover corruption invariant + prepare split; `teacher_labeler`, `train`, and `evaluate` have no pytest files (relies on plan one-liner checks). Adding unit tests for the eval Hungarian-matching / nullable-leaf logic would be valuable but is out of the immediate Ruff-fix scope.
