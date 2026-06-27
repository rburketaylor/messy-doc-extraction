---
template_version: 1
date: 2026-06-26T15:31:17-0300
author: Burke T
commit: no-commit
branch: main
repository: messy-doc-extraction
topic: "Validation of Dirty-to-clean synthetic data fine-tuning pipeline"
status: ready
verdict: fail
parent: ".rpiv/artifacts/plans/2026-06-25_21-35-07_dirty-data-extraction-finetune.md"
tags: [validation, plan, greenfield, fine-tuning, qlora, lfm2.5-vl, deepseek, structured-extraction, synthetic-data, evaluation, reproducibility]
last_updated: 2026-06-26T15:31:17-0300
---

## Validation Report: Dirty-to-clean synthetic data fine-tuning pipeline

### Implementation Status

- ⚠️ Phase 1: Project scaffold + schema/config foundation — Partially implemented (dependency ranges diverge from the approved plan; see Findings)
- ✓ Phase 2: Clean synthetic invoice generator — Fully implemented
- ✓ Phase 3: Canonicalization + label-preserving corruption — Fully implemented
- ✓ Phase 4: Teacher labeling (DeepSeek) — Fully implemented for offline/verifiable behavior; live API smoke remains manual
- ✓ Phase 5: Dataset preparation — Fully implemented
- ✓ Phase 6: Training (QLoRA SFT) — Recipe implemented and offline checks pass; live GPU training remains manual
- ✓ Phase 7: Evaluation harness — Pure-logic harness implemented and checked; live base-vs-finetuned eval remains manual
- ✓ Phase 8: Orchestration + reproducibility + reflection — Fully implemented; end-to-end live run remains manual

Git evidence: the repository has no commits yet (`git log` empty) and the implementation files are untracked, so commit-to-commit diff evidence was unavailable. Validation is based on working-tree file inspection plus the plan's automated checks.

### Automated Verification Results

- ✓ Package install: `pip install -e .[dev]` — passed in a temporary Python 3.11 venv. The same command is blocked in the ambient system Python 3.14 by PEP 668, which is an environment constraint rather than a package build failure.
- ✓ Project tests: `pytest -q` — 12 passed.
- ✓ Phase 1 schema/import checks: schema constraints, round-trip, rejection of bad date/currency, import smoke, and `test -f README.md` all passed.
- ✓ Phase 2 generator checks: generated 20 docs, validated all `clean_json`, verified totals/null consistency, currency-code presence, all 4 templates, and deterministic byte-identical output.
- ✓ Phase 3 corruption checks: `pytest tests/test_corrupt_invariant.py -q`, generate→corrupt CLI, real-output invariant, normalizer inversion, and gold preservation all passed.
- ✓ Phase 4 teacher-labeler offline checks: client wiring, JSON-mode prompt content, validator behavior, semantic-vs-transport exception split, and `_load_seen_ids` resumability all passed.
- ✓ Phase 5 prepare checks: `pytest tests/test_prepare_split.py -q`, invalid filtering, deterministic split, strict-JSON completions, and manifest contents all passed.
- ✓ Phase 6 training recipe checks: offline imports, config wiring, and source assertions for `set_seed`, `data_seed`, `use_cache = False`, `use_reentrant`, `get_peft_model`, and `completion_only_loss=True` all passed.
- ✓ Phase 7 evaluation checks: perfect-match metrics, format-only amount normalization, hard parse gate, and `scipy` import all passed.
- ✓ Phase 8 orchestration checks: `python -c "from doc_extract import run_all; print('ok')"`, `make -n all`, README H1, and no real API key in README all passed.
- ✗ Additional quality check: `ruff check .` — failed with 59 errors under the repository's own Ruff config (examples include `src/doc_extract/evaluate.py:70` E702, `src/doc_extract/evaluate.py:87` B905, `src/doc_extract/corrupt.py:127` E741, and `src/doc_extract/teacher_labeler.py:123` E501).

### Code Review Findings

#### Matches Plan:

- `src/doc_extract/schema.py:25`, `src/doc_extract/schema.py:37`, `src/doc_extract/schema.py:55`, `src/doc_extract/schema.py:59` — Pydantic `LineItem`/`Invoice`, `INVOICE_JSON_SCHEMA`, and 16-entry `FIELD_TYPE_REGISTRY` are present as the single source of truth.
- `src/doc_extract/generate.py:44`, `src/doc_extract/generate.py:175`, `src/doc_extract/generate.py:193` — seeded Faker generation, four template families, and validated clean JSON/text records match Phase 2.
- `src/doc_extract/canon.py:21`, `src/doc_extract/canon.py:77`, `src/doc_extract/canon.py:94`, `src/doc_extract/corrupt.py:69`, `src/doc_extract/corrupt.py:145` — shared canonicalizers and the five label-preserving corruption transforms are implemented.
- `src/doc_extract/teacher_labeler.py:40`, `src/doc_extract/teacher_labeler.py:127`, `src/doc_extract/teacher_labeler.py:154`, `src/doc_extract/teacher_labeler.py:155`, `src/doc_extract/teacher_labeler.py:208` — DeepSeek/OpenAI client wiring, JSON schema validation, JSON mode, thinking disabled, and resumable batch labeling match Phase 4.
- `src/doc_extract/prepare.py:38`, `src/doc_extract/prepare.py:51`, `src/doc_extract/prepare.py:62`, `src/doc_extract/prepare.py:93` — invalid filtering, deterministic split, strict JSON prompt/completion records, and HF snapshots match Phase 5.
- `src/doc_extract/train.py:28`, `src/doc_extract/train.py:48`, `src/doc_extract/train.py:56`, `src/doc_extract/train.py:63`, `src/doc_extract/train.py:79` — 4-bit NF4, deterministic seed, LoRA `all-linear`, `SFTConfig`, and `completion_only_loss=True` match Phase 6.
- `src/doc_extract/evaluate.py:23`, `src/doc_extract/evaluate.py:85`, `src/doc_extract/evaluate.py:170`, `src/doc_extract/evaluate.py:282`, `src/doc_extract/evaluate.py:367` — parse→schema→canonicalized-leaf eval, Hungarian line-item matching, corpus bootstrap CI, and `learning_proven` match Phase 7.
- `Makefile:1`, `Makefile:20`, `Makefile:32`, `src/doc_extract/run_all.py:9`, `docs/REFLECTION.md:29` — orchestration targets, DeepSeek key check, baseline target, programmatic runner, and reflection checklist are present.

#### Deviations from Plan:

- `pyproject.toml:13`, `pyproject.toml:14`, `pyproject.toml:18`, `pyproject.toml:21`, `pyproject.toml:25` and `requirements.txt:1-15` — dependency ranges differ materially from the approved Phase 1 plan. The plan specified `transformers>=4.46,<5`, `trl>=0.15,<0.17`, `datasets>=3,<4`, `openai>=1.40,<2`, and `faker>=28,<30`; the implementation uses newer major ranges (`transformers>=5,<6`, `trl>=1,<2`, `datasets>=5,<6`, `openai>=2,<3`, `faker>=29,<41`). Offline imports pass, but this is still a plan deviation that should either be reverted or explicitly accepted through `/skill:revise` after live training/label/eval compatibility is proven.

#### Pattern Conformance:

- ✓ `src/doc_extract/*.py` modules follow the planned `python -m doc_extract.<stage>` CLI pattern, with shared `config` and `schema` imports.
- ✓ Tests focus on the plan's risk-bearing invariants: corruption label preservation and deterministic prepare split.
- Minor observation: `src/doc_extract/prepare.py:69` adds `_to_hf_dataset()` so empty splits still snapshot with explicit columns. This is acceptable hardening, not a deviation.
- Minor observation: `Makefile:21` omits `--seed` when invoking `teacher_labeler`, but `teacher_labeler` defaults to `config.SEED` and seed is a no-op at `temperature=0`; acceptable variation.

#### Potential Issues:

- `ruff check .` currently fails with 59 style/lint errors despite the project declaring Ruff config in `pyproject.toml`. This is not one of the plan's listed automated commands, but it is a repository-defined quality gate and should be fixed before considering the implementation merge-ready.
- `src/doc_extract/config.py:18` leaves `STUDENT_REVISION = "main"`; no runtime manifest currently captures the resolved Hugging Face commit SHA during train/eval. This weakens the plan's reproducibility goal for model artifacts.

### Manual Testing Required:

1. DeepSeek teacher labeling:
   - [ ] With `DEEPSEEK_API_KEY` set, run `python -m doc_extract.teacher_labeler --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl`.
   - [ ] Inspect several `labeled.jsonl` outputs against `Invoice` and the generated gold values; confirm quarantine rate is small and transport failures remain retryable.
2. GPU training:
   - [ ] After `data/sft/train.jsonl` exists, run `python -m doc_extract.train` on the RTX 3090 and confirm no OOM, adapter output, optional merged output, and decreasing loss.
   - [ ] Verify `--skip-merge` skips merge and the default merge writes `artifacts/checkpoints/merged`.
3. Live evaluation:
   - [ ] Run `python -m doc_extract.evaluate` after training; confirm `artifacts/metrics.json` includes base/finetuned metrics and `learning_proven` reflects the bootstrap CI.
4. End-to-end orchestration:
   - [ ] Run `make all` and `make baseline` with API key and GPU available.
   - [ ] Fill in `docs/REFLECTION.md` with actual per-stage observations after the live run.

### Recommendations:

- Reconcile the dependency-version deviation: either restore the plan's dependency ranges or revise the plan to document and approve the newer major-version stack.
- Fix the Ruff violations, or remove/relax the Ruff gate if lint-clean code is intentionally out of scope.
- Capture the resolved student model commit SHA in a training/evaluation manifest before relying on results for reproducibility.
- Re-run `/skill:validate` after the above fixes and the live manual checks are complete.
