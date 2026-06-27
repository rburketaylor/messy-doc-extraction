# doc-extract

Dirty-to-clean synthetic-data SFT/QLoRA extraction pipeline — a hands-on learning project.

> The process is the product: mastery of the end-to-end fine-tuning loop is the goal; a working
> model is a bonus.

This project practices the full fine-tuning loop: generate realistic *dirty* synthetic invoices,
label them to clean JSON with a strong teacher (DeepSeek V4 Flash), QLoRA-fine-tune a small open
vision-language model (Liquid LFM2.5-VL-1.6B-Extract), and evaluate field-level extraction vs gold.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
export DEEPSEEK_API_KEY=sk-...   # teacher; never commit it
```

Requires a CUDA GPU with >=16GB VRAM (developed on a 24GB RTX 3090).

## Run

```bash
make data      # generate -> corrupt -> teacher-label (needs DEEPSEEK_API_KEY)
make prepare   # deterministic train/test split -> data/sft
make train     # QLoRA SFT -> artifacts/checkpoints/{adapter,merged}
make evaluate  # base vs fine-tuned -> artifacts/metrics.json
make all       # the whole loop
make baseline  # evaluate the base model only (no training)
```

Or programmatically: `python -m doc_extract.run_all`.

## Architecture

Four stages plus an evaluation harness, sharing one Pydantic invoice schema as the single source
of truth (teacher validation, eval canonicalization, and synthetic generation all read it):

1. **generate** — Faker self-consistent invoices rendered to 4 markdown template families.
2. **corrupt** — label-preserving corruption (date/amount reformat, boilerplate, reorder); verified
   by an invariant test that value tokens survive.
3. **teacher_labeler** — DeepSeek V4 Flash extracts clean JSON with retry/repair/quarantine.
4. **prepare** — re-validate, deterministic seeded split, strict-JSON `{prompt, completion}`.
5. **train** — 4-bit NF4 QLoRA (TRL SFTTrainer + PEFT) of LFM2.5-VL-1.6B-Extract.
6. **evaluate** — 3-layer gate (parse -> schema -> canonicalized-leaf micro-F1), Hungarian line
   items, paired bootstrap CI proving the fine-tuned model beats the base.

See `docs/REFLECTION.md` for the per-stage learning reflection (the real deliverable).
