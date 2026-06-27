# Reflection — "the process is the product"

A per-stage learning journal. Fill each section after running that stage; this is the real
deliverable of the project.

## Generate
What I learned about building realistic, self-consistent synthetic data and why determinism matters.

## Corrupt
What I learned about *label-preserving* augmentation: why corruption is only valid if a
deterministic canonicalizer recovers the identical gold (the invariant test), and the difference
between surface noise and value-changing (label-breaking) edits.

## Teacher-label
What I learned about using an API teacher as a label factory: JSON mode, retry/repair/quarantine,
and why invalid labels must never reach training.

## Prepare
What I learned about deterministic splitting and SFT prompt/completion formatting.

## Train
What I learned about QLoRA: 4-bit quantization, LoRA adapters, gradient checkpointing, VRAM, and
the completion-only loss. Observations: loss curve, time/epoch, VRAM used.

## Evaluate
What I learned about objective, field-level evaluation: the 3-layer gate, canonicalization,
Hungarian line-item matching, and using a paired bootstrap CI to *prove* the model learned.

## Acceptance criteria checklist
- [ ] generate produced 500+ dirty docs
- [ ] each doc has a paired teacher JSON; invalids filtered + counted
- [ ] deterministic split reproducible from a seed
- [ ] training completes without OOM, writes a checkpoint
- [ ] eval prints JSON-validity, per-field exact-match, F1
- [ ] fine-tuned F1/exact-match exceeds the base model's
- [ ] this reflection captures what was learned at each stage
