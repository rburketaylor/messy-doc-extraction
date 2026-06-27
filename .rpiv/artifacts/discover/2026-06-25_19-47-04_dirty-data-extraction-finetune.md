---
date: 2026-06-25T19:47:04-0300
author: Burke T
commit: no-commit
branch: no-branch
repository: unknown
topic: "Dirty-to-clean synthetic data fine-tuning pipeline"
tags: [intent, frd, fine-tuning, synthetic-data, structured-extraction, qlora, teacher-distillation]
status: ready
last_updated: 2026-06-25T19:47:04-0300
last_updated_by: Burke T
---

# FRD: Dirty-to-clean synthetic data fine-tuning pipeline

## Summary
A hands-on learning project to master end-to-end fine-tuning. Generate realistic *dirty* synthetic documents that mimic messy multi-vendor real-world data, use a strong API teacher model (DeepSeek V4 flash/pro or GLM 5.2) to produce clean structured-extraction targets, pair them as `(dirty input → clean JSON)` training examples, and fine-tune a small open model — **Liquid lfm2.5-vl-1.6b**, a vision-language model chosen deliberately so image/PDF inputs can be added later without a model swap — via QLoRA on a local GPU. Success is measured by field-level exact-match/F1 against the teacher's gold targets on a held-out set — and by the mastery gained, since the process is the product.

## Problem & Intent
The developer's words:

> "I want to learn fine tuning."

> "in the field the data will be from many different vendors and not clean I want to be able to learn the process of taking that dirty data and cleaning it."

> "I'm not 100% sure how it works."

The primary intent is **mastery of the end-to-end fine-tuning pipeline** — the process is the product; a working model is a bonus. The developer's real-world motivation is that their field's data arrives dirty and inconsistent from many different vendors, and they want to learn to take that dirty data, clean/normalize it, and fine-tune on it. Synthetic generation is the scaffold to practice on *realistic* dirtiness before (or instead of) touching real data.

Three conceptual knots surfaced and were resolved during the interview (captured so downstream skills inherit the corrected mental model):
1. **Cleaning happens *before* training, always.** You never feed dirty data into fine-tuning hoping the model learns to be clean — that just teaches it the mess.
2. **The "ideal" is the teacher's job.** The teacher takes a dirty doc and produces the clean, correct extraction; the student learns to reproduce it. This *unifies* the cleaning pipeline and the fine-tuning task into one thing.
3. **"Fine-tune with a teacher" here is SFT with teacher-generated labels** (not logit distillation, not RLHF). The two-stage curriculum the developer originally wondered about is real but unnecessary for a first project — both stages would use cleaned data anyway, so a single clean→SFT stage suffices.

## Goals
- Master the end-to-end fine-tuning loop hands-on: dirty-data generation → cleaning/curation → teacher labeling → SFT training → objective evaluation.
- Learn to take dirty, inconsistent, multi-vendor-style data and clean/normalize it into fine-tuning-ready `(input → JSON)` pairs.
- Learn to evaluate a fine-tuned model *objectively* (field-level metrics vs gold) rather than eyeballing.
- Produce a working (if modest) fine-tuned extraction model as concrete proof of learning.

## Non-Goals
- **PDF parsing / OCR** — deferred to a later stage; this round trains and evaluates on text/markdown/JSON synthetic docs only.
- **Preference/DPO, self-correction, or contrastive training** — out of scope; single-stage SFT only this round (others are natural follow-ups).
- **Deploying/serving the model** (REST API, UI, inference server) — *inferred* out of scope; the deliverable is a checkpoint + evaluation report, not a service.
- **Production MLOps** (orchestration, data versioning, experiment-tracking servers, CI) — *inferred* out of scope; lightweight local scripts + notebooks for learning.

## Functional Requirements
1. The pipeline SHALL generate 500–2,000 synthetic "dirty" documents as text/markdown/JSON, embedding realistic defect types modeled on multi-vendor messiness (format variance, missing/extra fields, inconsistent units/encodings, typos, conflicting values).
2. The pipeline SHALL use a strong API teacher model (DeepSeek V4 flash/pro or GLM 5.2) to produce a clean structured-extraction target (JSON) for each dirty document.
3. The pipeline SHALL pair each `(dirty input → clean JSON output)` into an instruction-response record, filter out invalid/empty teacher outputs, and split deterministically into train / held-out test sets.
4. The system SHALL fine-tune the Liquid **lfm2.5-vl-1.6b** model (vision-language, ~1.6B) via QLoRA/LoRA on the local GPU.
5. The system SHALL evaluate the fine-tuned model on the held-out test set with field-level metrics — JSON-validity rate, per-field exact-match/accuracy, overall F1 — against the teacher's gold targets.
6. The pipeline SHALL be reproducible end-to-end from local scripts/notebooks (generate → train → evaluate).

## Non-Functional Requirements
- **Performance**: No latency SLO. Training should iterate in minutes-per-epoch (small model + QLoRA). API teacher throughput governs data-generation wall-clock.
- **Security**: Teacher API keys handled via env vars / secrets, never committed. Synthetic data only — no real/vendor PII in the repo.
- **UX / Accessibility**: n/a (developer-facing scripts/notebooks). Prioritize clear logging, determinism, and reproducibility over polish.
- **Reliability**: Generation must be resumable/idempotent with API retry/backoff; malformed teacher outputs must be quarantined/filtered before training so bad labels don't poison the run.

## Constraints & Assumptions
- Local GPU with **8–16GB VRAM** → base model capped at ~3B params under QLoRA.
- API access + budget for a frontier teacher (GPT-4o / Claude / Gemini); estimated ~$5–30 data-gen cost at 500–2,000 docs.
- **Greenfield** — no existing codebase, tooling, or git repo; everything built from scratch.
- *Assumption*: synthetic dirtiness can be made realistic enough that the cleaning/extraction skills transfer to real multi-vendor data — needs the defect types modeled on real-domain patterns (flagged for research).
- *Assumption*: lfm2.5-vl-1.6b is sufficient to demonstrate non-trivial structured extraction and supports QLoRA fine-tuning on 8–16GB VRAM (verify availability + QLoRA-compat during research).
- *Assumption*: the document domain/schema is not yet specified — extraction needs a schema, so a concrete domain (e.g., invoices, contracts, receipts) must be chosen before generation (Open Question).

## Acceptance Criteria
- [ ] Running the generation script produces 500–2,000 synthetic dirty docs on disk.
- [ ] Each doc has a paired teacher-generated clean JSON target; invalid/empty targets are filtered out and counted.
- [ ] A deterministic train/test split exists and is reproducible from a seed.
- [ ] The training command completes QLoRA fine-tuning on the local GPU without OOM and writes a checkpoint.
- [ ] The evaluation command runs the checkpoint on the held-out set and prints JSON-validity rate, per-field exact-match, and overall F1 vs gold.
- [ ] The fine-tuned model's F1/exact-match **exceeds the un-finetuned base model's** on the same held-out set (concrete proof of learning).
- [ ] A short written reflection captures what was learned at each stage — the "process is the product" deliverable.

## Recommended Approach
A four-stage local pipeline of Python scripts/notebooks: (1) generate realistic dirty text docs with a cheap/fast model; (2) call a strong API teacher (DeepSeek V4 flash/pro or GLM 5.2) to produce clean JSON extraction targets, filtering invalids with retry/backoff; (3) SFT/QLoRA **Liquid lfm2.5-vl-1.6b** (vision-language, ~1.6B) with Hugging Face PEFT/TRL (or Unsloth); (4) evaluate field-level exact-match/F1 vs gold on a held-out set. No PDF, no deployment, no orchestration — the focus is the learnable core loop.

## Decisions

### Primary goal: learn the craft
**Question**: What does success look like at the end of this effort?
**Recommended**: n/a — intent question (no recommendation offered).
**Chosen**: "Learn the craft" — mastery of the end-to-end pipeline is the real win; a working model is a bonus.
**Rationale**: Developer-stated intent; the process is the product. This framing de-prioritizes deployment/polish and prioritizes pedagogical clarity at every downstream fork.

### Compute: local GPU (real fine-tuning)
**Question**: What hardware/access do you have to work with? (Decides whether you touch real weights or just call an API.)
**Recommended**: Local GPU (best fit for "learn the craft" — full control, real LoRA/QLoRA mechanics, no per-run cost).
**Chosen**: Local GPU.
**Rationale**: API-only fine-tuning teaches the least about how fine-tuning actually works; local weights are essential to the stated learning goal. Confirmed GPU has 8–16GB VRAM.

### Role of the data: cleaned SFT (flaws as realistic messiness to practice cleaning)
**Question**: What role do the "issues and inconsistencies" play? (SFT vs DPO vs self-correction.)
**Recommended**: Cleaned SFT (flaws as waste to curate) — exposes that deliberate-flaw-then-clean is redundant unless the flaws carry signal.
**Chosen**: Cleaned SFT, *re-framed* — the developer clarified the real motivation: their field data is inherently dirty, and the synthetic flaws are a scaffold to *practice cleaning realistic messiness*. The teacher produces the clean targets; the student learns to reproduce the cleanup.
**Rationale**: Developer correction of the original framing. Resolves the core "issue in the thinking": the flaws aren't contrastive signal, they're realistic dirtiness to clean — and the teacher's clean output IS the training target, unifying cleaning + fine-tuning into one task. DPO/self-correction deferred as follow-ups.

### Model task: structured extraction
**Question**: What should the fine-tuned model actually DO?
**Recommended**: Structured extraction (messy doc → clean normalized JSON) — unifies the two goals.
**Chosen**: Structured extraction.
**Rationale**: Makes the dirty→clean framing concrete and crisply defines the teacher's output (clean JSON). Very common real-world use case; excellent pedagogical fit. A concrete domain/schema is still needed (Open Question).

### Source format: text-first, PDF later
**Question**: What source format should the synthetic dirty documents take?
**Recommended**: Text-first (markdown/JSON), defer PDF parsing to a follow-up stage.
**Chosen**: Text-first, PDF later.
**Rationale**: PDF/OCR is a separate skill that would dominate and obscure the fine-tuning learning in a first project. Text keeps focus on the actual skill; PDF becomes a clean follow-up once the core loop works.

### Teacher model: strong API teacher (DeepSeek V4 or GLM 5.2)
**Question**: What teacher will produce the clean extraction targets? (Its quality caps your student.)
**Recommended**: Strong API teacher (GPT-4o/Claude/Gemini) — highest label quality, no local VRAM contention.
**Chosen**: Strong API teacher — specifically **DeepSeek V4 flash/pro** or **GLM 5.2** (developer's preferred candidates).
**Rationale**: Label quality caps the student; an API teacher avoids fighting the single training GPU for VRAM and is cheap at learning scale. The specific teacher will be selected during research based on structured-JSON output quality, price, and API availability of the candidates.

### Base model: Liquid lfm2.5-vl-1.6b (vision-language)
**Question**: Roughly how much VRAM does your GPU have, and which specific model? (Picks base model size; smallest-sufficient is best.)
**Recommended**: 8–16GB → a 1.5B–3B open instruct model via QLoRA (iteration-speed sweet spot; Qwen 2.5/3 cited as an example).
**Chosen**: 8–16GB → **Liquid lfm2.5-vl-1.6b**, a vision-language model.
**Rationale**: Smallest-sufficient model = fastest iteration, maximizing learning per hour. The -vl variant is a deliberate forward-looking choice (see modality decision below). Verify availability and QLoRA-compatibility during research.

### Student model modality: vision-language (forward-looking)
**Question**: lfm2.5-vl-1.6b is a vision-language model, but phase 1 is text-only — is the -vl choice deliberate?
**Recommended**: Keep the VL model (forward-looking — harmonizes with 'text-first, PDF later').
**Chosen**: Keep the VL model.
**Rationale**: Deliberate. The VL capability is unused in the text-first phase but means the PDF/image follow-up is a continuation of the same model (new input type) rather than a swap + re-train.

### Data volume: 500–2,000 docs
**Question**: How many synthetic dirty documents should the pipeline generate?
**Recommended**: 500–2,000 docs — enough for real learning signal, cheap, fast iteration.
**Chosen**: 500–2,000 docs.
**Rationale**: Sufficient for extraction to learn meaningfully; API cost ~$5–30; training iterates in minutes. Larger volumes have diminishing returns for a first project.

### Evaluation: field-level metrics vs gold
**Question**: How will you measure whether the fine-tuned model is any good?
**Recommended**: Automated field-level metrics vs teacher gold on a held-out set (JSON-validity, per-field exact-match, F1).
**Chosen**: Field-level metrics vs gold.
**Rationale**: Extraction has an objective gold target, making exact-match unusually clean and reproducible — and it teaches the ML-evaluation skill most beginners skip. LLM-as-judge is noisier here since a gold target exists.

### Non-goals confirmed
**Question**: Confirm what's explicitly OUT of scope (multiSelect).
**Recommended**: PDF deferred, DPO out, deployment out, MLOps out.
**Chosen**: No explicit selection — firm non-goals stand from prior decisions (PDF deferred, SFT-not-DPO); deployment and MLOps recorded as *inferred* soft non-goals.
**Rationale**: PDF and DPO are firmly excluded by the format and data-role decisions above. Deployment/MLOps were never raised by the developer and are assumed out for a learning project; flagged in Open Questions so they can be pulled in.

## Open Questions
- **Document domain/schema not chosen.** Structured extraction needs a concrete schema (fields to extract), which requires choosing a domain (e.g., invoices, contracts, receipts, shipping manifests). This must be decided before generation so defect types and the JSON schema are coherent. Carry into research/design.
- **Verify lfm2.5-vl-1.6b** — confirm availability on Hugging Face, QLoRA/PEFT fine-tuning compatibility, VRAM fit on 8–16GB, and that its text-input extraction behavior is sound for the text-first phase. Verify during research.
- **Select the specific teacher** — pick among DeepSeek V4 flash, DeepSeek V4 pro, and GLM 5.2 based on structured-JSON output quality, price, and API availability/reliability. Verify during research.
- **Deployment and production MLOps** were only *inferred* out of scope; confirm before design if the developer wants a served endpoint, experiment tracking (W&B), or data versioning (DVC) included.
- **Realism of synthetic dirtiness** — how to model defect types on the developer's actual multi-vendor domain so cleaning skills transfer to real data (needs the domain from the first open question).

## Suggested Follow-ups
- **PDF/image parsing stage** — deferred; natural phase 2. Because the student is a vision-language model, this is a *continuation* (feed images straight to lfm2.5-vl) rather than a model swap; text-layer parsing (PyMuPDF) or OCR may still be useful for certain PDFs.
- **Preference/DPO or self-correction (STaR) training** — the contrastive alternative to SFT; natural follow-up once SFT is mastered.
- **Two-stage curriculum / domain adaptation** (clean synthetic → real cleaned data) — the technique the developer originally wondered about; viable later, not needed for round one.
- **Deploying/serving the fine-tuned model** (inference server / UI) — out of scope this round.
- **Lightweight experiment tracking + data versioning** (W&B, DVC) — optional polish to make iteration more rigorous.

## References
- Input: developer free-text discover invocation, 2026-06-25 ("I want to learn fine tuning…").
- Greenfield project — no existing codebase, git repo, or prior artifacts to reference.
