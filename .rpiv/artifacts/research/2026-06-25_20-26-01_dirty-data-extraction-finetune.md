---
date: 2026-06-25T20:26:01-0300
author: Burke T
commit: no-commit
branch: no-branch
repository: unknown
topic: "Dirty-to-clean synthetic data fine-tuning pipeline"
tags: [research, greenfield, fine-tuning, qlora, lfm2.5-vl, deepseek, structured-extraction, synthetic-data, evaluation, reproducibility]
status: ready
last_updated: 2026-06-25T20:26:01-0300
last_updated_by: Burke T
---

# Research: Dirty-to-clean synthetic data fine-tuning pipeline

## Research Question
For a **greenfield hands-on fine-tuning learning project**, verify and concretize every
external surface of a four-stage local pipeline — (1) generate realistic *dirty* synthetic
docs, (2) label them to clean JSON with a strong API teacher, (3) SFT/QLoRA a small open
vision-language student model, (4) evaluate field-level metrics vs gold. Specifically: does
the chosen student `Liquid lfm2.5-vl-1.6b` actually exist and QLoRA-train within the VRAM
budget? Which teacher (DeepSeek V4 flash/pro vs GLM 5.2) is the best label factory? What
domain/schema and defect taxonomy make synthetic dirtiness both realistic and unambiguously
cleanable? How should field-level extraction be evaluated and proven to beat the base model?
And where does the local reproducibility stack end and MLOps/deployment begin?

The "process is the product" — mastery of the loop is the real win; a working model is a bonus.

## Summary
All five external surfaces are verified and concretized. **The project is fully buildable as
specified** — and a 24GB RTX 3090 (disclosed at checkpoint) gives comfortable headroom.

- **Student base is real & trainable:** `LiquidAI/LFM2.5-VL-1.6B` and a purpose-built
  **`LFM2.5-VL-1.6B-Extract`** (chosen at checkpoint) are verified on Hugging Face. It is a
  dense hybrid conv/GQA model (~1.6B BF16 params, 128K ctx), loads in vanilla `transformers`,
  and QLoRA is documented-feasible at ~16GB (free Colab T4) → ample at 24GB.
- **Framework:** **TRL + PEFT** is the recommended path for this VL model; Unsloth's VL path
  (`FastVisionModel`) is rougher and its text-only `FastLanguageModel` doesn't apply. PEFT
  merge/export is safer/predictable than Unsloth's vision export.
- **Teacher:** **DeepSeek V4 Flash** (`deepseek-v4-flash`) wins decisively — best structured-
  JSON reliability (~0.27% error) and cheapest ($0.14/$0.28 per M tokens) vs GLM 5.2
  (~1.6–2.6% error, $1.40/$4.40). Use JSON mode + `jsonschema` validation + retry-once/repair +
  quarantine.
- **Data:** **Invoices** are the best domain (regular + vendor-varied + deterministic JSON).
  A concrete schema and a **safe-vs-label-breaking defect taxonomy** are defined; the governing
  rule is "only corrupt if a deterministic canonicalizer recovers the same JSON."
- **Eval:** A 3-layer gate — `parse_rate` → `schema_pass_rate` → **canonicalized-leaf micro-F1**
  (primary), `record_exact` (headline). Prove learning with a **paired bootstrap CI vs the base
  model** on the same held-out split.
- **Reproducibility:** Fully local — pinned deps + `set_seed(deterministic=True)` + Trainer
  checkpoints + PEFT adapters + JSONL logs. W&B / DVC / serving are **out of scope** for round 1.

## Detailed Findings

### Student base model — `LiquidAI/LFM2.5-VL-1.6B-Extract`
- **Existence verified.** Liquid's current VL family: `LFM2.5-VL-1.6B`, `LFM2.5-VL-450M`, and
  the extraction-tuned `LFM2.5-VL-1.6B-Extract` (128K ctx, JSON extraction). Older `LFM2-VL-*`
  ids are deprecated. The separate `LFM2.5-8B-A1B` is a text MoE, **not** the VL line.
  (docs.liquid.ai/lfm/models/lfm25-vl-1.6b, .../lfm25-vl-1.6b-extract, HF LiquidAI org)
- **Architecture:** Dense (`LFM2.5-VL (Dense)`), built on `LFM2.5-1.2B-Base` + dynamic SigLIP2
  image encoder; proprietary **hybrid conv/GQA** text backbone (not a plain Transformer, not
  the MoE 8B). HF metadata: 1,596,625,904 BF16 params (~1.6B).
- **Text-first usability:** Loads with standard `AutoModelForImageTextToText` + `AutoProcessor`;
  the model card shows text-only tool use / function calling; use
  `processor.apply_chat_template()`. **No repo-specific wrapper required.** The model is
  genuinely text-capable, so a text-first phase won't break it.
- **VRAM:** Unsloth's LFM2.5 docs cite the 1.6B VL model fitting a free Colab T4 (~16GB) and
  training ~2× faster / ~50% less VRAM. Liquid's conservative TRL VLM recipe: **4-bit NF4,
  `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `max_length=512`,
  `gradient_checkpointing=True`.** → At **24GB (RTX 3090)** this is comfortable; the 8GB
  fallback (`LFM2.5-VL-450M`) is **not needed**.
- **Gap:** No published 8GB QLoRA training run for this exact checkpoint — moot given 24GB.
- **Decision at checkpoint:** chose **`-Extract`** over base. Trade-off noted: it is pre-tuned
  for extraction → likely higher final F1 / faster convergence, but a **smaller visible
  base→fine-tuned delta** (fine for "process is the product"; if a dramatic improvement signal
  is desired later, the base `LFM2.5-VL-1.6B` shows the bigger delta).

### Fine-tuning framework — TRL + PEFT (recommended) vs Unsloth
- **TRL + PEFT wins for this VL model.** Liquid's own catalog marks `LFM2.5-VL-1.6B` trainable
  via **TRL**; Liquid's cookbook lists both "OCR with Unsloth" and "Medical Vision Fine-tuning
  with TRL" for the VL family.
- **Unsloth correction:** `FastLanguageModel` is **text-only**. For LFM2.5-VL you must use
  **`FastVisionModel` + `UnslothVisionDataCollator`**, and Unsloth has a real prior
  `LFM2.5-VL-1.6B` bug (system-prompt normalization + SigLIP2) plus open packing gaps for
  processor-based models. Its vision **merge/export has a bug history** (`save_pretrained_merged`).
- **PEFT merge/export is safer/more predictable** for a first project (`merge_and_unload()` /
  `save_pretrained()`).
- **VLM packing is immature in both stacks** — keep sequences short (`max_length=512`) and
  don't rely on packing for phase 1.
- Practical guidance: if phase-1 ever needed images/OCR → `LFM2.5-VL-1.6B` + TRL+PEFT; if a
  pure-text base were ever wanted → text-only `LFM2.5-1.2B-Instruct` + Unsloth
  `FastLanguageModel` (rejected at checkpoint — breaks forward-looking VL continuity).

### Teacher model — DeepSeek V4 Flash (`deepseek-v4-flash`)
- **IDs verified.** DeepSeek endpoints are `deepseek-v4-flash` and `deepseek-v4-pro` (no single
  `deepseek-v4`). GLM endpoint is `glm-5.2`. (api-docs.deepseek.com, docs.z.ai, HF, OpenRouter)
- **Ranking for a label factory: (1) DeepSeek V4 Flash, (2) DeepSeek V4 Pro, (3) GLM 5.2.**
  DeepSeek has the stronger native JSON contract (JSON Output + beta strict tool schema);
  GLM 5.2 docs only expose `json_object` + app-side validation, and `tool_choice` is auto-only.
- **Proxy metrics (OpenRouter, not SLAs):**
  - DeepSeek V4 Flash — structured-output error **0.27%**, p50 0.57s, 87 tok/s, ~98% uptime,
    **$0.14/M in, $0.28/M out** (cache-hit in $0.0028/M).
  - DeepSeek V4 Pro — 0.79% error, 0.48s, 54 tok/s, ~95.7% uptime, $0.435/$0.87 per M.
  - GLM 5.2 — 1.6–2.6% error, 0.29s (fastest), 85 tok/s, ~94.5% uptime, $1.40/$4.40 per M.
- **Strict-JSON + retry recipe:** single-turn, non-streaming, non-thinking calls; hard-prompt
  JSON with one schema example; `json.loads()` then `jsonschema`/Pydantic validate; **treat
  empty content as failure even on HTTP 200**; retry transport errors (429/500/503 or Z.AI
  1302/1305) with exponential backoff; on semantic failure do **one** repair prompt then
  **quarantine** — never let invalids into the training set.
- **DeepSeek guardrails:** if using tool calls in thinking mode you must preserve
  `reasoning_content` on the assistant turn or it returns 400; avoid `tool_choice="required"`
  on V4 thinking paths (400s); treat `/beta` strict-tool path as probe-only until verified.
- **GLM guardrails (if ever used):** prefer `n=1`, no streaming; if streaming tool calls,
  accumulate `delta.tool_calls[*].function.arguments` by tool index; disable thinking for
  extraction (content can misroute in batched structured-output paths).

### Synthetic data design — Invoices + label-preserving defect taxonomy
- **Domain = Invoices** (chosen at checkpoint). Best balance of regularity + vendor variation +
  deterministic JSON. B2B/industrial invoices beat receipts (too simple), contracts (too
  free-form/ambiguous), and shipping manifests (niche). Evidence: arXiv 2206.11229 (B2B docs
  incl. invoices/POs/delivery notes; "thousands of invoice templates"), DocILE (arXiv 2302.05658),
  RealKIE (arXiv 2403.20101), FATURA (10k invoices / 50 layouts), MIDD (630 supplier invoices).
- **Recommended JSON schema** (amounts/quantities as **strings**, ISO-8601 dates, ISO-4217
  currency, `line_items` in fixed document order, 1–8 line items for phase 1, `null` for absences):
  ```json
  {"vendor_name":"","buyer_name":null,"invoice_number":"","invoice_date":"YYYY-MM-DD",
   "currency":"USD","purchase_order_number":null,"subtotal":null,"tax_total":null,
   "shipping_total":null,"discount_total":null,"grand_total":"",
   "line_items":[{"description":"","quantity":"","unit":null,"unit_price":"","amount":""}]}
  ```
- **Defect taxonomy — governing rule:** *only corrupt if a deterministic canonicalizer can
  recover the same JSON without guessing.* If it needs semantic inference/reconciliation, it is
  **label-breaking**.
  - **SAFE (use):** missing optional fields; extra non-schema boilerplate ("thank you",
    remittance/legal footers); key-label synonyms / block reorder (`Invoice No.`/`Bill #`/
    `Ref.`, totals-above-items) — **labels/blocks only, never values**; date-format drift
    (`2026-06-25`/`25/06/2026`/`25 Jun 2026`); currency/unit formatting drift
    (`$1,234.50`/`USD 1,234.50`, `EA`/`pcs`/`each`); char-level OCR/typos (`Invo1ce`, `T0tal`);
    non-semantic duplicate lines / layout noise (repeated headers, table↔bullets, wrapped rows,
    page breaks).
  - **LABEL-BREAKING (reject or eval-only):** actual value swaps / conflicting facts
    (subtotal↔total, vendor↔buyer, two totals/two currencies); unrecoverable truncation or
    paraphrase of value fields; row permutation without IDs.
- **Best hybrid generation:** template perturbation for layout (markdown table ↔ key:value
  blocks, header-first vs totals-first, repeated headers); rule-based noise for OCR/typos/format
  drift; **LLM paraphrase only for non-schema boilerplate** (never value-bearing fields); **Faker**
  for realistic vendors/buyers/dates/currencies/items; **cleanlab** QA to catch
  near-duplicates/outliers/label issues.
- **Tooling:** nlpaug (`OcrAug`,`KeyboardAug`,`SpellingAug`), textaugment (EDA), textnoisr
  (controlled char noise), scrambledtext (OCR-like noise), Microsoft genalog (template-driven
  doc gen/alignment), Faker, cleanlab.

### Evaluation methodology — 3-layer gate, micro-F1 primary
- **Use a 3-layer gate; do NOT use raw JSON string equality.**
  1. **`parse_rate`** — valid JSON after stripping obvious wrappers.
  2. **`schema_pass_rate`** — `jsonschema`/Pydantic pass (enable `format_checker` if using
     `format` keywords).
  3. **Canonicalized field scoring** — per-leaf exact match after type-aware normalization.
  - If layer 1 or 2 fails, set that record's downstream value score to **0**.
- **Canonicalization rules:** date/datetime → ISO-8601 at declared granularity (exact); number →
  locale-aware `Decimal` (exact or tolerance); quantity+unit → `(base_unit_value, canonical_unit)`
  tuple (exact); string id/code → NFKC + trim + collapse spaces (+ casefold if declared); free
  text → NFKC + trim + collapse (ANLS optional secondary); boolean → map true/false/yes/no/1/0;
  null/missing/empty → distinct sentinels, schema-driven equality; scalar arrays → ordered or
  sorted-multiset; object arrays → **Hungarian item match** → item-level micro-F1.
- **Aggregation:** **micro-F1 over canonicalized leaves** (primary learning metric); macro-F1 by
  field type/path (diagnostic); **`record_exact`** perfect-response (headline business metric);
  optional weighted `overall = Σ(wᵢ·exactᵢ)/Σ(wᵢ)` (w=1 scalars, w=#items arrays — ExtractBench-style).
- **Proving learning:** evaluate base **and** fine-tuned on the **same held-out set**; report
  **delta micro-F1** and **delta record_exact**; claim victory only if a **paired bootstrap CI
  excludes 0**.
- **Evidence/benchmarks:** jsonschema (format only checked with format_checker), DeepEval
  ExactMatch (string-level) / JsonCorrectness (schema-only), RAGAS (too soft for IDs/dates),
  SROIE (content+category F1), DocILE (no date standardization; line-item micro-F1 with max
  matching), KIEval (entity+group Hungarian), **ExtractBench** (per-field string_exact /
  number_tolerance, missing/null handling, arrays matched/missed/spurious), **SOB** (path-flatten
  to leaves, hard parse-gate zeros downstream), StructEval (path rules), NTX (numeric units /
  temporal normalization).
- **Harnesses:** Inspect AI (custom scorer + `grouped()` metrics), lm-evaluation-harness
  (`process_docs`/filters/custom metrics), pandera (flatten line items to DataFrame for
  type/nullability/business-rule checks).

### Reproducibility stack & MLOps boundary — fully local is sufficient
- **Seed control:** `transformers.set_seed(seed, deterministic=True)`; stronger
  `enable_full_determinism(seed)`; `TrainingArguments(seed=, data_seed=, full_determinism=)`
  (data_seed needs `accelerate>=1.1.0`). Caveat: `full_determinism` slows training and has
  multi-node DeepSpeed issues — don't default it on unless needed.
- **Logging:** Trainer callbacks; `report_to` defaults to `"none"`; built-in `PrinterCallback` /
  `TensorBoardCallback`; `logging_strategy` + `logging_steps`. TRL `SFTTrainer` logs global_step,
  epoch, num_tokens, loss, entropy, mean_token_accuracy, lr, grad_norm.
- **Checkpoint/resume:** `save_strategy`/`save_steps`/`save_total_limit`/`save_strategy="best"`/
  `load_best_model_at_end=True`; `trainer.train(resume_from_checkpoint=True)` restores optimizer,
  scheduler, RNG.
- **Dataset versioning:** `load_dataset(..., revision=<sha/tag>)`; `load_from_disk()` for a frozen
  snapshot; `DatasetInfo` tracks version; deterministic split via `shuffle(seed=...)` + persist.
  (HF Hub dataset repos are git-versioned; Storage Buckets are non-versioned/mutable.)
- **PEFT:** `save_pretrained()` saves adapter weights + `adapter_config.json`;
  `PeftModel.from_pretrained()` loads; **pin the base model revision** in a manifest.
- **OUT of scope (confirmed):** remote experiment servers (W&B/MLflow/Trackio/Comet), data
  governance (DVC/orchestration/scheduling), serving/inference (REST/OpenAI-compat, SGLang, LM
  Studio, llama-server), GGUF export-for-deploy. Unsloth's own docs make this split explicit.
- **Minimal local recipe:** generate (log model_revision/seed/do_sample/temp/top_p/top_k/
  max_new_tokens) → label (JSONL/CSV + schema/rubric version) → train (`set_seed(deterministic=True)`,
  `seed`/`data_seed`, local checkpoints, `report_to="none"`) → evaluate (frozen split +
  `compute_metrics` + metrics JSON). Start with **Transformers + Datasets + TRL + PEFT**; add
  Unsloth only if speed/VRAM savings are needed; skip W&B/DVC/servers unless crossing into
  collaboration/orchestration/deployment.

## Code References
*(Greenfield — no source files yet. These are the external assets the planner must instantiate.)*
- `LiquidAI/LFM2.5-VL-1.6B-Extract` (HF) — chosen student base; dense hybrid conv/GQA VL, ~1.6B BF16, 128K.
- `LiquidAI/LFM2.5-VL-1.6B`, `LiquidAI/LFM2.5-VL-450M` — base / 8GB-fallback alternatives.
- `LFM2.5-1.2B-Base` / `LFM2.5-1.2B-Instruct` (HF) — text-only base (rejected; breaks VL continuity).
- `deepseek-v4-flash` / `deepseek-v4-pro` (api-docs.deepseek.com) — teacher endpoints; Flash chosen.
- `glm-5.2` (docs.z.ai) — alternative teacher (not chosen).
- HuggingFace `transformers` + `trl.SFTTrainer` + `peft` (LoraConfig, 4-bit NF4) — training stack.
- `transformers.set_seed` / `TrainingArguments(seed,data_seed,full_determinism)` — determinism knobs.
- `jsonschema` (+ `format_checker`) / Pydantic — label & eval validation.
- `Faker`, `nlpaug`, `textnoisr`, `cleanlab` — synthetic-data generation & QA.

## Integration Points
*(Greenfield — no inbound code consumers yet. Wiring = the pipeline's external dependencies and stage boundaries, all local.)*

### Outbound Dependencies (external services the pipeline calls)
- **DeepSeek API** (`deepseek-v4-flash`) — teacher label factory (stage 2). JSON mode +
  `jsonschema` validate + retry/backoff + quarantine. API key via env var, never committed.
- **Hugging Face Hub** — download student base (`LiquidAI/LFM2.5-VL-1.6B-Extract`) + processor;
  optional dataset push (out of scope for round 1). Pin base model revision in a manifest.
- **Local GPU (RTX 3090, 24GB VRAM, CUDA/BitsAndBytes)** — QLoRA training (stage 3) + inference
  eval (stage 4). 4-bit NF4, batch=1, grad_accum=16, grad-checkpointing, max_len=512.

### Infrastructure Wiring (local pipeline data flow)
- **Stage 1 → 2:** dirty docs (markdown/JSON) on disk → teacher client.
- **Stage 2 → 3:** `(dirty input → clean JSON)` instruction-response records as **JSONL**,
  filtered/quarantined, deterministic `shuffle(seed=...)` train/test split, persisted snapshot.
- **Stage 3 → 4:** PEFT adapter checkpoint (`save_pretrained`) + pinned base revision →
  `PeftModel.from_pretrained` for eval inference.
- **Stage 4:** frozen held-out split + `compute_metrics` → metrics JSON (parse_rate,
  schema_pass_rate, micro-F1, record_exact) for base vs fine-tuned comparison.

### Inbound References
- None yet (greenfield). Downstream consumers will be the `design`/`blueprint` artifacts.

## Architecture Insights
- **Cleaning and fine-tuning are one task.** The teacher's clean JSON *is* the training target;
  the student learns to reproduce the cleanup. Never feed dirty data into SFT hoping the model
  learns to be clean — that teaches the mess.
- **"Fine-tune with a teacher" here = SFT with teacher-generated labels** (not logit distillation,
  not RLHF). Single-stage clean→SFT suffices for round one; DPO/STaR are natural follow-ups.
- **Label-preserving corruption is the central data principle:** corruption is only valid if a
  deterministic canonicalizer recovers the identical gold JSON. This keeps the teacher's label
  unambiguous and the eval objective clean.
- **Eval as a hard-gated pipeline:** parse → schema → canonicalized-leaf scoring. A parse failure
  must zero the downstream value score (SOB pattern), preventing partial-credit noise.
- **Prove learning objectively:** paired bootstrap CI of micro-F1/record_exact vs the base model
  on the same split — the metric that turns "it works" into "it learned."
- **Forward-looking VL continuity:** choosing a VL base means the deferred PDF/image phase is a
  *continuation* (new input type to the same model) rather than a swap + re-train.
- **Reproducibility without MLOps:** seeds + checkpoints + persisted splits + JSONL logs are
  sufficient; heavy experiment-tracking/serving is a deliberate non-goal for a learning project.

## Precedents & Lessons
No git history available (`commit: no-commit`, no repository) — `precedent-locator` was skipped.

### Composite Lessons (synthesized from research)
- **Verify the model name before designing around it.** The developer's "lfm2.5-vl-1.6b" and
  "DeepSeek V4 / GLM 5.2" all proved real, but exact HF repo ids (`LFM2.5-VL-1.6B-Extract`) and
  API endpoint ids (`deepseek-v4-flash`, not `deepseek-v4`) differ from colloquial names — always
  pin canonical ids early.
- **Non-standard architectures have non-standard support.** Liquid's hybrid conv/GQA backbone
  means Unsloth's mature *text* path (`FastLanguageModel`) does not apply to the VL model; the VL
  path (`FastVisionModel`) is rougher and has export bugs. Default to the vendor's own recommended
  stack (Liquid → TRL) for exotic architectures.
- **Teacher reliability > teacher cleverness for a label factory.** Pick the model with the lowest
  structured-output error and best JSON contract (DeepSeek Flash), not the highest general
  benchmark — and always quarantine invalids, since one bad label poisons a training example.
- **The cheapest defensible VRAM recipe wins for iteration speed.** Liquid's conservative
  NF4/batch=1/grad_accum=16/max_len=512/grad-checkpointing recipe is the documented-safe baseline;
  24GB gives room to relax `max_len` or batch size once the loop runs end-to-end.

## Historical Context (from `.rpiv/artifacts/`)
- `.rpiv/artifacts/discover/2026-06-25_19-47-04_dirty-data-extraction-finetune.md` — the FRD this
  research was chained from (intent, goals, decisions, open questions).

## Developer Context

**Discover decisions (carried forward):**
- **Q (discover: Primary goal): What does success look like?** A: "Learn the craft" — mastery of
  the end-to-end pipeline is the real win; a working model is a bonus.
- **Q (discover: Compute): What hardware/access?** A: Local GPU.
- **Q (discover: Role of the data): What role do the "issues/inconsistencies" play?** A: Cleaned
  SFT, re-framed — synthetic flaws are a scaffold to practice cleaning realistic messiness; the
  teacher's clean output IS the target.
- **Q (discover: Model task): What should the model DO?** A: Structured extraction (dirty doc →
  clean normalized JSON).
- **Q (discover: Source format): What source format?** A: Text-first (markdown/JSON); PDF deferred.
- **Q (discover: Teacher model): What teacher produces the targets?** A: Strong API teacher —
  DeepSeek V4 flash/pro or GLM 5.2 (to be selected in research).
- **Q (discover: Base model): VRAM + which model?** A: 8–16GB → Liquid lfm2.5-vl-1.6b (VL).
- **Q (discover: Student modality): Is the -vl choice deliberate given text-first phase 1?** A:
  Keep the VL model (forward-looking — harmonizes with "text-first, PDF later").
- **Q (discover: Data volume): How many docs?** A: 500–2,000.
- **Q (discover: Evaluation): How to measure goodness?** A: Field-level metrics vs teacher gold
  (JSON-validity, per-field exact-match, F1).
- **Q (discover: Non-goals): What's out of scope?** A: PDF deferred, DPO out, deployment out,
  MLOps out (last two inferred).

**Research checkpoint decisions (this artifact):**
- **Q (research: Domain): Which domain should synthetic dirty docs mimic?** A: **Invoices** — best
  regularity + vendor variation + deterministic JSON.
- **Q (research: GPU VRAM): What exact GPU/VRAM?** A: **RTX 3090, 24GB** — well above the 16GB
  documented-feasible threshold; LFM2.5-VL-1.6B-Extract QLoRA has comfortable headroom (450M
  fallback not needed).
- **Q (research: Student base): Base LFM2.5-VL-1.6B vs -Extract vs text-only?** A:
  **LFM2.5-VL-1.6B-Extract** — purpose-built for JSON extraction; higher final F1, smaller visible
  delta (acceptable for "process is the product").
- **Q (research: Teacher API): Which teacher / existing access?** A: **DeepSeek V4 Flash** — best
  JSON reliability + cheapest.

## Related Research
- None yet (first research artifact for this greenfield project).

## Open Questions
*(3 of the FRD's 5 open questions were resolved during the checkpoint; residuals below.)*
1. **Empirical transfer to real data (residual of FRD OQ5).** Research provides a defect taxonomy
   and the label-preserving rule, but whether synthetic *invoice* dirtiness transfers to the
   developer's actual multi-vendor field data can only be validated **empirically after training** —
   by spot-testing the fine-tuned model on real (de-identified) samples. Keep a small holdout of
   real docs if obtainable.
2. **Lightweight experiment tracking / data versioning (residual of FRD OQ4).** Deployment and
   production MLOps are **confirmed out of scope** for round 1. If iteration rigor later matters,
   W&B (metrics) and/or DVC (dataset versioning) are the natural optional add-ons — pull in only if
   the developer wants them.
3. **Visible-delta vs final-score trade-off.** The `-Extract` base may show a small base→fine-tuned
   improvement because it's already good at extraction. If a dramatic "I made it better" signal is
   pedagogically desired, switch to the base `LFM2.5-VL-1.6B` for a larger delta (decidable at
   design time or after a first run).
