---
date: 2026-06-25T21:35:07-0300
author: Burke T
commit: no-commit
branch: no-branch
repository: unknown
topic: "Dirty-to-clean synthetic data fine-tuning pipeline"
tags: [plan, greenfield, fine-tuning, qlora, lfm2.5-vl, deepseek, structured-extraction, synthetic-data, evaluation, reproducibility]
status: ready
parent: .rpiv/artifacts/research/2026-06-25_20-26-01_dirty-data-extraction-finetune.md
phase_count: 8
phases:
  - { n: 1, title: Project scaffold + schema/config foundation }
  - { n: 2, title: Clean synthetic invoice generator }
  - { n: 3, title: Canonicalization + label-preserving corruption }
  - { n: 4, title: Teacher labeling (DeepSeek) }
  - { n: 5, title: Dataset preparation }
  - { n: 6, title: Training (QLoRA SFT) }
  - { n: 7, title: Evaluation harness }
  - { n: 8, title: Orchestration + reproducibility + reflection }
unresolved_phase_count: 0
last_updated: 2026-06-25T21:35:07-0300
last_updated_by: Burke T
---

# Dirty-to-clean SFT/QLoRA Extraction Pipeline — Implementation Plan

## Overview
A `src/doc_extract/` Python package implementing an end-to-end learning pipeline: generate
realistic *dirty* synthetic invoices, label them to clean JSON with the DeepSeek V4 Flash
teacher, SFT/QLoRA fine-tune `LiquidAI/LFM2.5-VL-1.6B-Extract` on the `(dirty→clean)` pairs,
and evaluate with a 3-layer field-level harness (parse→schema→canonicalized-leaf micro-F1)
proving the fine-tuned model beats the base. One Pydantic invoice schema is the single source
of truth driving teacher validation, eval canonicalization, and synthetic generation. Stages
run as `python -m doc_extract.<stage>` CLIs glued by a Makefile for reproducible end-to-end
runs on the local RTX 3090 (24GB). The process is the product.

## Requirements
- Generate 500–2,000 synthetic *dirty* invoice documents (text/markdown) with realistic,
  multi-vendor-style messiness, each paired with a known-clean ground-truth JSON.
- Use DeepSeek V4 Flash (`deepseek-v4-flash`) as a teacher to produce a clean JSON extraction
  target per dirty doc, with retry/repair/quarantine and resumable, idempotent batching.
- Corruption is **label-preserving** by construction: a deterministic canonicalizer must recover
  the identical ground-truth values (verified by an automated invariant test).
- Fine-tune `LiquidAI/LFM2.5-VL-1.6B-Extract` via 4-bit NF4 QLoRA (TRL `SFTTrainer` + PEFT)
  on the local GPU without OOM; save adapter + a merged export.
- Deterministic, seeded train/test split; reproducible from a seed.
- Evaluate on the held-out set: JSON-validity rate, per-field exact-match, overall micro-F1 vs
  the teacher's gold, plus `record_exact`; prove the fine-tuned model beats the base via a
  paired bootstrap confidence interval.
- Reproducible end-to-end via local scripts + Makefile; pinned dependencies; a written
  per-stage reflection ("process is the product" deliverable).

## Current State Analysis
**Greenfield** — no existing codebase, no git repo. Only the discover FRD and the chained
research artifact exist. Python 3.14.6 is installed; no ML libraries are present yet
(installed by this plan). `repo: unknown`, `commit: no-commit`, `branch: no-branch`.

### Key Discoveries (verified during research — see parent research artifact)
- **Student base verified real & trainable:** `LiquidAI/LFM2.5-VL-1.6B-Extract` (dense hybrid
  conv/GQA, ~1.6B BF16, 128K ctx). Loads via `AutoModelForImageTextToText` + `AutoProcessor`;
  QLoRA documented-feasible at ~16GB → ample on 24GB.
- **Training stack:** TRL `SFTTrainer` with `SFTConfig` (the current knob container, a
  `TrainingArguments` subclass). For LFM2.5-VL whose chat template lacks `{% generation %}`
  markers, use **prompt/completion format + `completion_only_loss=True`** (NOT
  messages/`assistant_only_loss`). 4-bit via `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
  bnb_4bit_use_double_quant=True)`. LoRA `target_modules="all-linear"` (explicit LFM module
  names unverified for the hybrid backbone — `all-linear` is the verified-safe default).
  `prepare_model_for_kbit_training` → `get_peft_model`. Merge via `merge_and_unload` on a
  full-precision base. `model.config.use_cache=False` + `gradient_checkpointing_kwargs=
  {"use_reentrant": False}` during training.
- **Teacher verified:** DeepSeek uses the **OpenAI SDK** with `base_url="https://api.deepseek.com"`
  (no `/v1`, no official DeepSeek python SDK). Model `deepseek-v4-flash`. JSON mode =
  `response_format={"type":"json_object"}` + the word "json" in the prompt; **thinking defaults
  ON → disable via `extra_body={"thinking":{"type":"disabled"}}`**; `temperature=0`.
  Empty content can occur on HTTP 200 → treat as failure.
- **Eval contract:** 3-layer gate (parse → schema → canonicalized-leaf); parse/schema failure
  zeros downstream score. Micro-F1 over canonicalized leaves is the primary learning metric;
  `record_exact` is the headline.
- **Determinism:** `transformers.set_seed(seed, deterministic=True)` + `TrainingArguments/
  SFTConfig(seed=, data_seed=)`; deterministic split via `shuffle(seed=...)` persisted to a snapshot.

### Patterns to follow
- TRL's own `examples/scripts/sft_vlm.py` (current VLM SFT shape).
- DeepSeek `api-docs.deepseek.com/guides/json_mode` (JSON-mode call + empty-content caveat).
- Liquid's conservative VLM QLoRA recipe (NF4, batch=1, grad_accum, max_len, grad-checkpointing).

### Constraints to work within
- Local RTX 3090 24GB; keep sequences short (`max_length` ≤ 1024) since VLM packing is immature.
- DeepSeek API key via env var (`DEEPSEEK_API_KEY`), never committed. Synthetic data only — no PII.
- Python 3.14 is bleeding-edge; some ML libs may lag — pin versions (see Verification Notes).

## Desired End State
A developer runs the full loop locally and sees objective proof of learning:

```bash
# one-time environment
python -m venv .venv && source .venv/bin/activate
pip install -e .

# the loop (each stage is also runnable standalone)
make data      # generate → corrupt → (uses DEEPSEEK_API_KEY) teacher-label
make prepare   # filter + deterministic split + SFT-format snapshot
make train     # QLoRA SFT on LFM2.5-VL-1.6B-Extract → adapter + merged
make evaluate  # base vs fine-tuned on held-out → metrics.json + verdict

# or all at once
make all
```

`evaluate` prints and writes `metrics.json`:
```json
{"base":     {"parse_rate":1.0,"schema_pass_rate":0.98,"micro_f1":0.71,"record_exact":0.34},
 "finetuned":{"parse_rate":1.0,"schema_pass_rate":1.00,"micro_f1":0.93,"record_exact":0.71},
 "delta_micro_f1":0.22,"delta_record_exact":0.37,
 "paired_bootstrap_ci_micro_f1":[0.14,0.30],"learning_proven":true}
```

## What We're NOT Doing
- **PDF / OCR / images** — text/markdown/JSON only this round (deferred phase 2; the VL base
  keeps it a continuation, not a swap).
- **DPO / preference / STaR / self-correction training** — single-stage SFT only.
- **Deployment / serving** (REST, SGLang, LM Studio, llama-server) and **GGUF export-for-deploy**.
- **Production MLOps** — W&B/MLflow experiment servers, DVC, orchestration, CI (confirmed out).
- **Real-data transfer validation** — empirical spot-testing on real vendor data is a
  post-training follow-up, out of scope for this plan (only the synthetic scaffold is built).
- Not modifying any existing files (greenfield — everything is NEW).

## Decisions

### Schema source of truth → Pydantic model
**Ambiguity:** one schema must drive teacher JSON validation (jsonschema), eval canonicalization
(needs field-type metadata), and the synthetic generator (needs the field shape).
**Explored:** (A) Pydantic model → `.model_json_schema()` for teacher + field-type registry for
eval — single source, typed, pedagogically clear; (B) plain jsonschema dict — less ceremony but
field types re-derived by hand for canonicalization.
**Decision:** **Pydantic** (developer-confirmed). `schema.py` defines `Invoice`/`LineItem`;
exposes `INVOICE_JSON_SCHEMA` (via `model_json_schema()`) and a `FIELD_TYPE_REGISTRY` mapping
each leaf path → canonical kind (date/amount/quantity/string/null).

### Project layout → src package + per-stage CLIs + Makefile
**Decision:** `src/doc_extract/` package, each stage module runnable via `python -m
doc_extract.<stage>` with an argparse `main()`, plus a `Makefile` to run end-to-end.
(Developer-confirmed; rejected flat scripts/notebooks — shared schema/config imports get awkward.)

### Tests → targeted pytest on invariants
**Decision:** pytest unit tests for the deterministic, risk-bearing pieces: corruption
round-trips through the canonicalizer to the identical gold; canonicalization/normalization edge
cases; deterministic split reproducibility. (Developer-confirmed; the eval harness is NOT the
only test — a silent label-breaker could poison a run before eval catches it.)

### Teacher labeling → sequential + resumable
**Decision:** one call at a time with append-only JSONL + skip-seen-ids resume. Simplest to
reason about, trivially resumable on rate-limit, ideal for a learning project. ~4–17 min total
is acceptable. (Developer-confirmed; rejected concurrent threadpool.)

### Eval harness → custom canonicalized-leaf harness (not lm-eval/Inspect)
**Decision:** build a small custom 3-layer harness rather than wrangle lm-eval-harness's task
API for a custom JSON-extraction task. The metric is itself the ML-eval skill being learned;
canonicalization primitives (`canon.py`) are shared with the generation invariant.

### Student / teacher / data / modality → inherited from research (locked)
- Student: `LiquidAI/LFM2.5-VL-1.6B-Extract`, TRL+PEFT, 4-bit NF4 (research-confirmed).
- Teacher: `deepseek-v4-flash`, JSON mode, thinking disabled, retry/repair/quarantine.
- Data: invoices; label-preserving corruption; 500–2,000 docs (defaulted ~500 for first run,
  knob to scale).
- Modality: VL base kept for forward-looking PDF continuity; phase-1 is text-only.

## Phase 1: Project scaffold + schema/config foundation
### Overview
Foundation: make the package importable/installable and define the Pydantic invoice schema +
central config that every later phase depends on. Depends on nothing.

### Changes Required:

#### 1. pyproject.toml
**File**: pyproject.toml
**Changes**: NEW — package metadata, pinned dependencies (transformers/trl/peft/bitsandbytes/datasets/torch/pydantic/openai/tenacity/jsonschema/faker; dev: pytest/ruff), ruff + pytest config, src layout. No `[project.scripts]` — CLIs run via `python -m doc_extract.<stage>` (each stage module ships its own `__main__` guard).

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "doc-extract"
version = "0.1.0"
description = "Dirty-to-clean synthetic-data SFT/QLoRA extraction pipeline (learning project)"
requires-python = ">=3.11"
readme = "README.md"
authors = [{ name = "Burke T" }]
dependencies = [
    "transformers>=4.46,<5",
    "trl>=0.15,<0.17",
    "peft>=0.13,<1",
    "accelerate>=1.1,<2",
    "bitsandbytes>=0.44,<1",
    "datasets>=3,<4",
    "torch>=2.3,<3",
    "pydantic>=2.7,<3",
    "openai>=1.40,<2",
    "tenacity>=8.5,<10",
    "jsonschema>=4.23,<5",
    "scipy>=1.13,<2",
    "faker>=28,<30",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9", "ruff>=0.6,<1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

#### 2. requirements.txt
**File**: requirements.txt
**Changes**: NEW — pinned, installable dependency list mirroring pyproject (for `pip install -r`).

```text
transformers>=4.46,<5
trl>=0.15,<0.17
peft>=0.13,<1
accelerate>=1.1,<2
bitsandbytes>=0.44,<1
datasets>=3,<4
torch>=2.3,<3
pydantic>=2.7,<3
openai>=1.40,<2
tenacity>=8.5,<10
jsonschema>=4.23,<5
scipy>=1.13,<2
faker>=28,<30
pytest>=8,<9
ruff>=0.6,<1
```

#### 3. README.md
**File**: README.md
**Changes**: NEW — minimal stub so `readme = "README.md"` in pyproject resolves at install. Phase 8 expands this to full setup/usage docs (MODIFY).

```markdown
# doc-extract

Dirty-to-clean synthetic-data SFT/QLoRA extraction pipeline — a hands-on learning project.

> The process is the product: mastery of the end-to-end fine-tuning loop is the goal; a working
> model is a bonus.

Full setup and usage docs are added in a later phase. See
`.rpiv/artifacts/plans/2026-06-25_21-35-07_dirty-data-extraction-finetune.md` for the plan.
```

#### 4. src/doc_extract/__init__.py
**File**: src/doc_extract/__init__.py
**Changes**: NEW — package init + version.

```python
"""doc_extract: dirty-to-clean synthetic-data SFT/QLoRA extraction pipeline.

Four stages: generate dirty synthetic docs -> teacher-label to clean JSON -> QLoRA SFT on
LFM2.5-VL-1.6B-Extract -> field-level evaluation vs teacher gold. "The process is the product."
"""

__version__ = "0.1.0"
```

#### 5. src/doc_extract/schema.py
**File**: src/doc_extract/schema.py
**Changes**: NEW — Pydantic `LineItem`/`Invoice` models (amounts/quantities as strings, ISO-8601 dates, ISO-4217 currency, nullable optional fields). Constraints are encoded IN the JSON Schema — `invoice_date` via `Field(pattern=...)` (→ "pattern"), `currency` via `Literal` (→ "enum") — so the teacher validator enforces them structurally. Exposes `INVOICE_JSON_SCHEMA = Invoice.model_json_schema()` and `FIELD_TYPE_REGISTRY` (leaf path → canonical kind; 16 leaves).

```python
"""Single source of truth for the invoice extraction schema.

A Pydantic model defines the fields; `.model_json_schema()` feeds the teacher's jsonschema
validation (Phase 4) AND encodes the currency/date constraints structurally, and
`FIELD_TYPE_REGISTRY` feeds the eval canonicalizer (Phase 3/7). Amounts/quantities are strings
to avoid float drift; dates are ISO-8601; currency is ISO-4217.
"""

from __future__ import annotations

import typing
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

# ISO-4217 subset. Single source: the schema encodes it as an enum, the generator reads the set.
Currency = Literal["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CNY", "INR"]
CURRENCIES: set[str] = set(typing.get_args(Currency))

_ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """A single invoice line. amount == quantity * unit_price (the generator keeps it consistent)."""

    description: str
    quantity: str  # string to preserve "12" / "12.5" / "1,000"
    unit: Optional[str] = None  # e.g. EA, pcs, each, kg
    unit_price: str  # string amount
    amount: str  # string amount


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor_name: str
    buyer_name: Optional[str] = None
    invoice_number: str
    invoice_date: str = Field(pattern=_ISO_DATE_PATTERN)  # -> "pattern" in JSON Schema
    currency: Currency = "USD"  # -> "enum" in JSON Schema
    purchase_order_number: Optional[str] = None
    subtotal: Optional[str] = None
    tax_total: Optional[str] = None
    shipping_total: Optional[str] = None
    discount_total: Optional[str] = None
    grand_total: str
    line_items: list[LineItem] = Field(min_length=1, max_length=8)


# Canonical JSON Schema (Draft 2020-12) handed to the teacher + the jsonschema validator.
INVOICE_JSON_SCHEMA = Invoice.model_json_schema()

# Leaf path -> canonical kind. Drives BOTH generation-time canonicalization (Phase 3) and
# eval-time leaf scoring (Phase 7). Nullable kinds allow the value to be None.
FIELD_TYPE_REGISTRY: dict[str, str] = {
    "vendor_name": "string",
    "buyer_name": "string_nullable",
    "invoice_number": "string",
    "invoice_date": "date",
    "currency": "currency",
    "purchase_order_number": "string_nullable",
    "subtotal": "amount_nullable",
    "tax_total": "amount_nullable",
    "shipping_total": "amount_nullable",
    "discount_total": "amount_nullable",
    "grand_total": "amount",
    "line_items[].description": "string",
    "line_items[].quantity": "quantity",
    "line_items[].unit": "unit_nullable",
    "line_items[].unit_price": "amount",
    "line_items[].amount": "amount",
}

NULLABLE_KINDS = {k for k, v in FIELD_TYPE_REGISTRY.items() if v.endswith("_nullable")}
```

#### 6. src/doc_extract/config.py
**File**: src/doc_extract/config.py
**Changes**: NEW — data dir layout, `SEED`/`DATA_SEED`, student/teacher model ids, student model revision (`"main"` for learning; a concrete SHA is captured into the runtime manifest for reproducibility), default doc count, eval bootstrap params, path helpers + `ensure_dirs()`.

```python
"""Central config: reproducibility seeds, pinned model ids, and data/checkpoint paths."""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---
SEED = 42
DATA_SEED = 42

# --- Data volume (small for the first end-to-end run; scale via CLI --n-docs) ---
DEFAULT_N_DOCS = 500
TRAIN_SPLIT = 0.9

# --- Model ids (research-verified) ---
STUDENT_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-Extract"
# "main" for an iterative learning run; for a strictly reproducible run, capture the resolved
# commit SHA (model.config._commit_hash at load time) into the split manifest and pin it here.
STUDENT_REVISION = "main"
TEACHER_MODEL_ID = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # NOTE: no /v1 (research-verified)

# --- Teacher labeling ---
TEACHER_MAX_TOKENS = 2048
TEACHER_TEMPERATURE = 0  # ignored when thinking is enabled, but we disable thinking

# --- Evaluation ---
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# --- Paths (resolved relative to the repo root = two parents above this file) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
ADAPTER_DIR = CHECKPOINT_DIR / "adapter"
MERGED_DIR = CHECKPOINT_DIR / "merged"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Stage data files
CLEAN_JSONL = DATA_DIR / "clean.jsonl"
DIRTY_JSONL = DATA_DIR / "dirty.jsonl"
LABELED_JSONL = DATA_DIR / "labeled.jsonl"
QUARANTINE_JSONL = DATA_DIR / "quarantine.jsonl"
SFT_DIR = DATA_DIR / "sft"


def ensure_dirs() -> None:
    """Create all output directories. Safe to call repeatedly."""
    for p in (DATA_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR, ADAPTER_DIR, MERGED_DIR, SFT_DIR):
        p.mkdir(parents=True, exist_ok=True)
```

### Success Criteria:

#### Automated Verification:
- [x] Package installs in dev mode: `pip install -e .[dev]` exits 0 (use Python 3.11/3.12 if 3.14 wheels are missing).
- [x] JSON Schema encodes constraints: `python -c "from doc_extract.schema import INVOICE_JSON_SCHEMA; p=INVOICE_JSON_SCHEMA['properties']; assert 'pattern' in p['invoice_date'] and 'enum' in p['currency']; print('ok')"` prints ok.
- [x] Schema round-trips: a valid invoice dict constructs via `Invoice(**d).model_dump()` and `Invoice.model_json_schema()` has an `"properties"` key.
- [x] Schema rejects bad input: `invoice_date:'25/06/2026'` raises `ValidationError`; currency `'XYZ'` raises `ValidationError`.
- [x] Import smoke: `python -c "from doc_extract import schema, config; assert config.STUDENT_MODEL_ID=='LiquidAI/LFM2.5-VL-1.6B-Extract'; assert config.DEEPSEEK_BASE_URL=='https://api.deepseek.com'; assert len(schema.FIELD_TYPE_REGISTRY)==16; print('ok')"` prints ok.
- [x] README resolves: `test -f README.md` passes (so `readme="README.md"` in pyproject is valid at install).

#### Manual Verification:
- [x] `schema.FIELD_TYPE_REGISTRY` has an entry for every leaf of `Invoice` (11 scalar + 5 line_items leaves = 16), none missing/extra.
- [x] `config` paths resolve under the repo root and `ensure_dirs()` creates them without error.

## Phase 2: Clean synthetic invoice generator
### Overview
Build realistic *clean* invoices from Faker + 3–5 markdown template families, emitting the
ground-truth clean JSON plus rendered clean text. Depends on Phase 1; produces `clean.jsonl`.

### Changes Required:

#### 1. src/doc_extract/generate.py
**File**: src/doc_extract/generate.py
**Changes**: NEW — Faker-based clean invoice builder (vendors, buyers, dates, currencies, 1–8 line items, self-consistent totals), 4 markdown template renderers (wholesale/services/compact/freight), seeded RNG, argparse `main()` writing `clean.jsonl` ({id, clean_json, clean_text, template}). Zero tax/shipping/discount → null (absence ⟺ null); currency code rendered verbatim for unambiguous gold.

```python
"""Phase 2: generate realistic CLEAN synthetic invoices and render them to markdown text.

Each record carries a known-clean ground-truth JSON (validated against the Pydantic schema) and a
rendered clean_text. Phase 3 corrupts clean_text -> dirty_text; the teacher (Phase 4) extracts it
back to JSON. Totals are self-consistent; zero totals are null so a field's absence in text maps
exactly to a null gold value (no zero-vs-null ambiguity), and the currency code is rendered
verbatim so extraction is unambiguous.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from doc_extract import config
from doc_extract.schema import CURRENCIES, Invoice

UNITS = ["EA", "pcs", "each", "kg", "box", "set", "hr", "m", "L"]
ITEM_NOUNS = [
    "widget", "gasket", "bearing", "valve", "bracket", "harness", "module", "actuator",
    "nozzle", "assembly", "filter", "sensor", "fixture", "fastener", "unit",
]
_CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "CAD": "C$", "AUD": "A$", "INR": "₹",
}


def _money(rng: random.Random, lo: float, hi: float) -> str:
    return f"{rng.uniform(lo, hi):.2f}"


def _quantity(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return str(rng.randint(1, 25))
    return f"{round(rng.uniform(1, 50), 2):.2f}"


def build_clean_invoice(faker: Faker, rng: random.Random) -> dict:
    """Build a self-consistent clean invoice dict; validated against the Pydantic schema."""
    n_items = rng.randint(1, 8)
    line_items = []
    for _ in range(n_items):
        unit_price = _money(rng, 5, 800)
        q = _quantity(rng)
        amount = f"{float(q) * float(unit_price):.2f}"
        line_items.append({
            "description": f"{faker.word().capitalize()} {rng.choice(ITEM_NOUNS)}".strip(),
            "quantity": q,
            "unit": rng.choice(UNITS),
            "unit_price": unit_price,
            "amount": amount,
        })
    subtotal = sum(float(li["amount"]) for li in line_items)
    tax = subtotal * rng.choice([0.0, 0.05, 0.075, 0.10])
    shipping = rng.choice([0.0, 15.0, 25.0, 45.0])
    discount = rng.choice([0.0, round(subtotal * 0.02, 2)])
    grand = subtotal + tax + shipping - discount
    inv = {
        "vendor_name": faker.company(),
        "buyer_name": faker.company() if rng.random() < 0.7 else None,
        "invoice_number": f"INV-{rng.randint(1000, 99999)}",
        "invoice_date": (date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))).isoformat(),
        "currency": rng.choice(sorted(CURRENCIES)),
        "purchase_order_number": f"PO-{rng.randint(100, 9999)}" if rng.random() < 0.5 else None,
        "subtotal": f"{subtotal:.2f}",
        # zero totals -> None so a field's absence in text maps exactly to null gold
        "tax_total": f"{tax:.2f}" if tax else None,
        "shipping_total": f"{shipping:.2f}" if shipping else None,
        "discount_total": f"{discount:.2f}" if discount else None,
        "grand_total": f"{grand:.2f}",
        "line_items": line_items,
    }
    Invoice(**inv)  # validates + raises on inconsistency
    return inv


def _amt(currency: str, value: str) -> str:
    return f"{_CURRENCY_SYMBOL.get(currency, '')}{value}"


def _totals_lines(inv: dict) -> list[str]:
    c = inv["currency"]
    out = [f"Subtotal: {_amt(c, inv['subtotal'])}"]
    if inv["tax_total"] is not None:
        out.append(f"Tax: {_amt(c, inv['tax_total'])}")
    if inv["shipping_total"] is not None:
        out.append(f"Shipping: {_amt(c, inv['shipping_total'])}")
    if inv["discount_total"] is not None:
        out.append(f"Discount: -{_amt(c, inv['discount_total'])}")
    out.append(f"TOTAL DUE: {_amt(c, inv['grand_total'])}")
    return out


def render_wholesale(inv: dict) -> str:
    c = inv["currency"]
    lines = ["# INVOICE", "", inv["vendor_name"]]
    if inv.get("buyer_name"):
        lines += ["", f"Bill To: {inv['buyer_name']}"]
    lines += ["", f"Invoice No: {inv['invoice_number']}", f"Date: {inv['invoice_date']}",
              f"Currency: {inv['currency']}"]
    if inv.get("purchase_order_number"):
        lines.append(f"PO: {inv['purchase_order_number']}")
    lines += ["", "| Description | Qty | Unit | Unit Price | Amount |",
              "|---|---|---|---|---|"]
    for li in inv["line_items"]:
        lines.append(
            f"| {li['description']} | {li['quantity']} | {li['unit']} | "
            f"{_amt(c, li['unit_price'])} | {_amt(c, li['amount'])} |"
        )
    lines += [""] + _totals_lines(inv)
    return "\n".join(lines)


def render_services(inv: dict) -> str:
    c = inv["currency"]
    lines = [f"{inv['vendor_name']} — Invoice"]
    if inv.get("buyer_name"):
        lines.append(f"Client: {inv['buyer_name']}")
    lines.append(f"Ref {inv['invoice_number']}  |  {inv['invoice_date']}  |  {inv['currency']}")
    if inv.get("purchase_order_number"):
        lines.append(f"PO {inv['purchase_order_number']}")
    lines.append("")
    for li in inv["line_items"]:
        lines.append(
            f"- {li['description']}: {li['quantity']} {li['unit']} x "
            f"{_amt(c, li['unit_price'])} = {_amt(c, li['amount'])}"
        )
    lines.append("")
    lines += _totals_lines(inv)
    return "\n".join(lines)


def render_compact(inv: dict) -> str:
    c = inv["currency"]
    buyer = f" to {inv['buyer_name']}" if inv.get("buyer_name") else ""
    po = f" (PO {inv['purchase_order_number']})" if inv.get("purchase_order_number") else ""
    lines = [f"{inv['invoice_number']} | {inv['vendor_name']}{buyer} | "
             f"{inv['invoice_date']}{po} | {inv['currency']}"]
    for li in inv["line_items"]:
        lines.append(
            f"{li['description']} [{li['quantity']} {li['unit']} @ {_amt(c, li['unit_price'])}] "
            f"{_amt(c, li['amount'])}"
        )
    lines.append("; ".join(_totals_lines(inv)))
    return "\n".join(lines)


def render_freight(inv: dict) -> str:
    # totals-first layout; currency code shown verbatim in the TOTAL line
    c = inv["currency"]
    lines = [f"TOTAL: {_amt(c, inv['grand_total'])}  ({inv['currency']})", "",
             inv["vendor_name"], f"Invoice {inv['invoice_number']} — {inv['invoice_date']}"]
    if inv.get("buyer_name"):
        lines.append(f"Consignee: {inv['buyer_name']}")
    if inv.get("purchase_order_number"):
        lines.append(f"PO {inv['purchase_order_number']}")
    lines.append("")
    for i, li in enumerate(inv["line_items"], 1):
        lines.append(
            f"{i}. {li['description']} — qty {li['quantity']} {li['unit']} @ "
            f"{_amt(c, li['unit_price'])} → {_amt(c, li['amount'])}"
        )
    lines.append("")
    for t in _totals_lines(inv):
        lines.append(f"  {t}")
    return "\n".join(lines)


TEMPLATES = {
    "wholesale": render_wholesale,
    "services": render_services,
    "compact": render_compact,
    "freight": render_freight,
}


def generate(n_docs: int, seed: int, out_path: Path) -> int:
    faker = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    template_names = list(TEMPLATES)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n_docs):
            inv = build_clean_invoice(faker, rng)
            tname = rng.choice(template_names)
            text = TEMPLATES[tname](inv)
            rec = {"id": f"doc-{i:05d}", "clean_json": inv, "clean_text": text, "template": tname}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate clean synthetic invoices -> clean.jsonl")
    p.add_argument("--n-docs", type=int, default=config.DEFAULT_N_DOCS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", type=Path, default=config.CLEAN_JSONL)
    args = p.parse_args(argv)
    config.ensure_dirs()
    n = generate(args.n_docs, args.seed, args.out)
    print(f"wrote {n} clean docs -> {args.out}")


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [x] CLI runs: `python -m doc_extract.generate --n-docs 20 --seed 42 --out /tmp/clean_p2.jsonl` writes exactly 20 lines.
- [x] All clean_json validate: `python -c "import json;from doc_extract.schema import Invoice;[Invoice(**json.loads(l)['clean_json']) for l in open('/tmp/clean_p2.jsonl')]"` exits 0.
- [x] Totals self-consistent: for every record, `abs(sum(line amounts) + (tax or 0) + (shipping or 0) - (discount or 0) - grand_total) < 0.01` and each `abs(qty*unit_price - amount) < 0.01`.
- [x] Null/value consistency: for tax/shipping/discount, `value is None` ⟺ the field label is absent from clean_text; and no non-null total equals "0.00".
- [x] Currency code verbatim: every clean_text contains `rec['clean_json']['currency']` (e.g. "USD") as a substring.
- [x] Determinism: re-running `--seed 42` yields byte-identical output (`cmp` of two runs passes).

#### Manual Verification:
- [x] All 4 templates appear across the 20 docs; each clean_text shows the currency code and every non-null scalar field's value verbatim.

## Phase 3: Canonicalization + label-preserving corruption
### Overview
Introduce type-aware canonicalization primitives (`canon.py`) and the **safe** corruption
taxonomy (`corrupt.py`), verified by an invariant test that corruption round-trips to the
identical gold. Depends on Phases 1–2. **`canon.py` is reused by Phase 7 eval** (cross-phase).

### Changes Required:

#### 1. src/doc_extract/canon.py
**File**: src/doc_extract/canon.py
**Changes**: NEW — type-aware value normalizers (`normalize_date`, `normalize_amount`, `normalize_quantity`, `normalize_unit`, `normalize_currency`, `normalize_string`) + `normalize_value(value, kind, currency)` dispatcher + `canonicalize_invoice(dict) -> dict`; uses `FIELD_TYPE_REGISTRY` from `schema.py`. Foundation for both the generation invariant (here) and eval leaf scoring (Phase 7).

```python
"""Type-aware canonicalization primitives.

Shared by Phase 3 (label-preserving invariant) and Phase 7 (eval leaf scoring). Each normalizer
maps a possibly-messy value to a canonical string so format-only differences collapse to equality.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from doc_extract.schema import FIELD_TYPE_REGISTRY

_DATE_FORMATS = [
    "%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y",
]
_UNIT_ALIASES = {"ea": "EA", "each": "EA", "pc": "EA", "pcs": "EA", "unit": "EA"}


def normalize_date(value):
    if value is None:
        return None
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {value!r}")


def normalize_amount(value, currency=None):
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip()
    s = re.sub(r"^(C\$|A\$|[¥$€£₹]|[A-Za-z]{3})\s?", "", s)
    s = s.replace(",", "").strip()
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        raise ValueError(f"unparseable amount: {value!r}")
    if "." not in s:
        s += ".00"
    elif len(s.split(".", 1)[1]) == 1:
        s += "0"
    return s


def normalize_quantity(value):
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip().replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        raise ValueError(f"unparseable quantity: {value!r}")
    return str(int(float(s))) if float(s).is_integer() else f"{float(s):.2f}"


def normalize_unit(value):
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return _UNIT_ALIASES.get(s, s.upper())


def normalize_currency(value):
    if value is None:
        return None
    return str(value).strip().upper()


def normalize_string(value):
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", s).strip()


def normalize_value(value, kind, currency=None):
    if value is None:
        return None
    base = kind.replace("_nullable", "")
    if base == "date":
        return normalize_date(value)
    if base == "amount":
        return normalize_amount(value, currency)
    if base == "quantity":
        return normalize_quantity(value)
    if base == "unit":
        return normalize_unit(value)
    if base == "currency":
        return normalize_currency(value)
    return normalize_string(value)


def canonicalize_invoice(inv):
    """Return a deep copy of inv with every leaf normalized. Structure preserved (incl. order)."""
    cur = inv.get("currency")

    def norm(obj, path):
        if isinstance(obj, dict):
            return {k: norm(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [norm(v, f"{path}[]") for v in obj]
        kind = FIELD_TYPE_REGISTRY.get(path, "string")
        return normalize_value(obj, kind, cur)

    return norm(inv, "")
```

#### 2. src/doc_extract/corrupt.py
**File**: src/doc_extract/corrupt.py
**Changes**: NEW — label-preserving corruption. **Implemented taxonomy (5 transforms):** date-format drift (`_corrupt_dates`, day-first slash only — unambiguous under canon), amount/currency formatting drift (`_corrupt_amounts`, in-place reformat via regex sub), OCR-noisy boilerplate injection (`_add_boilerplate` + `_ocr_word`), structural-line repeat (`_repeat_header`, restricted to `#` headers + the markdown table separator row only), whole-block reorder (`_reorder_blocks`). **Excluded by design:** key-label/layout synonym drift is provided by the 4 Phase-2 template families (not the corruptor); dropping a *present* optional field is label-breaking, so optional-field nulling is the generator's job, not the corruptor's. `is_label_preserving(...)` = value-token multiset survival (rejects value loss + injection). API: `apply_corruptions(clean_text, clean_json, rng) -> dirty_text`, `corrupt_file(in, out, seed)`, argparse `main()` writing `dirty.jsonl`.

```python
"""Phase 3: apply LABEL-PRESERVING corruption to clean invoice text -> dirty text.

Governing rule: a corruption is valid only if a deterministic canonicalizer recovers the identical
gold values. Corruption perturbs ONLY surface format and non-value text. `is_label_preserving` is
a value-token survival check: it rejects value LOSS and value INJECTION (the real failure modes).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

from doc_extract import canon, config

_CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "CAD": "C$", "AUD": "A$", "INR": "₹",
}

_MONEY_RE = re.compile(r"(C\$|A\$|[¥$€£₹]|[A-Z]{3})\s?([\d,]+\.\d{2})")
_MONEY_TOKEN_RE = re.compile(r"(?:C\$|A\$|[¥$€£₹]|[A-Z]{3})\s?[\d,]+\.\d{2}")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_TOKEN_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]{3} \d{4}|\d{1,2}/\d{1,2}/\d{4}"
    r"|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2}-[A-Za-z]{3}-\d{4}"
)
_DATE_FORMATS = ["%d %b %Y", "%d/%m/%Y", "%B %d, %Y", "%d-%b-%Y"]
_TABLE_SEP_RE = re.compile(r"^\|[-|\s:]*-+[-|\s:]*\|?$")

_BOILERPLATE = [
    "Thank you for your business.",
    "Please remit payment within the stated terms.",
    "This document was generated automatically.",
    "For questions about this invoice contact billing.",
    "Goods remain property of the vendor until paid in full.",
]
_OCR_MAP = {"o": "0", "O": "0", "l": "1", "e": "3", "a": "@", "i": "1", "s": "5"}


def _value_multisets(text):
    dates = sorted(canon.normalize_date(d) for d in _DATE_TOKEN_RE.findall(text))
    amounts = sorted(canon.normalize_amount(a) for a in _MONEY_TOKEN_RE.findall(text))
    return dates, amounts


def _gold_string_tokens(clean_json):
    """Key gold string values that corruption must never drop (vendor, invoice #, currency code,
    buyer if present, PO if present, line-item descriptions). Dates/amounts are covered by the
    multiset check above; these cover the remaining recoverable value leaves."""
    toks = []
    for k in ("vendor_name", "invoice_number", "currency"):
        v = clean_json.get(k)
        if v:
            toks.append(str(v))
    for k in ("buyer_name", "purchase_order_number"):
        v = clean_json.get(k)
        if v:
            toks.append(str(v))
    for li in clean_json.get("line_items", []):
        if li.get("description"):
            toks.append(str(li["description"]))
    return toks


def is_label_preserving(clean_text, clean_json, dirty_text):
    """True iff the date/amount value-token multisets are unchanged AND every key gold string
    token still survives in dirty_text. Rejects value loss + injection (multiset) and value-string
    deletion (substring check)."""
    cd, ca = _value_multisets(clean_text)
    dd, da = _value_multisets(dirty_text)
    if cd != dd or ca != da:
        return False
    for tok in _gold_string_tokens(clean_json):
        if tok not in dirty_text:
            return False
    return True


def _corrupt_dates(text, rng):
    def repl(m):
        iso = m.group(0)
        try:
            d = datetime.strptime(iso, "%Y-%m-%d")
        except ValueError:
            return iso
        return d.strftime(rng.choice(_DATE_FORMATS))
    return _ISO_DATE_RE.sub(repl, text)


def _corrupt_amounts(text, rng, currency):
    sym = _CURRENCY_SYMBOL.get(currency, "")
    code = currency

    def repl(m):
        num = m.group(2)
        digits = num.replace(",", "")
        marker = code if rng.random() < 0.5 else (sym or code)
        sep = " " if marker.isalpha() else ""
        if rng.random() < 0.5 and "." in digits:
            ip, fr = digits.split(".", 1)
            if ip.isdigit() and int(ip) >= 1000:
                ip = f"{int(ip):,}"
            digits = f"{ip}.{fr}"
        return f"{marker}{sep}{digits}"
    return _MONEY_RE.sub(repl, text)


def _ocr_word(word, rng):
    if len(word) < 4:
        return word
    return "".join(_OCR_MAP.get(c, c) if (c in _OCR_MAP and rng.random() < 0.15) else c
                   for c in word)


def _add_boilerplate(text, rng):
    lines = rng.sample(_BOILERPLATE, k=rng.randint(1, 2))
    noisy = [" ".join(_ocr_word(w, rng) for w in ln.split()) for ln in lines]
    return text + "\n\n" + "\n".join(noisy)


def _repeat_header(text, rng):
    lines = text.split("\n")
    structural = [l for l in lines if l.lstrip().startswith("#") or _TABLE_SEP_RE.match(l.strip())]
    if not structural:
        return text
    cand = rng.choice(structural)
    pos = rng.randint(1, max(1, len(lines) - 1))
    lines.insert(pos, cand)
    return "\n".join(lines)


def _reorder_blocks(text, rng):
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < 3:
        return text
    i, j = rng.sample(range(len(blocks)), 2)
    blocks[i], blocks[j] = blocks[j], blocks[i]
    return "\n\n".join(blocks)


TRANSFORMS = [_corrupt_dates, _corrupt_amounts, _add_boilerplate, _repeat_header, _reorder_blocks]


def apply_corruptions(clean_text, clean_json, rng):
    """Apply a random subset of label-preserving transforms. Deterministic given rng."""
    currency = clean_json["currency"]
    text = clean_text
    if rng.random() < 0.9:
        text = _corrupt_dates(text, rng)
    if rng.random() < 0.9:
        text = _corrupt_amounts(text, rng, currency)
    if rng.random() < 0.7:
        text = _add_boilerplate(text, rng)
    if rng.random() < 0.5:
        text = _repeat_header(text, rng)
    if rng.random() < 0.4:
        text = _reorder_blocks(text, rng)
    return text


def corrupt_file(in_path, out_path, seed):
    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(in_path.open(encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rng = random.Random(seed + i)
            dirty = apply_corruptions(rec["clean_text"], rec["clean_json"], rng)
            out = {"id": rec["id"], "dirty_text": dirty, "clean_json": rec["clean_json"],
                   "template": rec.get("template")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description="Corrupt clean invoices -> dirty.jsonl")
    p.add_argument("--in", dest="inp", type=Path, default=config.CLEAN_JSONL)
    p.add_argument("--out", type=Path, default=config.DIRTY_JSONL)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)
    n = corrupt_file(args.inp, args.out, args.seed)
    print(f"wrote {n} dirty docs -> {args.out}")


if __name__ == "__main__":
    main()
```

#### 3. tests/test_corrupt_invariant.py
**File**: tests/test_corrupt_invariant.py
**Changes**: NEW — pytest asserting (a) normalizer round-trips, (b) each corruption type preserves value-token multisets (parametrized over `corrupt.TRANSFORMS`), (c) combined random combos preserve values over records × seeds via `is_label_preserving`, (d) label-breaking is detected (value LOSS by dropping a date; value INJECTION by appending a spurious amount), (e) `dirty.jsonl` carries the gold intact.

```python
"""Invariant: corruption preserves the gold's value tokens (dates + amounts)."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime

import pytest

from doc_extract import canon, corrupt, generate

_DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]{3} \d{4}|\d{1,2}/\d{1,2}/\d{4}"
    r"|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2}-[A-Za-z]{3}-\d{4}"
)
_MONEY_RE = re.compile(r"(?:C\$|A\$|[¥$€£₹]|[A-Z]{3})\s?[\d,]+\.\d{2}")


def _date_multiset(text):
    return sorted(canon.normalize_date(d) for d in _DATE_RE.findall(text))


def _amount_multiset(text):
    return sorted(canon.normalize_amount(a) for a in _MONEY_RE.findall(text))


@pytest.fixture(scope="module")
def clean_records(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "clean.jsonl"
    generate.generate(15, 42, p)
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def test_normalizer_roundtrips():
    iso = "2026-06-25"
    base = datetime.strptime(iso, "%Y-%m-%d")
    for fmt in ("%d %b %Y", "%d/%m/%Y", "%B %d, %Y", "%d-%b-%Y"):
        assert canon.normalize_date(base.strftime(fmt)) == iso
    assert canon.normalize_amount("$1,234.56") == "1234.56"
    assert canon.normalize_amount("USD 1234.56") == "1234.56"
    assert canon.normalize_amount("C$1,000.00") == "1000.00"
    assert canon.normalize_quantity("2.00") == "2"
    assert canon.normalize_unit("each") == "EA"


@pytest.mark.parametrize("transform", corrupt.TRANSFORMS)
def test_per_corruption_type_preserves_values(clean_records, transform):
    for rec in clean_records:
        rng = random.Random(hash((rec["id"], transform.__name__)) % (2**31))
        if transform is corrupt._corrupt_amounts:
            dirty = transform(rec["clean_text"], rng, rec["clean_json"]["currency"])
        else:
            dirty = transform(rec["clean_text"], rng)
        assert _date_multiset(dirty) == _date_multiset(rec["clean_text"]), \
            f"{transform.__name__} changed date multiset: id={rec['id']}"
        assert _amount_multiset(dirty) == _amount_multiset(rec["clean_text"]), \
            f"{transform.__name__} changed amount multiset: id={rec['id']}"


def test_value_multiset_preserved(clean_records):
    for rec in clean_records:
        for seed in range(6):
            dirty = corrupt.apply_corruptions(rec["clean_text"], rec["clean_json"],
                                              random.Random(seed))
            assert corrupt.is_label_preserving(rec["clean_text"], rec["clean_json"], dirty), \
                f"value multiset changed: id={rec['id']} seed={seed}"


def test_label_breaking_detected(clean_records):
    """Value LOSS and value INJECTION must fail is_label_preserving (the invariant is meaningful)."""
    rec = clean_records[0]
    clean = rec["clean_text"]
    a_date = _DATE_RE.search(clean)
    assert a_date is not None
    dropped = clean.replace(a_date.group(0), "", 1)
    assert not corrupt.is_label_preserving(clean, rec["clean_json"], dropped)
    injected = clean + "\n$9.99"
    assert not corrupt.is_label_preserving(clean, rec["clean_json"], injected)


def test_dirty_jsonl_carries_gold(clean_records, tmp_path):
    clean_path = tmp_path / "clean.jsonl"
    dirty_path = tmp_path / "dirty.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r) + "\n")
    n = corrupt.corrupt_file(clean_path, dirty_path, 42)
    assert n == len(clean_records)
    rows = [json.loads(l) for l in dirty_path.open(encoding="utf-8")]
    assert len(rows) == len(clean_records)
    for r in rows:
        assert r["dirty_text"]
        assert r["clean_json"]
```

### Success Criteria:

#### Automated Verification:
- [x] Tests pass: `pytest tests/test_corrupt_invariant.py -q` is green (normalizer round-trips; per-corruption-type value-multiset preservation ×5 transforms; combined preservation ×6 seeds; label-breaking detection via loss+injection; dirty.jsonl carries gold).
- [x] CLI runs end-to-end self-contained: `python -m doc_extract.generate --n-docs 20 --seed 42 --out /tmp/c_p3.jsonl && python -m doc_extract.corrupt --in /tmp/c_p3.jsonl --out /tmp/d_p3.jsonl --seed 42` writes 20 dirty lines.
- [x] Invariant holds on real output: for /tmp/d_p3.jsonl joined to /tmp/c_p3.jsonl by id, every record passes `corrupt.is_label_preserving(clean_text, clean_json, dirty_text)`.
- [x] Normalizers invert: `canon.normalize_date` parses the 4 messy formats to ISO; `canon.normalize_amount` strips symbol/code + thousands sep to `dddd.dd`.
- [x] No value-drop/swap: dirty.jsonl records contain a `clean_json` identical to the matching clean.jsonl record's; no line-item permutation.

#### Manual Verification:
- [x] Spot-check 3 dirty docs (incl. a wholesale one): dates/amounts reformatted (not removed/changed), boilerplate added, separator/header repeated at most, no vendor/total/line-item values altered.

## Phase 4: Teacher labeling (DeepSeek)
### Overview
Teacher label factory: DeepSeek V4 Flash turns each dirty invoice into clean JSON with retry /
one-repair / quarantine and resumable append-only batching. Depends on Phase 1 (schema/config);
consumes `dirty.jsonl`, produces `labeled.jsonl` + `quarantine.jsonl`.

### Changes Required:

#### 1. src/doc_extract/teacher_labeler.py
**File**: src/doc_extract/teacher_labeler.py
**Changes**: NEW — OpenAI-SDK client (`base_url="https://api.deepseek.com"`, no `/v1`), `deepseek-v4-flash` JSON-mode call (`response_format={"type":"json_object"}`, `extra_body={"thinking":{"type":"disabled"}}`, `temperature=0`, `stream=False`), tenacity transport-retry, `jsonschema` Draft2020-12 validation, one repair-then-quarantine on second semantic failure, resumable append-only JSONL batch with skip-seen-ids + counts. Transport failures left UNSEEN (re-run retries); semantic failures marked seen (quarantined). argparse `main()` with `--in/--out/--quarantine/--model/--seed/--max-tokens` (`--seed` is a no-op at `temperature=0`, kept for a consistent CLI surface).

```python
"""Phase 4: DeepSeek V4 Flash teacher label factory.

Turns each dirty invoice text into a clean JSON extraction target, with retry/repair/quarantine
and resumable append-only batching. Two failure classes:
  - TRANSPORT (rate-limit/connection/timeout/5xx) -> tenacity exponential backoff retry
  - SEMANTIC (malformed JSON, schema fail, empty content, non-stop finish) -> ONE repair, then quarantine
Invalids never enter the training set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from doc_extract import config
from doc_extract.schema import INVOICE_JSON_SCHEMA

logger = logging.getLogger(__name__)


class RetryableTransportError(RuntimeError):
    pass


class SemanticExtractionError(RuntimeError):
    pass


def _make_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY env var is not set")
    return OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_seen_ids(*paths: Path) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "id" in rec:
                    seen.add(str(rec["id"]))
    return seen


def _schema_example(schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:
    """Build an example JSON payload from the schema, resolving $ref against $defs and picking the
    non-null branch of anyOf (Pydantic emits both for Optional fields and nested models)."""
    root = root if root is not None else schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#").lstrip("/").split("/"):
            node = node[part]
        return _schema_example(node, root)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            if branch.get("type") != "null":
                return _schema_example(branch, root)
        return None
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "object":
        return {k: _schema_example(v, root) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_schema_example(schema.get("items", {}), root)]
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    if t == "null":
        return None
    return ""


def build_messages(dirty_text: str, *, repair_error: str | None = None,
                   bad_output: str | None = None) -> list[dict[str, str]]:
    schema_text = json.dumps(INVOICE_JSON_SCHEMA, ensure_ascii=False, indent=2)
    example_text = json.dumps(_schema_example(INVOICE_JSON_SCHEMA), ensure_ascii=False, indent=2)
    system = (
        "You are an invoice extraction engine. Return only valid json that matches the schema. "
        "Do not add keys not in the schema. Use null only where the schema allows. "
        "No markdown, no code fences, no commentary."
    )
    user_parts = ["invoice text:", dirty_text, "", "json schema:", schema_text, "",
                  "example json shape:", example_text]
    if bad_output is not None:
        user_parts += ["", "previous model output:", bad_output]
    if repair_error is not None:
        user_parts += ["", "validation error to fix:", repair_error]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(user_parts)}]


def _make_validator() -> Draft202012Validator:
    Draft202012Validator.check_schema(INVOICE_JSON_SCHEMA)
    return Draft202012Validator(INVOICE_JSON_SCHEMA, format_checker=FormatChecker())


def _validate_payload(raw_json: str, validator: Draft202012Validator) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise SemanticExtractionError(f"Malformed JSON: {exc.msg} at char {exc.pos}") from exc
    try:
        validator.validate(payload)
    except ValidationError as exc:
        raise SemanticExtractionError(f"jsonschema validation failed: {exc.message}") from exc
    if not isinstance(payload, dict):
        raise SemanticExtractionError(f"Top-level JSON must be an object, got {type(payload).__name__}")
    return payload


def _call_teacher(client: OpenAI, messages: list[dict[str, str]], *, model: str,
                  max_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=config.TEACHER_TEMPERATURE,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
    except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise RetryableTransportError(str(exc)) from exc
    except openai.APIStatusError as exc:
        if getattr(exc, "status_code", None) in {429, 500, 503}:
            raise RetryableTransportError(str(exc)) from exc
        raise
    if not resp.choices:
        raise SemanticExtractionError("No choices returned on HTTP 200")
    choice = resp.choices[0]
    if choice.finish_reason != "stop":
        raise SemanticExtractionError(f"finish_reason={choice.finish_reason}")
    content = choice.message.content or ""
    if not content.strip():
        raise SemanticExtractionError("Empty assistant content on HTTP 200")
    return content


_call_teacher_retry = retry(
    retry=retry_if_exception_type(RetryableTransportError),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)(_call_teacher)


def extract_invoice_json(*, client: OpenAI, doc_id: str, dirty_text: str,
                         quarantine_path: Path, model: str, max_tokens: int) -> dict[str, Any]:
    validator = _make_validator()
    raw = ""
    try:
        raw = _call_teacher_retry(client, build_messages(dirty_text), model=model, max_tokens=max_tokens)
        return _validate_payload(raw, validator)
    except SemanticExtractionError as first_err:
        raw2 = ""
        try:
            raw2 = _call_teacher_retry(
                client,
                build_messages(dirty_text, repair_error=str(first_err), bad_output=raw),
                model=model, max_tokens=max_tokens,
            )
            return _validate_payload(raw2, validator)
        except SemanticExtractionError as second_err:
            _append_jsonl(quarantine_path, {
                "id": doc_id, "model": model, "input_text": dirty_text,
                "first_error": str(first_err), "first_output": raw,
                "second_error": str(second_err), "second_output": raw2,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            raise


def label_batch(*, client: OpenAI, in_path: Path, out_path: Path, quarantine_path: Path,
                model: str, max_tokens: int) -> dict[str, int]:
    seen = _load_seen_ids(out_path, quarantine_path)
    processed = labeled = quarantined = transport_failed = skipped = 0
    with Path(in_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            processed += 1
            rec = json.loads(line)
            doc_id = str(rec["id"])
            if doc_id in seen:
                skipped += 1
                continue
            dirty_text = rec["dirty_text"]
            try:
                payload = extract_invoice_json(
                    client=client, doc_id=doc_id, dirty_text=dirty_text,
                    quarantine_path=quarantine_path, model=model, max_tokens=max_tokens,
                )
                _append_jsonl(out_path, {
                    "id": doc_id, "model": model, "input_text": dirty_text, "output": payload,
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                seen.add(doc_id)
                labeled += 1
            except SemanticExtractionError:
                seen.add(doc_id)          # quarantined -> don't retry on re-run
                quarantined += 1
            except RetryableTransportError:
                transport_failed += 1     # transient -> leave UNSEEN so a re-run retries it
    counts = {"processed": processed, "labeled": labeled, "quarantined": quarantined,
              "transport_failed": transport_failed, "skipped": skipped}
    logger.info("label_batch %s", counts)
    return counts


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Teacher-label dirty invoices -> labeled.jsonl")
    p.add_argument("--in", dest="inp", type=Path, default=config.DIRTY_JSONL)
    p.add_argument("--out", type=Path, default=config.LABELED_JSONL)
    p.add_argument("--quarantine", type=Path, default=config.QUARANTINE_JSONL)
    p.add_argument("--model", default=config.TEACHER_MODEL_ID)
    p.add_argument("--seed", type=int, default=config.SEED)  # no-op at temperature=0; kept for CLI consistency
    p.add_argument("--max-tokens", type=int, default=config.TEACHER_MAX_TOKENS)
    args = p.parse_args(argv)
    config.ensure_dirs()
    client = _make_client()
    counts = label_batch(client=client, in_path=args.inp, out_path=args.out,
                         quarantine_path=args.quarantine, model=args.model,
                         max_tokens=args.max_tokens)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [x] Imports + client wiring: `python -c "from doc_extract import teacher_labeler as t; assert t.build_messages('x')[1]['content'].count('json')>=1; from doc_extract import config as c; assert c.DEEPSEEK_BASE_URL=='https://api.deepseek.com'; print('ok')"` prints ok.
- [x] Validator builds against the locked schema: `_make_validator()` accepts a valid invoice dict; an invalid payload (bad date) raises `SemanticExtractionError` (wrapping the schema `ValidationError`) via `_validate_payload`.
- [x] Semantic-vs-transport classification: `_validate_payload('{not json', _make_validator())` raises `SemanticExtractionError`; the two exception classes are distinct and handled differently in `label_batch`.
- [x] Resumability: `_load_seen_ids` dedups ids across two tmp JSONL files (offline check).
- [x] No live API call in automated criteria (live labeling is a MANUAL check below).

#### Manual Verification:
- [ ] Live smoke (needs DEEPSEEK_API_KEY + a few dirty docs): `python -m doc_extract.teacher_labeler --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl` labels docs; quarantine rate is small (DeepSeek Flash ~0.27% structured-output error per research); transport failures leave docs unseen for re-run.
- [ ] Inspect a few labeled.jsonl `output` payloads — they validate against `Invoice` and recover the gold values (teacher's clean JSON ≈ the clean_json gold).

## Phase 5: Dataset preparation
### Overview
Turn teacher labels into SFT-ready data: filter invalids/empty, deterministic seeded train/test
split, format to prompt/completion, persist a frozen snapshot. Depends on Phases 1, 4.

### Changes Required:

#### 1. src/doc_extract/prepare.py
**File**: src/doc_extract/prepare.py
**Changes**: NEW — load `labeled.jsonl`, re-validate each teacher `output` against `Invoice` + filter invalids (count `n_filtered`), deterministic `random.Random(seed).shuffle` split (stable id-sort; config.TRAIN_SPLIT/DATA_SEED), build `{id, prompt, completion}` records (prompt = extraction instruction + dirty `input_text`; completion = strict JSON via `json.dumps`, no fences), persist JSONL + HF `datasets` `save_to_disk` snapshot + a split manifest (seed, split, counts, schema_version), argparse `main()`.

```python
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
from doc_extract.schema import SCHEMA_VERSION, Invoice

_INSTRUCTION = (
    "Extract the invoice fields from the document below and return ONLY a JSON object matching "
    "the invoice schema. Use null for fields not present. Do not include markdown or commentary."
)


def _load_labeled(in_path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(in_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _filter_valid(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid, n_filtered = [], 0
    for r in rows:
        out = r.get("output")
        try:
            Invoice(**out)  # re-validate (schema carries pattern/enum)
        except Exception:
            n_filtered += 1
            continue
        valid.append(r)
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


def prepare(in_path: Path, out_dir: Path, seed: int, split: float) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_labeled(in_path)
    valid, n_filtered = _filter_valid(rows)
    train_rows, test_rows = _split(valid, seed, split)
    train_sft, test_sft = _to_sft(train_rows), _to_sft(test_rows)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    for p, data in ((train_path, train_sft), (test_path, test_sft)):
        with p.open("w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    Dataset.from_list(train_sft).save_to_disk(str(out_dir / "train_hf"))
    Dataset.from_list(test_sft).save_to_disk(str(out_dir / "test_hf"))

    manifest = {
        "seed": seed, "train_split": split, "n_train": len(train_sft), "n_test": len(test_sft),
        "n_filtered": n_filtered, "n_loaded": len(rows), "schema_version": SCHEMA_VERSION,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
```

#### 2. tests/test_prepare_split.py
**File**: tests/test_prepare_split.py
**Changes**: NEW — pytest for invalid filtering (`n_filtered`), train/test disjointness + invalid exclusion, same-seed split reproducibility (identical train/test id sets across two runs), and strict-JSON completions (parses + no markdown fence).

```python
"""Phase 5: deterministic split reproducibility + invalid filtering."""

from __future__ import annotations

import json

from doc_extract import prepare

_VALID = {
    "vendor_name": "Acme", "invoice_number": "INV-1", "invoice_date": "2026-06-25",
    "currency": "USD", "grand_total": "100.00",
    "line_items": [{"description": "w", "quantity": "1", "unit_price": "100.00", "amount": "100.00"}],
}


def _row(i, output):
    return {"id": f"doc-{i:03d}", "model": "deepseek-v4-flash", "input_text": f"text {i}", "output": output}


def _write_labeled(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _ids(path):
    return sorted(json.loads(l)["id"] for l in path.open(encoding="utf-8"))


def test_filter_and_split(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    rows = [_row(i, dict(_VALID, invoice_number=f"INV-{i}")) for i in range(10)]
    rows.append(_row(99, dict(_VALID, invoice_date="25/06/2026")))   # invalid -> filtered
    rows.append(_row(98, dict(_VALID, currency="XYZ")))              # invalid -> filtered
    _write_labeled(labeled, rows)

    m = prepare.prepare(labeled, tmp_path / "out1", seed=42, split=0.8)
    assert m["n_filtered"] == 2
    assert m["n_train"] + m["n_test"] == 10
    train1, test1 = tmp_path / "out1" / "train.jsonl", tmp_path / "out1" / "test.jsonl"
    train_ids, test_ids = _ids(train1), _ids(test1)
    assert set(train_ids).isdisjoint(set(test_ids))     # held out
    assert "doc-099" not in train_ids and "doc-099" not in test_ids  # invalid excluded


def test_split_reproducible(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    rows = [_row(i, dict(_VALID, invoice_number=f"INV-{i}")) for i in range(10)]
    _write_labeled(labeled, rows)

    prepare.prepare(labeled, tmp_path / "a", seed=42, split=0.8)
    prepare.prepare(labeled, tmp_path / "b", seed=42, split=0.8)
    assert _ids(tmp_path / "a" / "train.jsonl") == _ids(tmp_path / "b" / "train.jsonl")
    assert _ids(tmp_path / "a" / "test.jsonl") == _ids(tmp_path / "b" / "test.jsonl")


def test_completions_are_strict_json(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    _write_labeled(labeled, [_row(0, _VALID)])
    prepare.prepare(labeled, tmp_path / "o", seed=42, split=1.0)
    rec = json.loads(next((tmp_path / "o" / "train.jsonl").open(encoding="utf-8")))
    assert set(rec) >= {"id", "prompt", "completion"}
    parsed = json.loads(rec["completion"])  # strict JSON, parses
    assert parsed["vendor_name"] == "Acme"
    assert not rec["completion"].lstrip().startswith("```")  # no markdown fence
```

### Success Criteria:

#### Automated Verification:
- [x] Tests pass: `pytest tests/test_prepare_split.py -q` is green (filtering + disjoint split + reproducibility + strict-JSON completions).
- [x] CLI runs self-contained: 10 valid + 2 invalid rows -> `python -m doc_extract.prepare --in <labeled> --out-dir <o> --seed 42 --split 0.8` -> n_filtered==2, n_train+n_test==10.
- [x] Determinism: re-running --seed 42 yields identical train/test id sets (now present in output records).
- [x] completions are strict JSON: every record's completion json.loads and has no leading ` ``` `.
- [x] Split manifest records seed/split/counts/schema_version.

#### Manual Verification:
- [x] Spot-check a train record: prompt = instruction + dirty doc; completion = teacher's clean JSON; test set disjoint from train.

## Phase 6: Training (QLoRA SFT)
### Overview
4-bit NF4 QLoRA SFT of `LFM2.5-VL-1.6B-Extract` via TRL `SFTTrainer`+PEFT on prompt/completion
data, saving the adapter and a merged export. Depends on Phase 5.

### Changes Required:

#### 1. src/doc_extract/train.py
**File**: src/doc_extract/train.py
**Changes**: NEW — `_load_base_4bit` via `AutoModelForImageTextToText`+`AutoProcessor` 4-bit NF4 (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`, `device_map="auto"`, `trust_remote_code=True`); `set_seed(seed, deterministic=True)`; `model.config.use_cache=False`; `prepare_model_for_kbit_training`; **explicit** `LoraConfig(target_modules="all-linear", r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")` + `get_peft_model` (visible mechanics; no `peft_config=` double-wrap); prompt/completion dataset; `SFTConfig(completion_only_loss=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False}, bf16=True, optim="paged_adamw_8bit", per_device_train_batch_size=1, gradient_accumulation_steps=8, max_length, seed, data_seed, report_to="none")`; `SFTTrainer(model=..., processing_class=processor, args, train_dataset).train()`; save adapter+processor; `merge()` on a fresh full-precision base via `PeftModel.from_pretrained`+`merge_and_unload`. argparse `main()`.

```python
"""Phase 6: QLoRA SFT of Liquid LFM2.5-VL-1.6B-Extract on prompt/completion invoice data.

TEXT-ONLY training (the -VL image path is unused this phase). Loads the base in 4-bit NF4, attaches
LoRA explicitly (prepare_model_for_kbit_training -> get_peft_model), and SFT-trains only the
completion loss (the chat template lacks {% generation %} markers, so assistant_only_loss is not
usable). Saves the adapter + a merged full-precision export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from doc_extract import config


def _load_base_4bit(model_id: str, revision: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, device_map="auto", dtype=torch.bfloat16,
        quantization_config=bnb, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    return model, processor


def train(
    *, train_file: Path, base_id: str, revision: str, adapter_dir: Path,
    epochs: int, seed: int, max_length: int,
) -> str:
    set_seed(seed, deterministic=True)
    model, processor = _load_base_4bit(base_id, revision)
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # Explicit LoRA attachment (pedagogically transparent; verified-safe default for the
    # hybrid conv/GQA backbone whose exact module names aren't published).
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)

    ds = load_dataset("json", data_files=str(train_file), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])

    args = SFTConfig(
        output_dir=str(adapter_dir) + "-runs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=seed,
        data_seed=seed,
        max_length=max_length,
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,           # already PEFT-wrapped above; do NOT also pass peft_config=
        processing_class=processor,
        args=args,
        train_dataset=ds,
    )
    trainer.train()

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    return str(adapter_dir)


def merge(adapter_dir: Path, base_id: str, revision: str, merged_dir: Path) -> str:
    # Merge on a FULL-PRECISION base (4-bit merge is unreliable per research).
    base = AutoModelForImageTextToText.from_pretrained(
        base_id, revision=revision, device_map="auto", dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(base_id, revision=revision, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    return str(merged_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="QLoRA SFT train LFM2.5-VL-1.6B-Extract")
    p.add_argument("--train-file", type=Path, default=config.SFT_DIR / "train.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--revision", default=config.STUDENT_REVISION)
    p.add_argument("--adapter-dir", type=Path, default=config.ADAPTER_DIR)
    p.add_argument("--merged-dir", type=Path, default=config.MERGED_DIR)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--skip-merge", action="store_true")
    args = p.parse_args(argv)
    config.ensure_dirs()
    adapter = train(
        train_file=args.train_file, base_id=args.base, revision=args.revision,
        adapter_dir=args.adapter_dir, epochs=args.epochs, seed=args.seed,
        max_length=args.max_length,
    )
    print(f"adapter -> {adapter}")
    if not args.skip_merge:
        merged = merge(args.adapter_dir, args.base, args.revision, args.merged_dir)
        print(f"merged  -> {merged}")


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [x] Imports resolve offline: `python -c "import torch;from trl import SFTConfig, SFTTrainer;from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training;from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed;print('ok')"` prints ok (no download).
- [x] Config wiring: `python -c "from doc_extract import config as c;assert c.STUDENT_MODEL_ID=='LiquidAI/LFM2.5-VL-1.6B-Extract';assert c.ADAPTER_DIR and c.MERGED_DIR;print('ok')"` prints ok.
- [x] Recipe knobs present: `python -c "import inspect,doc_extract.train as t;src=inspect.getsource(t);assert 'set_seed' in src and 'data_seed' in src and 'use_cache = False' in src and 'use_reentrant' in src and 'get_peft_model' in src and 'completion_only_loss=True' in src;print('ok')"` prints ok.
- [x] No live model download in automated criteria (training/merge are MANUAL below).

#### Manual Verification:
- [ ] Train runs without OOM on 24GB: ensure the earlier data stages have produced `data/sft/train.jsonl`, then `python -m doc_extract.train` completes its epochs and writes `artifacts/checkpoints/adapter` (adapter_config.json + adapter weights).
- [ ] `--skip-merge` skips the merge step; otherwise `merge` writes `artifacts/checkpoints/merged`, loadable later by an eval stage.
- [ ] Training loss decreases across steps (visible in logs); use_cache=False avoids the gradient-checkpointing runtime error.

## Phase 7: Evaluation harness
### Overview
3-layer field-level evaluation: parse → schema → canonicalized-leaf micro-F1 (reusing `canon.py`
from Phase 3), Hungarian line-item match, `record_exact`, run on base AND fine-tuned, paired
bootstrap CI to prove learning. Depends on Phases 1, 3, 6.

### Changes Required:

#### 1. src/doc_extract/evaluate.py
**File**: src/doc_extract/evaluate.py
**Changes**: NEW — `compute_metrics(predictions, golds)` with a 3-layer hard gate: parse (`json.loads(strip_fences)`) → schema (`Draft202012Validator(INVOICE_JSON_SCHEMA)`) → canonicalized-leaf scoring; parse/schema failure forces ALL non-null gold leaves (scalars + line items) to FN (zeros downstream). Path-flatten via `_flatten`; canonicalize via reused Phase-3 `canon.normalize_value`; scalar wrong-value = FN+FP (per-leaf exact-match contract); line items matched with `scipy.optimize.linear_sum_assignment` (path-keyed leaf alignment) → micro-F1 + macro-F1 (by normalized path) + `record_exact`. `per_record_counts`/`per_record_exact` + `paired_bootstrap_ci_corpus_f1` (seeded) on BOTH micro_f1 (corpus-resampled, aligns with `delta_micro_f1`) and record_exact deltas. `load_model(spec)` top-level (HF id OR local dir); greedy `generate` (`do_sample=False`); each model loaded/generated once; `main()` runs base vs fine-tuned, writes `metrics.json`, `learning_proven = ci_f1[0] > 0`. (Added `scipy>=1.13,<2` to Phase 1 deps.)

```python
"""Phase 7: field-level evaluation harness — 3-layer gate + paired bootstrap.

Layers: parse (json.loads) -> schema (INVOICE_JSON_SCHEMA) -> canonicalized-leaf micro-F1 (canon).
Line items matched optimally (scipy linear_sum_assignment) with path-aware per-leaf scoring.
Runs base vs fine-tuned; paired bootstrap CI on micro_f1 AND record_exact proves learning.
Reuses Phase 3 canon.py (single source of truth for normalization).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from scipy.optimize import linear_sum_assignment

from doc_extract import canon, config
from doc_extract.schema import FIELD_TYPE_REGISTRY, INVOICE_JSON_SCHEMA

_VALIDATOR = Draft202012Validator(INVOICE_JSON_SCHEMA)
_FENCE_RE = re.compile(r"^\s*```(?:json)?|```\s*$", re.MULTILINE)
_MISMATCH = object()


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _flatten(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _flatten(v, f"{path}[{i}]")
    else:
        out.append((path, obj))
    return out


def _leaf_kind(path):
    return FIELD_TYPE_REGISTRY.get(re.sub(r"\[\d+\]", "[]", path), "string")


def _field_key(path):
    return re.sub(r"\[\d+\]", "[]", path)


def _canon_leaves(leaves, currency):
    norm = []
    for p, v in leaves:
        try:
            norm.append((p, canon.normalize_value(v, _leaf_kind(p), currency)))
        except Exception:
            norm.append((p, _MISMATCH))
    return norm


def _score_line_items(pred_items, gold_items, currency):
    pl = [dict(_canon_leaves(_flatten(it, "line_items[]"), currency)) for it in (pred_items or [])]
    gl = [dict(_canon_leaves(_flatten(it, "line_items[]"), currency)) for it in (gold_items or [])]
    tp = fp = fn = 0
    per = {}

    def bump(key, dt, dp, dn):
        t, f, n = per.get(key, (0, 0, 0)); per[key] = (t + dt, f + dp, n + dn)

    def pair_cost(a, b):
        keys = set(a) | set(b)
        mism = 0
        for k in keys:
            if k in a and k in b:
                if a[k] != b[k]:
                    mism += 1
            else:
                mism += 1
        return mism

    if pl and gl:
        cost = np.array([[pair_cost(pa, gb) for gb in gl] for pa in pl])
        rows, cols = linear_sum_assignment(cost)
        matched_a, matched_b = set(rows.tolist()), set(cols.tolist())
        for a, b in zip(rows, cols):
            pa, gb = pl[a], gl[b]
            for k in set(pa) | set(gb):
                fk = _field_key(k)
                if _leaf_kind(k).endswith("nullable") and gb.get(k) is None:
                    # nullable-null gold = non-target; only penalize a spurious non-null pred
                    if k in pa and pa[k] is not None and pa[k] is not _MISMATCH:
                        fp += 1; bump(fk, 0, 1, 0)
                    continue
                if k in pa and k in gb:
                    if pa[k] == gb[k]:
                        tp += 1; bump(fk, 1, 0, 0)
                    else:
                        fn += 1; fp += 1; bump(fk, 0, 0, 1); bump(fk, 0, 1, 0)
                elif k in gb:
                    fn += 1; bump(fk, 0, 0, 1)
                else:
                    if pa[k] is _MISMATCH:
                        continue
                    fp += 1; bump(fk, 0, 1, 0)
        for b in range(len(gl)):
            if b not in matched_b:
                for k, v in gl[b].items():
                    if _leaf_kind(k).endswith("nullable") and v is None:
                        continue
                    fn += 1; bump(_field_key(k), 0, 0, 1)
        for a in range(len(pl)):
            if a not in matched_a:
                for k, v in pl[a].items():
                    if v is _MISMATCH:
                        continue
                    fp += 1; bump(_field_key(k), 0, 1, 0)
    else:
        for gb in gl:
            for k, v in gb.items():
                if _leaf_kind(k).endswith("nullable") and v is None:
                    continue
                fn += 1; bump(_field_key(k), 0, 0, 1)
        for pa in pl:
            for k, v in pa.items():
                if v is _MISMATCH:
                    continue
                fp += 1; bump(_field_key(k), 0, 1, 0)
    return tp, fp, fn, per


def _score_record(pred, gold):
    cur = gold.get("currency")
    gmap = dict(_canon_leaves(_flatten(gold, ""), cur))
    pmap = dict(_canon_leaves(_flatten(pred, ""), cur))
    tp = fp = fn = 0
    per = {}

    def bump(key, dt, dp, dn):
        t, f, n = per.get(key, (0, 0, 0)); per[key] = (t + dt, f + dp, n + dn)

    for p, gv in gmap.items():
        if p.startswith("line_items"):
            continue
        kind = _leaf_kind(p); key = _field_key(p)
        pv = pmap.get(p)
        if kind.endswith("nullable") and gv is None:
            # nullable-null gold = non-target; only penalize a spurious non-null prediction
            if pv is not None and pv is not _MISMATCH:
                fp += 1; bump(key, 0, 1, 0)
            continue
        if pv == gv:
            tp += 1; bump(key, 1, 0, 0)
        else:
            fn += 1; bump(key, 0, 0, 1)
            if pv is not None and pv is not _MISMATCH:
                fp += 1; bump(key, 0, 1, 0)
    for p, pv in pmap.items():
        if p.startswith("line_items"):
            continue
        if p not in gmap and pv is not None and pv is not _MISMATCH:
            fp += 1; bump(_field_key(p), 0, 1, 0)
    ltp, lfp, lfn, lper = _score_line_items(pred.get("line_items", []), gold.get("line_items", []), cur)
    for k, (a, b, c) in lper.items():
        bump(k, a, b, c)
    return tp + ltp, fp + lfp, fn + lfn, per


def compute_metrics(predictions, golds):
    n = len(golds)
    parsed = schema_ok = 0
    tp = fp = fn = 0
    perfect = 0
    per = {}

    def bump(key, dt, dp, dn):
        t, f, nn = per.get(key, (0, 0, 0)); per[key] = (t + dt, f + dp, nn + dn)

    for raw, gold in zip(predictions, golds):
        ok = True
        pred = None
        try:
            pred = json.loads(strip_fences(raw)); parsed += 1
        except Exception:
            ok = False
        if ok:
            try:
                _VALIDATOR.validate(pred); schema_ok += 1
            except Exception:
                ok = False
        if not ok:
            # HARD GATE: ALL gold non-null leaves (scalars + line items) -> FN, path-bucketed.
            for p, v in _canon_leaves(_flatten(gold, ""), gold.get("currency")):
                if _leaf_kind(p).endswith("nullable") and v is None:
                    continue
                fn += 1; bump(_field_key(p), 0, 0, 1)
            continue
        rtp, rfp, rfn, rper = _score_record(pred, gold)
        tp += rtp; fp += rfp; fn += rfn
        for k, (a, b, c) in rper.items():
            bump(k, a, b, c)
        if rfp == 0 and rfn == 0:
            perfect += 1
    denom = 2 * tp + fp + fn
    micro = (2 * tp) / denom if denom else 1.0
    f1s = []
    for a, b, c in per.values():
        d = 2 * a + b + c
        f1s.append((2 * a) / d if d else 1.0)
    macro = float(np.mean(f1s)) if f1s else 1.0
    return {"parse_rate": parsed / n if n else 0.0, "schema_pass_rate": schema_ok / n if n else 0.0,
            "micro_f1": micro, "macro_f1": macro, "record_exact": perfect / n if n else 0.0, "n": n}


def _gold_nonnull_leaf_count(gold):
    return sum(1 for p, v in _canon_leaves(_flatten(gold, ""), gold.get("currency"))
               if not (_leaf_kind(p).endswith("nullable") and v is None))


def per_record_counts(predictions, golds):
    """Per-record (tp, fp, fn) for corpus-micro-F1 bootstrap; parse/schema fail -> (0,0,all-FN)."""
    out = []
    for raw, gold in zip(predictions, golds):
        ok = True
        pred = None
        try:
            pred = json.loads(strip_fences(raw))
        except Exception:
            ok = False
        if ok:
            try:
                _VALIDATOR.validate(pred)
            except Exception:
                ok = False
        if not ok:
            out.append((0, 0, _gold_nonnull_leaf_count(gold))); continue
        tp, fp, fn, _ = _score_record(pred, gold)
        out.append((tp, fp, fn))
    return out


def per_record_exact(predictions, golds):
    out = []
    for raw, gold in zip(predictions, golds):
        ok = True
        pred = None
        try:
            pred = json.loads(strip_fences(raw))
        except Exception:
            ok = False
        if ok:
            try:
                _VALIDATOR.validate(pred)
            except Exception:
                ok = False
        if not ok:
            out.append(0.0); continue
        tp, fp, fn, _ = _score_record(pred, gold)
        out.append(1.0 if (fp == 0 and fn == 0) else 0.0)
    return out


def _corpus_f1(counts):
    tp = sum(c[0] for c in counts); fp = sum(c[1] for c in counts); fn = sum(c[2] for c in counts)
    d = 2 * tp + fp + fn
    return (2 * tp) / d if d else 1.0


def paired_bootstrap_ci(deltas, n_boot, ci, seed):
    """Bootstrap the mean of per-record deltas (used for record_exact)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return [0.0, 0.0]
    boots = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(boots, (1 - ci) / 2 * 100)),
            float(np.percentile(boots, (1 + ci) / 2 * 100))]


def paired_bootstrap_ci_corpus_f1(base_counts, ft_counts, n_boot, ci, seed):
    """Bootstrap CORPUS micro-F1 delta (resample records, recompute corpus TP/FP/FN) so the CI
    aligns with the headline delta_micro_f1 rather than a mean-of-per-record statistic."""
    rng = np.random.default_rng(seed)
    n = len(base_counts)
    if n == 0:
        return [0.0, 0.0]
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bf = _corpus_f1([base_counts[i] for i in idx])
        ff = _corpus_f1([ft_counts[i] for i in idx])
        deltas.append(ff - bf)
    return [float(np.percentile(deltas, (1 - ci) / 2 * 100)),
            float(np.percentile(deltas, (1 + ci) / 2 * 100))]


def load_model(spec):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(spec, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        spec, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    return model, processor


def generate(model, processor, prompts, max_new_tokens):
    import torch
    outs = []
    for prompt in prompts:
        text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outs.append(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0])
    return outs


def load_records(test_file):
    prompts, golds = [], []
    with Path(test_file).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            golds.append(json.loads(rec["completion"]))
    return prompts, golds


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate base vs fine-tuned extraction")
    p.add_argument("--test-file", type=Path, default=config.SFT_DIR / "test.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--ft", type=Path, default=config.MERGED_DIR)
    p.add_argument("--out", type=Path, default=config.METRICS_PATH)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args(argv)
    config.ensure_dirs()

    prompts, golds = load_records(args.test_file)
    base_mp = load_model(args.base)
    base_preds = generate(base_mp[0], base_mp[1], prompts, args.max_new_tokens)
    ft_mp = load_model(args.ft)
    ft_preds = generate(ft_mp[0], ft_mp[1], prompts, args.max_new_tokens)

    base_metrics = compute_metrics(base_preds, golds)
    ft_metrics = compute_metrics(ft_preds, golds)
    base_counts = per_record_counts(base_preds, golds)
    ft_counts = per_record_counts(ft_preds, golds)
    base_exact = per_record_exact(base_preds, golds)
    ft_exact = per_record_exact(ft_preds, golds)
    ci_f1 = paired_bootstrap_ci_corpus_f1(base_counts, ft_counts,
                                          config.N_BOOTSTRAP, config.BOOTSTRAP_CI, config.SEED)
    ci_re = paired_bootstrap_ci([f - b for f, b in zip(ft_exact, base_exact)],
                                config.N_BOOTSTRAP, config.BOOTSTRAP_CI, config.SEED)
    result = {
        "base": base_metrics, "finetuned": ft_metrics,
        "delta_micro_f1": ft_metrics["micro_f1"] - base_metrics["micro_f1"],
        "delta_record_exact": ft_metrics["record_exact"] - base_metrics["record_exact"],
        "paired_bootstrap_ci_micro_f1": ci_f1,
        "paired_bootstrap_ci_record_exact": ci_re,
        "learning_proven": ci_f1[0] > 0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [x] Pure-logic metrics: `python -c "from doc_extract.evaluate import compute_metrics; import json; g={...}; m=compute_metrics([json.dumps(g)],[g]); assert m['parse_rate']==1.0 and m['record_exact']==1.0 and m['micro_f1']==1.0; print('ok')"` prints ok.
- [x] Format-only diff scores perfect: a prediction with `grand_total:'$100.00'` vs gold `'100.00'` yields micro_f1==1.0 (canon normalizes).
- [x] Hard gate: `compute_metrics(['not json', json.dumps(g)], [g,g])` → parse_rate==0.5 and micro_f1<1.0 (line-item gold leaves counted as FN too).
- [x] scipy resolves (Phase 1 deps now include scipy): `python -c "from doc_extract.evaluate import compute_metrics; print('ok')"` prints ok.
- [x] No live model download in automated criteria (live base-vs-ft is MANUAL below).

#### Manual Verification:
- [ ] Live eval on GPU: `python -m doc_extract.evaluate` (after train) loads base + merged once each, generates over the test split, writes metrics.json; finetuned micro_f1/record_exact ≥ base; learning_proven reflects whether the micro_f1 bootstrap CI excludes 0.

## Phase 8: Orchestration + reproducibility + reflection
### Overview
Glue the stages into a reproducible end-to-end loop and capture the learning reflection.
Depends on all prior phases.

### Changes Required:

#### 1. Makefile
**File**: Makefile
**Changes**: NEW — targets `venv`, `data` (generate→corrupt→teacher_labeler, file-stamped for incremental rebuilds), `prepare`, `train`, `evaluate`, `baseline` (evaluate base model only, no train dep), `all`, `clean`. Each invokes `python -m doc_extract.<stage>` with locked CLI flags; `DEEPSEEK_API_KEY` checked via env, never embedded.

```makefile
.PHONY: venv data prepare train evaluate baseline all clean

PY ?= python
N_DOCS ?= 500
SEED ?= 42

venv:
	$(PY) -m venv .venv
	@echo "Run: source .venv/bin/activate && pip install -e .[dev]"

data: data/clean.jsonl data/dirty.jsonl data/labeled.jsonl

data/clean.jsonl:
	$(PY) -m doc_extract.generate --n-docs $(N_DOCS) --seed $(SEED) --out data/clean.jsonl

data/dirty.jsonl: data/clean.jsonl
	$(PY) -m doc_extract.corrupt --in data/clean.jsonl --out data/dirty.jsonl --seed $(SEED)

data/labeled.jsonl: data/dirty.jsonl
	@test -n "$$DEEPSEEK_API_KEY" || (echo "Set DEEPSEEK_API_KEY"; exit 1)
	$(PY) -m doc_extract.teacher_labeler --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl

prepare: data/labeled.jsonl
	$(PY) -m doc_extract.prepare --in data/labeled.jsonl --out-dir data/sft --seed $(SEED) --split 0.9

train: prepare
	$(PY) -m doc_extract.train

evaluate: train
	$(PY) -m doc_extract.evaluate

baseline: prepare
	$(PY) -m doc_extract.evaluate --ft $(shell $(PY) -c "from doc_extract import config;print(config.STUDENT_MODEL_ID)")

all: data prepare train evaluate

clean:
	rm -rf data/sft artifacts/checkpoints artifacts/metrics.json
```

#### 2. README.md
**File**: README.md
**Changes**: MODIFY — expand the Phase-1 stub (keep `# doc-extract` so pyproject `readme="README.md"` stays install-safe) with full setup (venv, `pip install -e .[dev]`, `export DEEPSEEK_API_KEY`), per-stage run instructions, 6-step architecture overview, data/eval contract, the "process is the product" framing, and a link to `docs/REFLECTION.md`.

```markdown
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
```

#### 3. src/doc_extract/run_all.py
**File**: src/doc_extract/run_all.py
**Changes**: NEW — programmatic end-to-end runner invoking each stage's `main([...])` in order with shared config/seed; mirrors the Makefile; usable via `python -m doc_extract.run_all`.

```python
"""Phase 8: programmatic end-to-end runner — mirrors the Makefile with shared seed/config."""

from __future__ import annotations

from doc_extract import config
from doc_extract import corrupt, evaluate, generate, prepare, teacher_labeler, train


def run_all(n_docs: int = config.DEFAULT_N_DOCS, seed: int = config.SEED) -> None:
    config.ensure_dirs()
    generate.main(["--n-docs", str(n_docs), "--seed", str(seed), "--out", str(config.CLEAN_JSONL)])
    corrupt.main(["--in", str(config.CLEAN_JSONL), "--out", str(config.DIRTY_JSONL), "--seed", str(seed)])
    teacher_labeler.main(["--in", str(config.DIRTY_JSONL), "--out", str(config.LABELED_JSONL),
                          "--quarantine", str(config.QUARANTINE_JSONL)])
    prepare.main(["--in", str(config.LABELED_JSONL), "--out-dir", str(config.SFT_DIR),
                  "--seed", str(seed), "--split", str(config.TRAIN_SPLIT)])
    train.main([])
    evaluate.main([])


if __name__ == "__main__":
    run_all()
```

#### 4. docs/REFLECTION.md
**File**: docs/REFLECTION.md
**Changes**: NEW — per-stage reflection scaffold (what I learned at generate/label/train/evaluate) — the "process is the product" deliverable; acceptance-criterion checklist.

```markdown
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
```

### Success Criteria:

#### Automated Verification:
- [x] Project baseline — install: `pip install -e .[dev]` exits 0.
- [x] Project baseline — tests: `pytest -q` is green (runs test_corrupt_invariant.py + test_prepare_split.py).
- [x] run_all imports resolve: `python -c "from doc_extract import run_all; print('ok')"` prints ok (no execution — run_all calls live APIs/GPU).
- [x] Makefile parses: `make -n all` dry-runs the target chain without error (no live execution).
- [x] README install-safe: H1 `# doc-extract` present (pyproject `readme="README.md"` still valid); no hard-coded DEEPSEEK_API_KEY value.

#### Manual Verification:
- [ ] End-to-end on GPU: `make all` runs generate→corrupt→label→prepare→train→evaluate; `artifacts/metrics.json` shows `learning_proven: true` (or documents why not).
- [ ] `make baseline` evaluates the untrained base model for comparison.

## Ordering Constraints
- **Strictly sequential**: Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each builds on the prior.
- **Foundation first**: Phase 1 (schema/config + installable package) gates everything.
- **`canon.py` cross-phase dependency**: introduced in Phase 3 (generation invariant), reused in
  Phase 7 (eval leaf canonicalization). Phase 7 must import from Phase 3's `canon.py`, not
  re-implement normalization.
- **Data flow**: Phase 2 `clean.jsonl` → Phase 3 `dirty.jsonl` → Phase 4 `labeled.jsonl` →
  Phase 5 SFT snapshot → Phase 6 adapter → Phase 7 `metrics.json`.
- **Base-eval-before-train (recommended, not blocking)**: `evaluate.py` can run on the base model
  immediately after Phase 7 is built to establish a baseline before/without training; the
  "finetuned" path needs the Phase 6 adapter.

## Verification Notes
- **Label-preserving invariant (risk-bearing core):** every corruption type + random combos must
  satisfy `canonicalize(corrupt(x)) == canonicalize(x)`. Verified by `tests/test_corrupt_invariant.py`.
- **VRAM / OOM:** Phase 6 must train without OOM on 24GB with batch=1, grad-checkpointing,
  `max_length` ≤ 1024. If OOM, lower `max_length` / sequence length first.
- **Teacher quarantine rate:** Phase 4 should quarantine only a small fraction (DeepSeek Flash
  ~0.27% structured-output error per research). A high quarantine rate signals a prompt/schema bug.
- **Thinking disabled:** teacher calls must include `extra_body={"thinking":{"type":"disabled"}}`
  or latency/cost balloon and `temperature` is ignored.
- **Deterministic split reproducibility:** re-running Phase 5 with the same seed must produce
  byte-identical split membership (test in Phase 5).
- **Prove learning:** Phase 7's `paired_bootstrap_ci_micro_f1` must exclude 0 to set
  `learning_proven: true`.
- **Python 3.14 compatibility (flag):** the host has Python 3.14.6 (bleeding-edge). Some pinned
  ML libs (bitsandbytes, trl, flash-attn) may not yet ship 3.14 wheels — if install fails, create
  a venv with Python 3.11 or 3.12 and pin deps to versions with 3.12 wheels. Confirm `pip install -e .`
  succeeds before Phase 2.
- **`use_reentrant=False` + `use_cache=False`:** required for gradient-checkpointing + QLoRA; a
  missing `use_cache=False` raises a runtime error during training.
- **Empty-content-on-200:** teacher must treat empty `content` as failure even on HTTP 200.

## Performance Considerations
- **API throughput governs labeling wall-clock:** sequential ~0.5s/call ≈ 4–17 min for 500–2,000
  docs. Resumable, so interruptions are cheap. (Concurrency explicitly deferred.)
- **VRAM:** 24GB is ample for the 1.6B model at 4-bit; headroom allows relaxing `max_length` or
  batch size once the loop runs. Default conservative recipe first.
- **Sequence length:** keep `max_length` ≤ 1024 (invoices are short; VLM packing is immature).
- **Inference eval cost:** greedy generation over the held-out set is the main Phase-7 cost; small
  (hundreds of short generations) — minutes on a 3090.

## Migration Notes
Not applicable — greenfield project with no existing persisted data or schema to migrate.

## Pattern References
- `trl/examples/scripts/sft_vlm.py` (current VLM `SFTTrainer` shape) — model for Phase 6.
- DeepSeek `api-docs.deepseek.com/guides/json_mode` — model for Phase 4 JSON-mode call.
- Liquid LFM2.5-VL QLoRA recipe (NF4/batch=1/grad_accum/max_len/grad-checkpointing) — model for Phase 6.
- ExtractBench/SOB leaf-flatten + hard parse-gate — model for Phase 7 eval contract.

## Plan Review (Step 8)

_Independent post-finalization review by artifact-code-reviewer and artifact-coverage-reviewer subagents. Findings triaged at Step 9. Coverage reviewer cleared all verification intents (no rows)._

| source   | plan-loc   | codebase-loc                       | severity   | dimension      | finding                                                                                                                                                            | recommendation                                                                                                  | resolution         |
| -------- | ---------- | ---------------------------------- | ---------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------ |
| code     | Phase 6 §1 | src/doc_extract/train.py           | blocker    | actionability  | Phase 1 pins `trl>=0.12,<0.17` but Phase 6 uses `SFTConfig(completion_only_loss=True, max_length=...)`, which only exists on newer TRL; an implementer on an in-range older TRL hits API errors. | Tighten the TRL pin to a range where `SFTConfig` carries these kwargs (e.g. `trl>=0.15,<0.17`).                   | applied: TRL pin tightened to `>=0.15,<0.17` (pyproject + requirements)             |
| code     | Phase 1 §5 | src/doc_extract/schema.py          | concern    | codebase-fit   | Pydantic models don't forbid extra fields, so teacher outputs with keys outside the schema can pass validation despite the "single source of truth" contract.                | Add `model_config = ConfigDict(extra="forbid")` to `Invoice`/`LineItem` (emits strict object constraints).      | applied: `extra="forbid"` on Invoice + LineItem                                          |
| code     | Phase 3 §2 | src/doc_extract/corrupt.py         | concern    | code-quality   | `is_label_preserving` only checks date/amount token multisets; string gold values (vendor, invoice #, currency, descriptions) could be lost/injected without failing it.        | Extend the invariant to also assert key gold string tokens survive (substring presence in dirty_text).           | applied: `is_label_preserving` now also asserts gold string tokens (vendor/invoice#/currency/buyer/PO/descriptions) survive |
| code     | Phase 4 §1 | src/doc_extract/teacher_labeler.py | concern    | code-quality   | `_schema_example()` doesn't resolve Pydantic's `$ref`/`$defs`/`anyOf`, so the prompt's "example json shape" contains invalid placeholders for nested/nullable fields.          | Resolve `$ref` against `$defs` and pick the non-null `anyOf` branch when building the example.                   | applied: `_schema_example` resolves `$ref`/`$defs` and picks the non-null `anyOf` branch |
| code     | Phase 7 §1 | src/doc_extract/evaluate.py        | concern    | code-quality   | `_score_record` counts nullable null-vs-null as TP and null-gold+non-null-pred as FN+FP, inflating micro-F1 and distorting optional-field scoring.                              | Treat nullable-null gold as a non-target (skip); count only spurious non-null preds at nullable fields as FP.    | applied: nullable-null gold = non-target (skip); spurious non-null pred = FP only (scalar + line-item loops) |
| code     | Phase 7 §1 | src/doc_extract/evaluate.py        | concern    | code-quality   | `learning_proven` bootstraps mean per-record F1 deltas, but the headline `delta_micro_f1` is corpus micro-F1 — different statistics.                                            | Bootstrap the corpus-level micro_f1 (resample records, recompute corpus TP/FP/FN) so the CI matches the headline. | applied: `paired_bootstrap_ci_corpus_f1` resamples records + recomputes corpus micro-F1; CI now aligns with `delta_micro_f1` |
| code     | Phase 5 §1 | src/doc_extract/prepare.py         | suggestion | codebase-fit   | `config.SPLIT_MANIFEST` is defined but unused — Phase 5 writes `out_dir/manifest.json`.                                                                            | Remove the unused `SPLIT_MANIFEST` constant (or route the manifest through it).                                  | applied: removed unused `config.SPLIT_MANIFEST`                                            |

## Developer Context
**Step 4 checkpoint (this plan):**
- **Q (schema source): How should the single schema be defined?** A: **Pydantic model** → jsonschema + field-type registry.
- **Q (layout): How to organize the code?** A: **src/ package + per-stage CLIs + Makefile.**
- **Q (tests): Include automated unit tests?** A: **Targeted pytest on invariants.**
- **Q (labeling I/O): Sequential or concurrent teacher calls?** A: **Sequential + resumable.**

**Step 9 triage (this plan):** Plan-reviewer found 7 legitimate findings (1 blocker + 5 concerns + 1 suggestion); coverage fully clear. Developer chose **apply all 7**. All applied via edits to the affected Phase code fences: (1) TRL pin tightened `>=0.15,<0.17`; (2) `extra="forbid"` on Invoice/LineItem; (3) `is_label_preserving` extended to gold string tokens; (4) `_schema_example` resolves `$ref`/`$defs`/`anyOf`; (5) nullable-null gold treated as non-target (spurious non-null pred = FP only) in scalar + line-item loops; (6) `paired_bootstrap_ci_corpus_f1` resamples records + recomputes corpus micro-F1 so `learning_proven` aligns with `delta_micro_f1`; (7) removed unused `config.SPLIT_MANIFEST`.

**Inherited (research/discover — locked, not re-asked):**
- Domain: invoices. GPU: RTX 3090 24GB. Student base: `LFM2.5-VL-1.6B-Extract`. Teacher: `deepseek-v4-flash`.
- Goal: learn the craft (process is the product). Modality: VL kept for forward-looking PDF continuity.
- Data role: cleaned SFT — teacher's clean output IS the target. Eval: field-level metrics vs teacher gold.
- Out of scope: PDF/OCR, DPO, deployment/serving, W&B/DVC/MLOps, real-data transfer (post-training).
- Open residual: the `-Extract` base may show a smaller visible base→fine-tuned delta (already good at
  extraction); switch to base `LFM2.5-VL-1.6B` if a dramatic delta is pedagogically desired.

## Plan History
- Phase 1: Project scaffold + schema/config foundation — approved as generated (added README.md stub; reconciled CLI/revision prose)
- Phase 2: Clean synthetic invoice generator — approved as generated (zero totals→null; currency code verbatim)
- Phase 3: Canonicalization + label-preserving corruption — approved as generated (day-first dates; separator-only repeat; multiset survival invariant; taxonomy/signature prose reconciled)
- Phase 4: Teacher labeling (DeepSeek) — approved as generated (added --seed; criterion reworded to SemanticExtractionError)
- Phase 5: Dataset preparation — approved as generated (added tests/test_prepare_split.py; SFT records carry id)
- Phase 6: Training (QLoRA SFT) — approved as generated (explicit get_peft_model; full-precision merge; self-contained manual criteria)
- Phase 7: Evaluation harness — approved as generated (scipy added to Phase 1; hard gate all-leaves; path-aware Hungarian; both bootstrap CIs; load_model top-level; per-leaf FN+FP)
- Phase 8: Orchestration + reproducibility + reflection — approved as generated (README MODIFY keeps install ref; Makefile/run_all arg lists match locked stages)

## References
- `.rpiv/artifacts/research/2026-06-25_20-26-01_dirty-data-extraction-finetune.md` — chained research (verified model ids, TRL+PEFT recipe, DeepSeek recipe, eval contract, repro boundary).
- `.rpiv/artifacts/discover/2026-06-25_19-47-04_dirty-data-extraction-finetune.md` — FRD (intent, goals, decisions, open questions).
- TRL docs (`huggingface.co/docs/trl`); DeepSeek docs (`api-docs.deepseek.com`); PEFT docs; Liquid docs (`docs.liquid.ai`).
