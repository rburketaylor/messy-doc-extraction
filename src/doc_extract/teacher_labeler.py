"""Phase 4: DeepSeek V4 Flash teacher label factory.

Turns each dirty invoice text into a clean JSON extraction target, with retry/repair/quarantine
and resumable append-only batching. Two failure classes:
  - TRANSPORT (rate-limit/connection/timeout/5xx) -> tenacity exponential backoff retry
  - SEMANTIC (malformed JSON, schema fail, empty content, non-stop finish)
    -> ONE repair, then quarantine
Invalids never enter the training set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from doc_extract import config
from doc_extract.jsonl import append_jsonl, iter_jsonl, sidecar_manifest_path, write_stage_manifest
from doc_extract.schema import INVOICE_JSON_SCHEMA, SCHEMA_VERSION
from doc_extract.validation import (
    InvoiceValidationError,
    make_invoice_validator,
    validate_and_canonicalize_invoice,
)

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
    append_jsonl(path, record, fsync=True)


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
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)}]


def _make_validator():
    return make_invoice_validator()


def _validate_payload(raw_json: str, validator) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise SemanticExtractionError(f"Malformed JSON: {exc.msg} at char {exc.pos}") from exc
    try:
        return validate_and_canonicalize_invoice(payload, validator=validator)
    except InvoiceValidationError as exc:
        raise SemanticExtractionError(str(exc)) from exc


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
        raw = _call_teacher_retry(client, build_messages(dirty_text),
                                  model=model, max_tokens=max_tokens)
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
                "ts": datetime.now(UTC).isoformat(),
            })
            raise


def label_batch(
    *,
    client: OpenAI,
    in_path: Path,
    out_path: Path,
    quarantine_path: Path,
    model: str,
    max_tokens: int,
    seed: int | None = None,
) -> dict[str, int]:
    seen = _load_seen_ids(out_path, quarantine_path)
    processed = labeled = quarantined = transport_failed = skipped = 0
    for rec in iter_jsonl(in_path):
        processed += 1
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
                "ts": datetime.now(UTC).isoformat(),
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
    write_stage_manifest(
        stage="label",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=seed,
        counts=counts,
        inputs={"dirty_jsonl": in_path},
        outputs={"labeled_jsonl": out_path, "quarantine_jsonl": quarantine_path},
        extra={"model": model, "max_tokens": max_tokens},
    )
    logger.info("label_batch %s", counts)
    return counts


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Teacher-label dirty invoices -> labeled.jsonl")
    p.add_argument("--in", dest="inp", type=Path, default=config.DIRTY_JSONL)
    p.add_argument("--out", type=Path, default=config.LABELED_JSONL)
    p.add_argument("--quarantine", type=Path, default=config.QUARANTINE_JSONL)
    p.add_argument("--model", default=config.TEACHER_MODEL_ID)
    p.add_argument("--seed", type=int, default=config.SEED)
    # no-op at temperature=0; kept for CLI consistency
    p.add_argument("--max-tokens", type=int, default=config.TEACHER_MAX_TOKENS)
    args = p.parse_args(argv)
    config.ensure_dirs()
    client = _make_client()
    counts = label_batch(client=client, in_path=args.inp, out_path=args.out,
                         quarantine_path=args.quarantine, model=args.model,
                         max_tokens=args.max_tokens, seed=args.seed)
    print(json.dumps(counts))
    if counts["transport_failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
