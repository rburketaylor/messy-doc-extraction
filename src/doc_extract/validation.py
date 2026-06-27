"""Shared invoice label validation and canonicalization."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema import ValidationError as JsonSchemaError
from pydantic import ValidationError as PydanticValidationError

from doc_extract import canon
from doc_extract.schema import INVOICE_JSON_SCHEMA, Invoice


class InvoiceValidationError(ValueError):
    """Raised when a candidate invoice label cannot become a canonical schema-valid invoice."""


def make_invoice_validator() -> Draft202012Validator:
    Draft202012Validator.check_schema(INVOICE_JSON_SCHEMA)
    return Draft202012Validator(INVOICE_JSON_SCHEMA, format_checker=FormatChecker())


def validate_and_canonicalize_invoice(
    payload: Any, *, validator: Draft202012Validator | None = None
) -> dict[str, Any]:
    """Validate a teacher label and return the canonicalized invoice dict.

    Schema/Pydantic checks enforce required fields, enums, structure, and extras. The canonicalizer
    then rejects semantically bad strings that the schema cannot express, such as impossible dates,
    negative amounts, and word quantities.
    """
    if not isinstance(payload, dict):
        raise InvoiceValidationError(
            f"Top-level JSON must be an object, got {type(payload).__name__}"
        )
    if validator is not None:
        try:
            validator.validate(payload)
        except JsonSchemaError as exc:
            raise InvoiceValidationError(f"jsonschema validation failed: {exc.message}") from exc
    try:
        parsed = Invoice.model_validate(payload).model_dump()
    except PydanticValidationError as exc:
        raise InvoiceValidationError(f"pydantic validation failed: {exc}") from exc
    try:
        canonical = canon.canonicalize_invoice(parsed)
    except Exception as exc:
        raise InvoiceValidationError(f"canonicalization failed: {exc}") from exc
    try:
        return Invoice.model_validate(canonical).model_dump()
    except PydanticValidationError as exc:
        raise InvoiceValidationError(f"canonicalized invoice failed validation: {exc}") from exc
