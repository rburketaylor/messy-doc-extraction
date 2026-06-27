"""Single source of truth for the invoice extraction schema.

A Pydantic model defines the fields; `.model_json_schema()` feeds the teacher's jsonschema
validation (Phase 4) AND encodes the currency/date constraints structurally, and
`FIELD_TYPE_REGISTRY` feeds the eval canonicalizer (Phase 3/7). Amounts/quantities are strings
to avoid float drift; dates are ISO-8601; currency is ISO-4217.
"""

from __future__ import annotations

import typing
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

# ISO-4217 subset. Single source: the schema encodes it as an enum, the generator reads the set.
Currency = Literal["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CNY", "INR"]
CURRENCIES: set[str] = set(typing.get_args(Currency))

_ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Invoice line: amount == quantity * unit_price (kept consistent by the generator)."""

    description: str
    quantity: str  # string to preserve "12" / "12.5" / "1,000"
    unit: str | None = None  # e.g. EA, pcs, each, kg
    unit_price: str  # string amount
    amount: str  # string amount


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor_name: str
    buyer_name: str | None = None
    invoice_number: str
    invoice_date: str = Field(pattern=_ISO_DATE_PATTERN)  # -> "pattern" in JSON Schema
    currency: Currency = "USD"  # -> "enum" in JSON Schema
    purchase_order_number: str | None = None
    subtotal: str | None = None
    tax_total: str | None = None
    shipping_total: str | None = None
    discount_total: str | None = None
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
