"""Direct invoice validation and canonicalization gates."""

from __future__ import annotations

import copy

import pytest

from doc_extract.validation import (
    InvoiceValidationError,
    make_invoice_validator,
    validate_and_canonicalize_invoice,
)

_VALID = {
    "vendor_name": "Acme",
    "buyer_name": None,
    "invoice_number": "INV-1",
    "invoice_date": "2026-06-25",
    "currency": "USD",
    "purchase_order_number": None,
    "subtotal": None,
    "tax_total": None,
    "shipping_total": None,
    "discount_total": None,
    "grand_total": "100.00",
    "line_items": [{
        "description": "Widget",
        "quantity": "1",
        "unit": None,
        "unit_price": "100.00",
        "amount": "100.00",
    }],
}


def _invoice(**updates):
    inv = copy.deepcopy(_VALID)
    inv.update(updates)
    return inv


def test_rejects_non_object_top_level():
    with pytest.raises(InvoiceValidationError, match="Top-level JSON"):
        validate_and_canonicalize_invoice([], validator=make_invoice_validator())


def test_rejects_extra_keys():
    with pytest.raises(InvoiceValidationError):
        validate_and_canonicalize_invoice(
            _invoice(unexpected="x"), validator=make_invoice_validator()
        )


def test_rejects_invalid_currency_enum():
    with pytest.raises(InvoiceValidationError):
        validate_and_canonicalize_invoice(
            _invoice(currency="XYZ"), validator=make_invoice_validator()
        )


def test_rejects_impossible_dates():
    with pytest.raises(InvoiceValidationError, match="canonicalization failed"):
        validate_and_canonicalize_invoice(_invoice(invoice_date="2026-02-31"))


@pytest.mark.parametrize("field,value", [("discount_total", "-1.00"), ("grand_total", "N/A")])
def test_rejects_negative_or_unparseable_amounts(field, value):
    with pytest.raises(InvoiceValidationError, match="canonicalization failed"):
        validate_and_canonicalize_invoice(_invoice(**{field: value}))


def test_returns_canonicalized_success_case():
    inv = _invoice(
        vendor_name="  Acme   Corp ",
        buyer_name=" Big   Buyer ",
        subtotal="$1,000.0",
        grand_total="USD 1,000",
    )
    inv["line_items"][0].update({
        "description": "  Widget   labor ",
        "quantity": "2.00",
        "unit": "each",
        "unit_price": "$500",
        "amount": "1,000.0",
    })

    out = validate_and_canonicalize_invoice(inv)

    assert out["vendor_name"] == "Acme Corp"
    assert out["buyer_name"] == "Big Buyer"
    assert out["subtotal"] == "1000.00"
    assert out["grand_total"] == "1000.00"
    assert out["line_items"][0] == {
        "description": "Widget labor",
        "quantity": "2",
        "unit": "EA",
        "unit_price": "500.00",
        "amount": "1000.00",
    }
