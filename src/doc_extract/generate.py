"""Phase 2: generate realistic CLEAN synthetic invoices and render them to markdown text.

Each record carries a known-clean ground-truth JSON (validated against the Pydantic schema) and a
rendered clean_text. Phase 3 corrupts clean_text -> dirty_text; the teacher (Phase 4) extracts it
back to JSON. Totals are self-consistent; zero totals are null so a field's absence in text maps
exactly to a null gold value (no zero-vs-null ambiguity), and the currency code is rendered
verbatim so extraction is unambiguous.
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from faker import Faker

from doc_extract import config
from doc_extract.jsonl import sidecar_manifest_path, write_jsonl, write_stage_manifest
from doc_extract.schema import CURRENCIES, SCHEMA_VERSION, Invoice

UNITS = ["EA", "pcs", "each", "kg", "box", "set", "hr", "m", "L"]
ITEM_NOUNS = [
    "widget", "gasket", "bearing", "valve", "bracket", "harness", "module", "actuator",
    "nozzle", "assembly", "filter", "sensor", "fixture", "fastener", "unit",
]
_CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "CAD": "C$", "AUD": "A$", "INR": "₹",
}
_CENT = Decimal("0.01")


def _round_money(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _money(rng: random.Random, lo: float, hi: float) -> str:
    lo_cents = int(Decimal(str(lo)) * 100)
    hi_cents = int(Decimal(str(hi)) * 100)
    return f"{Decimal(rng.randint(lo_cents, hi_cents)) / 100:.2f}"


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
        amount = f"{_round_money(Decimal(q) * Decimal(unit_price)):.2f}"
        line_items.append({
            "description": f"{faker.word().capitalize()} {rng.choice(ITEM_NOUNS)}".strip(),
            "quantity": q,
            "unit": rng.choice(UNITS),
            "unit_price": unit_price,
            "amount": amount,
        })
    subtotal = sum((Decimal(li["amount"]) for li in line_items), Decimal("0.00"))
    tax = _round_money(subtotal * rng.choice([
        Decimal("0.00"), Decimal("0.05"), Decimal("0.075"), Decimal("0.10"),
    ]))
    shipping = rng.choice([Decimal("0.00"), Decimal("15.00"), Decimal("25.00"), Decimal("45.00")])
    discount = rng.choice([Decimal("0.00"), _round_money(subtotal * Decimal("0.02"))])
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


def _iter_clean_records(n_docs: int, seed: int):
    faker = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    template_names = list(TEMPLATES)
    for i in range(n_docs):
        inv = build_clean_invoice(faker, rng)
        tname = rng.choice(template_names)
        text = TEMPLATES[tname](inv)
        yield {"id": f"doc-{i:05d}", "clean_json": inv, "clean_text": text, "template": tname}


def generate(n_docs: int, seed: int, out_path: Path) -> int:
    out_path = Path(out_path)
    n = write_jsonl(out_path, _iter_clean_records(n_docs, seed))
    write_stage_manifest(
        stage="generate",
        manifest_path=sidecar_manifest_path(out_path),
        schema_version=SCHEMA_VERSION,
        seed=seed,
        counts={"requested": n_docs, "written": n},
        outputs={"clean_jsonl": out_path},
        extra={"templates": sorted(TEMPLATES)},
    )
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
