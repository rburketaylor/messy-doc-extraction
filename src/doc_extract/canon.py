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
