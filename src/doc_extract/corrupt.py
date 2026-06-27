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
    structural = [
        line for line in lines
        if line.lstrip().startswith("#") or _TABLE_SEP_RE.match(line.strip())
    ]
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
