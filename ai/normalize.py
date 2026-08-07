"""Normalize bank descriptions into stable merchant keys.

Banks append reference numbers, store numbers and city/province codes that
vary between transactions of the same merchant. Normalization strips them so
that "AMZN MKTP CA*2K4LD8" and "AMZN MKTP CA*9XQ1" produce the same key.
"""

import re

# Transaction-type tokens that carry no merchant information (EN + FR)
TRANSACTION_TOKENS = {
    "POS",
    "DEBIT",
    "ACH",
    "PURCHASE",
    "RETRAIT",
    "PAIEMENT",
    "VIREMENT",
    "ACHAT",
    "INTERAC",
    "PREAUTORISE",
    "PREAUTHORIZED",
}

# Canadian province/territory codes seen at the end of bank descriptions
PROVINCE_CODES = {"QC", "ON", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "YT", "NT", "NU"}

_STAR_SUFFIX_RE = re.compile(r"\*\w+\s*$")  # AMZN MKTP CA*2K4LD8
_HASH_REF_RE = re.compile(r"#\w*")  # store numbers: #22028, #9086
_LONG_DIGITS_RE = re.compile(r"\d{4,}")  # card/ref numbers, incl. glued: SQDC77001
_PUNCT = ".,;:-/&*#'"


def _strip_trailing_location(tokens: list[str]) -> list[str]:
    """Drop a trailing province code and the city token preceding it."""
    if not tokens:
        return tokens

    last = tokens[-1]

    if last in PROVINCE_CODES:
        tokens = tokens[:-1]
        # Also drop the city token, but always keep at least the merchant name
        if len(tokens) >= 2:
            tokens = tokens[:-1]
        return tokens

    # Truncated exports glue the city to the province: "MONT-TREMBLANQC".
    # Only QC is safe to detect this way (many words legitimately end in ON/BC).
    if len(last) >= 4 and last.endswith("QC") and len(tokens) >= 2:
        return tokens[:-1]

    return tokens


def normalize(desc: str) -> str:
    """Collapse bank description variants to a stable merchant key.

    The key is UPPERCASE, without reference numbers (#1234, *2K4LD8, runs of
    4+ digits), without transaction-type words (POS, ACHAT, INTERAC…) and
    without the trailing city/province. A non-empty input never produces an
    empty key.
    """
    if desc is None:
        return ""

    original = str(desc).upper()
    s = original.strip()

    s = _STAR_SUFFIX_RE.sub(" ", s)
    s = _HASH_REF_RE.sub(" ", s)
    s = _LONG_DIGITS_RE.sub(" ", s)

    tokens = [t for t in s.split() if t.strip(_PUNCT)]  # drop "-", "/" separators
    tokens = [t for t in tokens if t.strip(_PUNCT) not in TRANSACTION_TOKENS]
    tokens = _strip_trailing_location(tokens)

    # Digit-stripping can leave orphan single letters ("SHELL C10214" -> "SHELL C")
    if len(tokens) > 1:
        tokens = [t for t in tokens if len(t) > 1]

    result = " ".join(tokens).strip()
    if not result:
        # Never return an empty key for a non-empty description
        return " ".join(original.split())
    return result
