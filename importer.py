"""Bank CSV import: encoding/separator detection, column mapping, normalization.

Pure pandas — no Streamlit. main.py adds the manual-mapping UI on top of the
auto-detection; scripts (e.g. the evaluation harness) use load_bank_csv.
"""

import re

import pandas as pd

DATE_ALIASES = [
    "transaction date", "date", "posted date", "trade date",
    "date de transaction", "date d'opération", "date operation",
]
DETAILS_ALIASES = [
    "description", "details", "merchant", "payee", "name", "memo",
    "libellé", "référence", "reference",
]
AMOUNT_ALIASES = [
    "transaction amount", "amount", "montant", "valeur", "amount (cad)", "amount cad",
]
DEBIT_ALIASES = ["debit", "withdrawal", "sortie", "débit"]
CREDIT_ALIASES = ["credit", "deposit", "entrée", "crédit"]


def _norm_col(c: str) -> str:
    return str(c).strip().lower()


def find_column(df: pd.DataFrame, aliases: list[str]):
    """Find the column matching one of the aliases (exact first, then contains)."""
    cols = list(df.columns)
    norm = {c: _norm_col(c) for c in cols}

    for a in aliases:
        a = a.lower()
        for c, n in norm.items():
            if n == a:
                return c

    for a in aliases:
        a = a.lower()
        for c, n in norm.items():
            if a in n:
                return c

    return None


def smart_read_csv(file) -> pd.DataFrame:
    """Try several encodings, separators and skiprows values (banks often add
    preamble lines above the header). Picks the parse with the most columns."""
    best_df = None
    best_cols = -1

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        decoded_any = False
        for sep in (",", ";", "\t"):
            for skip in range(7):
                try:
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=skip, sep=sep, encoding=encoding)
                except Exception:
                    continue
                decoded_any = True
                if not df.empty and df.shape[1] > best_cols:
                    best_df = df
                    best_cols = df.shape[1]
        if decoded_any:
            # This encoding decoded the bytes fine; trying more would only
            # reinterpret the same data.
            break

    if best_df is None:
        raise ValueError("Could not read CSV (tried multiple encodings, separators and header offsets).")

    best_df.columns = [str(c).strip() for c in best_df.columns]
    return best_df


def clean_amount(value):
    """Parse one amount string: $1,234.56 / 1 234,56 / (123.45) / -12,50 ..."""
    v = str(value).strip()
    if v.lower() in ("", "nan", "none", "-"):
        return None

    negative = v.startswith("(") and v.endswith(")")
    if negative:
        v = v[1:-1]
    if v.lstrip().startswith("-"):
        negative = True

    # keep only digits and separators (drops $, spaces, nbsp, currency codes)
    v = re.sub(r"[^\d,.]", "", v)
    if not v:
        return None

    if "," in v and "." in v:
        # the rightmost separator is the decimal one
        if v.rfind(",") > v.rfind("."):
            v = v.replace(".", "").replace(",", ".")
        else:
            v = v.replace(",", "")
    elif "," in v:
        head, _, tail = v.rpartition(",")
        if len(tail) <= 2:  # decimal comma: 12,50
            v = head.replace(",", "") + "." + tail
        else:  # thousands separator: 1,234
            v = v.replace(",", "")

    try:
        n = float(v)
    except ValueError:
        return None
    return -n if negative else n


def parse_amount_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.map(clean_amount), errors="coerce")


def parse_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # Day-first heuristic: if the first component ever exceeds 12 (and the
    # second never does), the format must be DD/MM/YYYY.
    dayfirst = False
    parts = s.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")
    if parts[0].notna().any():
        first = pd.to_numeric(parts[0], errors="coerce")
        second = pd.to_numeric(parts[1], errors="coerce")
        if (first > 12).any() and not (second > 12).any():
            dayfirst = True

    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")


def auto_detect_mapping(df_raw: pd.DataFrame) -> dict | None:
    """Detect the column layout; None when the file needs manual mapping."""
    date_col = find_column(df_raw, DATE_ALIASES)
    details_col = find_column(df_raw, DETAILS_ALIASES)
    amount_col = find_column(df_raw, AMOUNT_ALIASES)
    debit_col = find_column(df_raw, DEBIT_ALIASES)
    credit_col = find_column(df_raw, CREDIT_ALIASES)

    if date_col and details_col and amount_col:
        return {
            "mode": "single",
            "date_col": date_col,
            "details_col": details_col,
            "amount_col": amount_col,
        }
    if date_col and details_col and debit_col and credit_col:
        return {
            "mode": "split",
            "date_col": date_col,
            "details_col": details_col,
            "debit_col": debit_col,
            "credit_col": credit_col,
        }
    return None


def normalize_transactions(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame()

    out["Date"] = parse_dates(df_raw[mapping["date_col"]])
    out["Details"] = df_raw[mapping["details_col"]].astype(str).str.strip()

    if mapping["mode"] == "single":
        amt = parse_amount_series(df_raw[mapping["amount_col"]])
        out["Debit/Credit"] = amt.apply(lambda x: "Credit" if pd.notna(x) and x < 0 else "Debit")
        out["Amount"] = amt.abs()
    else:
        debit = parse_amount_series(df_raw[mapping["debit_col"]]).abs().fillna(0)
        credit = parse_amount_series(df_raw[mapping["credit_col"]]).abs().fillna(0)
        out["Debit/Credit"] = (credit > 0).map({True: "Credit", False: "Debit"})
        total = debit + credit
        out["Amount"] = total.where(total > 0)  # rows with no amount at all become NaN

    out = out.dropna(subset=["Date", "Amount"])
    return out[["Date", "Details", "Amount", "Debit/Credit"]].reset_index(drop=True)


def mapping_matches(mapping: dict, df: pd.DataFrame) -> bool:
    return all(mapping[k] in df.columns for k in mapping if k.endswith("_col"))


def load_bank_csv(path) -> pd.DataFrame:
    """Load a bank CSV via auto-detection only; raises ValueError on failure."""
    with open(path, "rb") as f:
        df_raw = smart_read_csv(f)
    mapping = auto_detect_mapping(df_raw)
    if mapping is None:
        raise ValueError(f"{path}: could not auto-detect the column layout")
    df = normalize_transactions(df_raw, mapping)
    if df.empty:
        raise ValueError(f"{path}: no valid transactions found")
    return df
