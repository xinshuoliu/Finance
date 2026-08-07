"""Statistical detection of recurring payments — no AI involved."""

import pandas as pd

# (name, min gap, max gap, nominal days)
_BUCKETS = [
    ("weekly", 6, 8, 7),
    ("biweekly", 12, 16, 14),
    ("monthly", 26, 33, 30),
    ("yearly", 350, 380, 365),
]

_COLUMNS = [
    "Merchant",
    "Frequency",
    "Occurrences",
    "AverageAmount",
    "LastDate",
    "NextExpected",
    "MonthlyEstimate",
]


def detect_recurring(
    df: pd.DataFrame, min_occurrences: int = 3, max_cv: float = 0.15
) -> pd.DataFrame:
    """Detect merchants charging at a steady rhythm with stable amounts.

    A merchant is recurring when it has at least min_occurrences charge days,
    consecutive gaps fitting one interval bucket (7d, 14d, 28-31d, 365d) for
    at least 75% of the gaps, and a coefficient of variation of the amounts
    below max_cv. Only debits are considered; same-day charges count as one
    occurrence.
    """
    if df.empty or "Merchant" not in df.columns:
        return pd.DataFrame(columns=_COLUMNS)

    debits = df[df["Debit/Credit"] == "Debit"]
    rows = []
    for merchant, group in debits.groupby("Merchant"):
        if not merchant:
            continue
        daily = group.groupby(group["Date"].dt.normalize())["Amount"].sum().sort_index()
        if len(daily) < min_occurrences:
            continue

        mean_amount = float(daily.mean())
        if mean_amount <= 0:
            continue
        # Population std: sample std over 3-4 points would over-penalize
        cv = float(daily.std(ddof=0)) / mean_amount
        if cv >= max_cv:
            continue

        gaps = daily.index.to_series().diff().dropna().dt.days
        median_gap = float(gaps.median())
        bucket = next(
            ((name, lo, hi, nominal) for name, lo, hi, nominal in _BUCKETS if lo <= median_gap <= hi),
            None,
        )
        if bucket is None:
            continue
        name, lo, hi, nominal = bucket
        if float(gaps.between(lo, hi).mean()) < 0.75:
            continue

        last_date = daily.index.max()
        rows.append(
            {
                "Merchant": merchant,
                "Frequency": name,
                "Occurrences": int(len(daily)),
                "AverageAmount": round(mean_amount, 2),
                "LastDate": last_date,
                "NextExpected": last_date + pd.Timedelta(days=nominal),
                "MonthlyEstimate": round(mean_amount * 30 / nominal, 2),
            }
        )

    result = pd.DataFrame(rows, columns=_COLUMNS)
    if not result.empty:
        result = result.sort_values("MonthlyEstimate", ascending=False).reset_index(drop=True)
    return result
