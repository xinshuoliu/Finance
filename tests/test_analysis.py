"""Tests for statistical recurring-payment detection."""

import pandas as pd
import pytest

from analysis import detect_recurring


def make_df(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    dates, merchants, amounts, kinds = zip(*rows)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(list(dates)),
            "Details": list(merchants),
            "Merchant": list(merchants),
            "Amount": list(amounts),
            "Debit/Credit": list(kinds),
            "Category": ["Autre"] * len(rows),
        }
    )


def test_monthly_subscription_detected() -> None:
    df = make_df(
        [
            ("2026-01-15", "NETFLIX.COM", 16.99, "Debit"),
            ("2026-02-15", "NETFLIX.COM", 16.99, "Debit"),
            ("2026-03-15", "NETFLIX.COM", 16.99, "Debit"),
        ]
    )
    result = detect_recurring(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["Merchant"] == "NETFLIX.COM"
    assert row["Frequency"] == "monthly"
    assert row["Occurrences"] == 3
    assert row["AverageAmount"] == 16.99
    assert row["MonthlyEstimate"] == pytest.approx(16.99, abs=0.01)
    assert row["NextExpected"] == pd.Timestamp("2026-03-15") + pd.Timedelta(days=30)


def test_weekly_detected() -> None:
    df = make_df(
        [(f"2026-01-{d:02d}", "SUPER GYM", 25.0, "Debit") for d in (5, 12, 19, 26)]
    )
    result = detect_recurring(df)
    assert list(result["Frequency"]) == ["weekly"]
    assert result.iloc[0]["MonthlyEstimate"] == pytest.approx(25.0 * 30 / 7, abs=0.01)


def test_yearly_detected() -> None:
    df = make_df(
        [
            ("2024-06-01", "DOMAIN RENEWAL", 20.0, "Debit"),
            ("2025-06-01", "DOMAIN RENEWAL", 20.0, "Debit"),
            ("2026-06-01", "DOMAIN RENEWAL", 20.0, "Debit"),
        ]
    )
    assert list(detect_recurring(df)["Frequency"]) == ["yearly"]


def test_unstable_amounts_rejected() -> None:
    df = make_df(
        [
            ("2026-01-15", "SHOP", 10.0, "Debit"),
            ("2026-02-15", "SHOP", 30.0, "Debit"),
            ("2026-03-15", "SHOP", 60.0, "Debit"),
        ]
    )
    assert detect_recurring(df).empty


def test_irregular_intervals_rejected() -> None:
    df = make_df(
        [
            ("2026-01-01", "SHOP", 20.0, "Debit"),
            ("2026-01-06", "SHOP", 20.0, "Debit"),
            ("2026-02-15", "SHOP", 20.0, "Debit"),
        ]
    )
    assert detect_recurring(df).empty


def test_too_few_occurrences_rejected() -> None:
    df = make_df(
        [
            ("2026-01-15", "NETFLIX.COM", 16.99, "Debit"),
            ("2026-02-15", "NETFLIX.COM", 16.99, "Debit"),
        ]
    )
    assert detect_recurring(df).empty


def test_credits_ignored() -> None:
    df = make_df(
        [
            ("2026-01-15", "EMPLOYER PAY", 2000.0, "Credit"),
            ("2026-02-15", "EMPLOYER PAY", 2000.0, "Credit"),
            ("2026-03-15", "EMPLOYER PAY", 2000.0, "Credit"),
        ]
    )
    assert detect_recurring(df).empty


def test_same_day_charges_count_once() -> None:
    df = make_df(
        [
            ("2026-01-15", "NETFLIX.COM", 8.0, "Debit"),
            ("2026-01-15", "NETFLIX.COM", 8.99, "Debit"),  # same day: one occurrence of 16.99
            ("2026-02-15", "NETFLIX.COM", 16.99, "Debit"),
            ("2026-03-15", "NETFLIX.COM", 16.99, "Debit"),
        ]
    )
    result = detect_recurring(df)
    assert len(result) == 1
    assert result.iloc[0]["Occurrences"] == 3
    assert result.iloc[0]["Frequency"] == "monthly"


def test_sorted_by_monthly_estimate() -> None:
    rows = []
    for month in (1, 2, 3):
        rows.append((f"2026-{month:02d}-10", "SMALL SUB", 5.0, "Debit"))
        rows.append((f"2026-{month:02d}-20", "BIG SUB", 50.0, "Debit"))
    result = detect_recurring(make_df(rows))
    assert list(result["Merchant"]) == ["BIG SUB", "SMALL SUB"]


def test_empty_dataframe() -> None:
    assert detect_recurring(pd.DataFrame()).empty
