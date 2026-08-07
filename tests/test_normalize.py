"""Tests for bank description normalization (stable merchant keys)."""

import pytest

from ai.normalize import normalize

# Real description formats from the banks this app already imports
# (taken from the learned keywords in categories.json)
REAL_FORMATS = {
    "MCDONALD'S #22028 MONT-TREMBLANQC": "MCDONALD'S",
    "DANS UN JARDIN #9086 ST-BRUNO QC": "DANS UN JARDIN",
    "SP SHOPPAY LAURA HAMPSTEAD QC": "SP SHOPPAY LAURA",
    "HUGO BOSS - 6186 St-Bruno-de-MQC": "HUGO BOSS",
    "SQDC77001 SQDC.CA MONTREAL QC": "SQDC SQDC.CA",
    "SHELL C10214 MONT-TREMBLANQC": "SHELL",
    "A&W #5264 ST-BRUNO-DE-MQC": "A&W",
    "POKE MONSTER SAINT-BRUNO QC": "POKE MONSTER",
    "CHRONO-RECHARGE OPUS MONTREAL QC": "CHRONO-RECHARGE OPUS",
    "ROSE BLOC GREENFIELD PAQC": "ROSE BLOC GREENFIELD",
    "SKI MONT BLANC MONT-BLANC QC": "SKI MONT BLANC",
    "DALDONGNAE MONTREAL QC": "DALDONGNAE",
    "PARC INDIGO - NO CMO28 MONTREAL QC": "PARC INDIGO NO CMO28",
    "BOOKING.COM": "BOOKING.COM",
}


@pytest.mark.parametrize("raw,expected", sorted(REAL_FORMATS.items()))
def test_real_bank_formats(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_amzn_reference_suffixes_collapse_to_same_key() -> None:
    a = normalize("AMZN MKTP CA*2K4LD8")
    b = normalize("AMZN MKTP CA*9XQ1")
    assert a == b == "AMZN MKTP CA"


def test_french_description() -> None:
    assert normalize("ACHAT INTERAC - PHARMACIE JEAN COUTU MONTREAL QC") == "PHARMACIE JEAN COUTU"


def test_french_transaction_tokens_dropped() -> None:
    assert normalize("PAIEMENT PREAUTORISE VIDEOTRON") == "VIDEOTRON"
    assert normalize("VIREMENT INTERAC LOYER") == "LOYER"


def test_english_transaction_tokens_dropped() -> None:
    assert normalize("POS PURCHASE TIM HORTONS #123 TORONTO ON") == "TIM HORTONS"


def test_long_digit_runs_stripped() -> None:
    assert normalize("NETFLIX.COM 8665797172 ON") == "NETFLIX.COM"


def test_case_and_whitespace_insensitive() -> None:
    assert normalize("  spotify   music  ") == normalize("SPOTIFY MUSIC")


def test_idempotent() -> None:
    for raw in REAL_FORMATS:
        once = normalize(raw)
        assert normalize(once) == once


def test_empty_inputs() -> None:
    assert normalize("") == ""
    assert normalize(None) == ""
    assert normalize("   ") == ""


def test_never_empty_for_nonempty_input() -> None:
    # Even a description made only of stripped patterns keeps a stable key
    assert normalize("#1234") != ""
