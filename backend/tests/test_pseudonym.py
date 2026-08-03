"""Pseudonym forsknings-ID (backend_spec §1, §7)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.services import pseudonym  # noqa: E402

USER = "11111111-2222-3333-4444-555555555555"


def _cfg(secret: str = "hemmelighet") -> Settings:
    return Settings(research_pseudonym_secret=secret, _env_file=None)


def test_deterministisk_for_samme_bruker():
    cfg = _cfg()
    assert pseudonym.for_user(cfg, USER) == pseudonym.for_user(cfg, USER)


def test_ulike_brukere_gir_ulike_pseudonymer():
    cfg = _cfg()
    assert pseudonym.for_user(cfg, USER) != pseudonym.for_user(cfg, USER[:-1] + "6")


def test_pseudonymet_avsloerer_ikke_konto_iden():
    pid = pseudonym.for_user(_cfg(), USER)
    assert USER not in pid
    assert USER.replace("-", "") not in pid


def test_ny_hemmelighet_gir_nytt_pseudonym():
    # Dokumenterer konsekvensen av rotasjon: koblingen til eksisterende
    # forskningsdata brytes. Se docstringen i services/pseudonym.py.
    assert pseudonym.for_user(_cfg("a"), USER) != pseudonym.for_user(_cfg("b"), USER)


def test_uten_hemmelighet_nekter_vi_aa_pseudonymisere():
    with pytest.raises(pseudonym.PseudonymNotConfigured):
        pseudonym.for_user(_cfg(""), USER)
