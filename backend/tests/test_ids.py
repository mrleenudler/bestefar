"""Offentlig bruker-ID (backend_spec §3.1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import ids  # noqa: E402


def test_generert_id_har_riktig_form_og_er_gyldig():
    for _ in range(200):
        pid = ids.generate()
        assert pid.startswith("BF-")
        assert len(pid) == len("BF-7Q4K-9F2M")
        assert ids.is_valid(pid)


def test_forvekslingssikre_tegn_finnes_ikke():
    # Crockford base32 utelater I, L, O og U nettopp fordi de forveksles.
    for _ in range(200):
        assert not set("ILOU") & set(ids.generate())


def test_normalisering_taaler_slurv():
    pid = ids.generate()
    kompakt = pid.replace("-", "")
    assert ids.is_valid(kompakt)
    assert ids.is_valid(kompakt.lower())
    assert ids.is_valid(f"  {pid}  ")
    assert ids.is_valid(kompakt[len("BF"):])          # uten prefiks


def test_forvekslede_tegn_foldes_til_riktig_verdi():
    # Skriver brukeren O for 0 eller I/L for 1, skal ID-en fortsatt godtas.
    body = "0123456"
    pid = ids.format_id(body + ids._checksum(body))
    slurvete = pid.replace("0", "O").replace("1", "I")
    assert ids.is_valid(slurvete)
    assert ids.normalize(slurvete) == ids.normalize(pid)


def test_sjekksiffer_fanger_tastefeil():
    pid = ids.normalize(ids.generate())
    # Endre ett tegn i den tilfeldige delen -> sjekksifferet skal ikke stemme.
    feil = ids.ALPHABET[(ids._VALUE[pid[0]] + 1) % 32] + pid[1:]
    assert not ids.is_valid(feil)


def test_ugyldig_lengde_avvises():
    assert ids.normalize("BF-123") is None
    assert not ids.is_valid("BF-123")
