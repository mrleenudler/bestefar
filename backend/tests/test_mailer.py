"""
Utsendingen i services/mailer.py - hvem meldingen faktisk gaar TIL.

Modulen har to leverandoergrener, og de har hatt hver sin sannhet: Resend-grenen
ble rettet da `to` ble innfoert (ellers havnet lag-invitasjoner i
utviklerinnboksen), mens SMTP-grenen ble staaende med utviklerinnboksen
hardkodet i To-hodet til 2026-08-22. Produksjonen bruker Resend, saa feilen var
sovende - den ville vaaknet den dagen SMTP ble tatt i bruk, og da paa
innloggingskoder.

Testene her kjoerer BEGGE grenene mot det samme kravet.
"""
import pytest

BRUKER = "mottaker@example.com"
UTVIKLER = "utvikler@example.com"


@pytest.fixture()
def cfg_smtp(monkeypatch, db_url):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("FEEDBACK_TO", UTVIKLER)
    monkeypatch.setenv("FEEDBACK_FROM", "ikke-svar@example.com")

    from app.config import settings
    settings.cache_clear()
    return settings()


@pytest.fixture()
def cfg_resend(monkeypatch, db_url):
    monkeypatch.setenv("RESEND_API_KEY", "test-noekkel")
    monkeypatch.setenv("FEEDBACK_TO", UTVIKLER)
    monkeypatch.setenv("FEEDBACK_FROM", "ikke-svar@example.com")

    from app.config import settings
    settings.cache_clear()
    return settings()


def _fang_smtp(monkeypatch):
    """Bytter ut smtplib.SMTP og returnerer lista over sendte meldinger."""
    from app.services import mailer

    sendt = []

    class FalskSMTP:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, *a):
            pass

        def send_message(self, msg):
            sendt.append(msg)

    monkeypatch.setattr(mailer.smtplib, "SMTP", FalskSMTP)
    return sendt


def test_smtp_sender_til_oppgitt_mottaker(monkeypatch, cfg_smtp):
    """
    Kjernen i feilen: `send_message` tar konvolutt-adressen fra To-hodet, saa et
    hardkodet cfg.feedback_to her sender brukerens innloggingskode til
    utvikleren.
    """
    from app.services import mailer

    sendt = _fang_smtp(monkeypatch)
    mailer.send(cfg_smtp, "Innloggingskode til Bestefar", "Koden din er 123456.",
                to=BRUKER)

    assert len(sendt) == 1
    assert sendt[0]["To"] == BRUKER
    assert UTVIKLER not in str(sendt[0])


def test_smtp_uten_mottaker_gaar_til_utvikleren(monkeypatch, cfg_smtp):
    """§10: feedback har ingen `to`, og skal da til utviklerinnboksen."""
    from app.services import mailer

    sendt = _fang_smtp(monkeypatch)
    mailer.send(cfg_smtp, "[Bestefar] Scan feiler", "kropp")

    assert sendt[0]["To"] == UTVIKLER


def test_resend_sender_til_oppgitt_mottaker(monkeypatch, cfg_resend):
    """Samme krav over den andre leverandoeren."""
    import httpx

    from app.services import mailer

    sett = {}

    def _post(url, **kw):
        sett.update(kw["json"])
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(mailer.httpx, "post", _post)
    mailer.send(cfg_resend, "Invitasjon til Bjoernejegerne", "lenke", to=BRUKER)

    assert sett["to"] == [BRUKER]
    assert sett["from"] == "ikke-svar@example.com"
