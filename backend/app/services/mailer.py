"""
E-postutsending, leverandoer-agnostisk.

Backend velges av konfigurasjonen, i denne rekkefoelgen:
  RESEND_API_KEY  -> Resend (HTTP API)
  SMTP_HOST       -> vanlig SMTP
  ellers          -> kun logging

Poenget er at §10-endepunktet kan bygges og deployes FOER vi har bestemt oss
for en e-postleverandoer: meldingen lagres uansett i databasen, og
videresendingen kobles paa ved aa sette en secret.
"""
import logging
import smtplib
from email.message import EmailMessage

import httpx

from ..config import Settings

log = logging.getLogger(__name__)


def backend_name(cfg: Settings) -> str:
    if cfg.resend_api_key:
        return "resend"
    if cfg.smtp_host:
        return "smtp"
    return "log"


def send(cfg: Settings, subject: str, body: str, reply_to: str | None = None,
         to: str | None = None) -> None:
    """
    Sender e-post. `to` er mottakeren; uten den gaar meldingen til
    cfg.feedback_to (utviklerens innboks), som er riktig for §10.

    NB: parameteren MAA brukes for alt som gaar til en sluttbruker -
    lag-invitasjoner (§4) og innloggingskoder (§1). Uten den havnet
    invitasjonene i utviklerinnboksen i stedet for hos den inviterte.

    Kaster ved feil (kalleren logger).
    """
    mottaker = to or cfg.feedback_to
    if not mottaker:
        log.info("Ingen mottaker (FEEDBACK_TO er tom) - e-post ikke sendt. Emne: %s",
                 subject)
        return

    name = backend_name(cfg)
    if name == "resend":
        payload: dict = {"from": cfg.feedback_from, "to": [mottaker],
                         "subject": subject, "text": body}
        if reply_to:
            payload["reply_to"] = reply_to
        r = httpx.post("https://api.resend.com/emails", json=payload, timeout=10.0,
                       headers={"Authorization": f"Bearer {cfg.resend_api_key}"})
        r.raise_for_status()
        return

    if name == "smtp":
        msg = EmailMessage()
        msg["From"] = cfg.feedback_from
        msg["To"] = cfg.feedback_to
        msg["Subject"] = subject
        if reply_to:
            msg["Reply-To"] = reply_to
        msg.set_content(body)
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10) as smtp:
            if cfg.smtp_starttls:
                smtp.starttls()
            if cfg.smtp_user:
                smtp.login(cfg.smtp_user, cfg.smtp_password)
            smtp.send_message(msg)
        return

    log.info("E-post (kun logg) til %s | %s\n%s", cfg.feedback_to, subject, body)
