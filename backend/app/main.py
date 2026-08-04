"""
Bestefar backend (backend_spec.md).

Ansvarsomraadene er atskilte routere (kravspec §5):
  /v1/auth             innlogging: Google, Apple, e-postkode (§1)
  /v1/profile          profil og delingsvalg (§1, §3)
  /v1/users, /v1/friends  soek, venneforespoersler, delt statistikk (§3, §3.1)
  /v1/teams            lag, invitasjoner, lederskap og avstemning (§4, §11)
  /v1/messages         meldingskoe hentet ved oppstart (§11)
  /v1/stats            brukerens egne resultatdata
  /v1/backup           klient-kryptert backup av logg og innstillinger (§2)
  /v1/failed-analyses  opt-in innsending av feilede analyser
  /v1/research         forskningsdata, strukturelt adskilt og samtykke-gatet
  /v1/feedback         melding til utvikler (§10)
"""
import logging

from fastapi import FastAPI

from .config import settings
from .routers import (auth, backup, failed_analyses, feedback, friends, health,
                      messages, profile, research, stats, teams)

cfg = settings()
logging.basicConfig(level=cfg.log_level.upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Bestefar backend", version="0.2",
              # Ingen offentlig API-dokumentasjon i produksjon.
              docs_url=None if cfg.is_prod else "/docs",
              redoc_url=None)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(feedback.router)
app.include_router(profile.router)
app.include_router(friends.router)
app.include_router(teams.router)
app.include_router(teams.invite_router)
app.include_router(messages.router)
app.include_router(stats.router)
app.include_router(backup.router)
app.include_router(failed_analyses.router)
app.include_router(research.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "bestefar-api", "version": app.version}
