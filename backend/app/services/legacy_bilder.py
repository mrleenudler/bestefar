"""
Flytter feilanalyse-bilder fra `image_legacy` i databasen til R2 (§6, AAP-B11).

Radene er donasjoner som ble tatt imot foer opplastingen ble koblet inn
2026-08-15. De skal ikke kastes: de er materialet AAP-U14 mangler for
deduplisering av over-deteksjon, og minst ett av dem er en ekte over-deteksjon
fra felt.

## Rekkefoelgen er hele poenget

For hver rad: last opp, LES TILBAKE og sammenlign byte for byte, og foerst da
toem `image_legacy`. En destruktiv operasjon skal lese ferdig foer den skriver
(rot-CLAUDE.md §7.3) - her er databasen den eneste kopien, saa en PUT som svarte
200 uten at objektet faktisk er lesbart, ville kostet bildet.

Én rad om gangen, med commit per rad. Feiler den fjerde, staar de tre foerste
flyttet og fjerde rad uroert - ingen halvveis tilstand aa rydde opp i.

## Grensen for opplasting gjelder IKKE her

`max_upload_bytes` (8 MB) er en regel for hva MOTTAKET tar imot, ikke for hva vi
flytter. En rad som er stoerre, skal flyttes som de andre; aa avvise den her
ville vaert aa kaste det stoerste bildet vi har fordi en senere regel ikke likte
det.

Merk at ingen av radene faktisk traff grensen. Maalt ved kjoeringen 2026-08-15:
fem rader, 13 byte til 3 844 036 byte, 10 509 298 byte til sammen. Modulen ble
skrevet med en 11 MB-rad som eksempel; det tallet var totalen, ikke en enkeltrad.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..config import Settings
from ..models import FailedAnalysis
from . import objstore

log = logging.getLogger(__name__)


@dataclass
class Utfall:
    flyttet: list[int] = field(default_factory=list)
    hoppet_over: list[tuple[int, str]] = field(default_factory=list)
    feilet: list[tuple[int, str]] = field(default_factory=list)
    byte_flyttet: int = 0

    @property
    def ok(self) -> bool:
        return not self.feilet


def _rader(s: OrmSession) -> list[FailedAnalysis]:
    return list(s.scalars(select(FailedAnalysis)
                          .where(FailedAnalysis.image_legacy.is_not(None))
                          .order_by(FailedAnalysis.id)))


def flytt(s: OrmSession, cfg: Settings, toerrkjoer: bool = True,
          skriv: Callable[[str], None] = log.info) -> Utfall:
    """
    Flytter alle rader med `image_legacy` til R2.

    `toerrkjoer=True` (standard) roerer verken R2 eller basen - den sier hva som
    ville skjedd. Standarden er tryggheten: verktoeyet som kaller denne, maa be
    om aa faa skrive.
    """
    utfall = Utfall()
    if not objstore.er_konfigurert(cfg):
        raise RuntimeError("R2 er ikke konfigurert - ingenting aa flytte til")

    for fa in _rader(s):
        data = fa.image_legacy
        antall = len(data)
        type_og_endelse = objstore.bildetype(data)
        if type_og_endelse is None:
            # Ikke et bilde vi kjenner igjen. Da lar vi raden staa: en blob vi
            # ikke kan navngi, er ikke en vi skal flytte i blinde.
            utfall.hoppet_over.append((fa.id, f"ukjent filtype ({antall} byte)"))
            skriv(f"rad {fa.id}: HOPPET OVER - ukjent filtype, {antall} byte")
            continue
        content_type, endelse = type_og_endelse

        # Datoen i noekkelen er da donasjonen kom inn, ikke da den ble flyttet.
        noekkel = objstore.objektnoekkel(fa.id, fa.tag.value, endelse,
                                         naa=fa.submitted_at)
        if toerrkjoer:
            skriv(f"rad {fa.id}: ville lastet opp {antall} byte "
                  f"({content_type}) til {noekkel}")
            utfall.flyttet.append(fa.id)
            utfall.byte_flyttet += antall
            continue

        try:
            objstore.legg(cfg, noekkel, data, content_type)
            tilbake = objstore.hent(cfg, noekkel)
        except objstore.LagringFeilet as exc:
            utfall.feilet.append((fa.id, str(exc)))
            skriv(f"rad {fa.id}: FEILET - {exc}")
            continue

        if tilbake != data:
            # Objektet finnes, men er ikke det vi la opp. Basen roeres ikke.
            utfall.feilet.append(
                (fa.id, f"lest tilbake {len(tilbake)} byte, la opp {antall}"))
            skriv(f"rad {fa.id}: FEILET - innholdet stemmer ikke, "
                  f"{len(tilbake)} byte tilbake av {antall}")
            continue

        fa.object_key = noekkel
        fa.image_legacy = None
        s.commit()
        utfall.flyttet.append(fa.id)
        utfall.byte_flyttet += antall
        skriv(f"rad {fa.id}: flyttet {antall} byte til {noekkel}")

    return utfall
