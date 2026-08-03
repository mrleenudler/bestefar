"""
Misbruksvern for brukersoek (backend_spec §3.1).

Telefonnumre er personsensitive, og et soekeendepunkt som svarer «finnes /
finnes ikke» er en enumereringsmaskin hvis det staar aapent. Spec-en setter
grensen: 5 mislykkede telefonsoek paa ett doegn gir karantene - 1 doegn ved
foerste overtredelse, eskalerende til 7 doegn ved gjentakelse.

To ting er verdt aa merke seg:

1. **Bare MISLYKKEDE soek telles.** Et treff er normal bruk; det er serien av
   bom som avslorer at noen prover seg fram.
2. **Telleren ligger i databasen**, ikke i minnet slik som ratelimit.py. En
   karantene som forsvinner ved omstart - eller som bare gjelder én av Flys to
   maskiner - er ingen karantene.

Bade konto og IP telles (§3.1: «rate-limit ogsaa per IP/subnett»). CAPTCHA ved
terskel er ikke bygget; det krever en klientflate som ikke finnes.
"""
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..models import QuarantineScope, SearchQuarantine, utcnow

WINDOW = timedelta(days=1)
FIRST_OFFENSE = timedelta(days=1)
REPEAT_OFFENSE = timedelta(days=7)

# §3.1: telefonsoek har hard grense; ID-soek er lavere risiko (ID-en er ikke
# PII) og faar «samme prinsipp med mildere terskel».
PHONE_LIMIT = 5
ID_LIMIT = 20


def _row(s: OrmSession, scope: QuarantineScope, subject: str) -> SearchQuarantine | None:
    return s.scalar(select(SearchQuarantine).where(
        SearchQuarantine.scope == scope, SearchQuarantine.subject == subject))


def blocked_until(s: OrmSession, scope: QuarantineScope, subject: str):
    """Returnerer utloepstidspunktet hvis subjektet er i karantene, ellers None."""
    row = _row(s, scope, subject)
    if row is None or row.quarantined_until is None:
        return None
    return row.quarantined_until if row.quarantined_until > utcnow() else None


def register_failure(s: OrmSession, scope: QuarantineScope, subject: str,
                     limit: int) -> None:
    """Teller et mislykket soek og setter karantene naar grensen naas."""
    now = utcnow()
    row = _row(s, scope, subject)
    if row is None:
        row = SearchQuarantine(scope=scope, subject=subject,
                               window_started_at=now, failed_count=0)
        s.add(row)

    if now - row.window_started_at > WINDOW:
        row.window_started_at = now
        row.failed_count = 0

    row.failed_count += 1
    if row.failed_count >= limit:
        row.quarantined_until = now + (FIRST_OFFENSE if row.offense_count == 0
                                       else REPEAT_OFFENSE)
        row.offense_count += 1
        row.failed_count = 0
        row.window_started_at = now
    s.commit()
