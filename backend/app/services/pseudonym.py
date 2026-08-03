"""
Pseudonym forsknings-ID (backend_spec §1, §7).

ID-en avledes deterministisk fra brukerens interne UUID med en
server-hemmelighet: HMAC-SHA256(hemmelighet, user_id).

Hvorfor HMAC og ikke en oppslagstabell: en tabell VILLE vaert en reversibel
kobling mellom konto og forskningsdata - nettopp det §7 forbyr. Med HMAC
inneholder forskningsskjemaet kun pseudonymet, og veien tilbake finnes ikke
uten hemmeligheten (som ligger i Fly secrets, ikke i databasen).

Konsekvens som maa vaere bevisst: byttes hemmeligheten, mister eksisterende
forskningsdata koblingen til nye innsendinger fra samme bruker. Hemmeligheten
skal derfor aldri roteres uten en plan for de dataene.
"""
import base64
import hashlib
import hmac

from ..config import Settings


class PseudonymNotConfigured(RuntimeError):
    pass


def for_user(cfg: Settings, user_id: str) -> str:
    if not cfg.research_pseudonym_secret:
        raise PseudonymNotConfigured(
            "RESEARCH_PSEUDONYM_SECRET er ikke satt - forskningsdata kan ikke "
            "pseudonymiseres, og endepunktet skal derfor ikke svare.")
    digest = hmac.new(cfg.research_pseudonym_secret.encode("utf-8"),
                      user_id.encode("utf-8"), hashlib.sha256).digest()
    # 160 bit er rikelig; base32 uten padding er trygt i URL-er og logger.
    return base64.b32encode(digest[:20]).decode("ascii").rstrip("=")
