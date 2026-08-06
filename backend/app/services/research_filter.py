"""
Filtrering av forskningsdata etter brukerens valg (backend_spec §7).

§7 sier at jakt-deling styres av brukerens valg - `{ vilt?, dato?,
posisjon(grovhet)?, skuddsituasjon?, skadedata? }` - og at skadedata er private
som standard. Dette laget haandhever det.

TO PRINSIPPER, begge bevisste:

  1. FILTRERINGEN SKJER PAA SERVEREN. Klienten filtrerer ogsaa, men det er
     bekvemmelighet, ikke vern: en modifisert klient kan sende hva som helst.
     Samme resonnement som moderation.py og §3-delingsvalgene.

  2. JAKTDATA FOELGER EN TILLATELSESLISTE, ikke en forbudsliste. Feltinnholdet
     er ikke endelig avklart (TODO(eier), kjerne-spec §6), og med en
     forbudsliste ville hvert nytt felt vaert delt som standard til noen kom
     paa aa forby det. Ukjente noekler forsvinner i stedet stille. Naar
     feltlista er klar, er det HER den utvides.

Treningsdata gaar uendret gjennom: §7 gir ingen felt-for-felt-valg for dem,
samtykket gjelder hele resultattypen.
"""
from datetime import datetime

from ..models import PositionGranularity, ResearchSharingPreference, ResultType

# Kanoniske noekler. Klienten maa bruke akkurat disse; alt annet i en
# jakt-payload droppes.
ART = ("species",)
SKUDDSITUASJON = ("shot_situation", "shooting_position", "position_modifier",
                  "distance_m", "rest_used")
POSISJON_EKSAKT = ("lat", "lon")
POSISJON_KOMMUNE = ("kommune",)
POSISJON_FYLKE = ("fylke",)

# Skadedata: gikk fra «aldri» til egen bryter (share_injury_data), fordi
# ettersoeksdata er den mest verdifulle delen av materialet - hvor ofte dyr
# skadeskytes, hvor langt de gaar, og om de blir funnet. §7 sier «private som
# standard», og det er standardverdien False som oppfyller: det kreves et
# aktivt valg, og bryteren staar for seg selv, ikke sammen med art og sted.
SKADEDATA = ("wounded", "injury", "hit_placement", "shots_fired",
             "tracking_distance_m", "tracking_time_min", "dog_used",
             "recovered")


def _tillatte_noekler(pref: ResearchSharingPreference) -> set[str]:
    lov: set[str] = set()
    if pref.share_species:
        lov.update(ART)
    if pref.share_shot_situation:
        lov.update(SKUDDSITUASJON)
    if pref.share_injury_data:
        lov.update(SKADEDATA)

    # Grovheten bestemmer HVILKE stedsfelt som slipper gjennom, ikke hvor mye
    # vi avrunder dem. Serveren har ingen kommune- eller fylkesgrenser aa slaa
    # opp i, og en avrunding av koordinater ville uansett ikke vaert det
    # brukeren valgte - «kommune» er et navn, ikke et antall desimaler.
    # Klienten kjenner stedet og sender navnet; vi haandhever hva som lagres.
    if pref.position_granularity == PositionGranularity.exact:
        lov.update(POSISJON_EKSAKT)
        lov.update(POSISJON_KOMMUNE)
        lov.update(POSISJON_FYLKE)
    elif pref.position_granularity == PositionGranularity.kommune:
        lov.update(POSISJON_KOMMUNE)
        lov.update(POSISJON_FYLKE)
    elif pref.position_granularity == PositionGranularity.fylke:
        lov.update(POSISJON_FYLKE)
    return lov


def filtrer_payload(result_type: ResultType, payload: dict,
                    pref: ResearchSharingPreference) -> dict:
    if result_type != ResultType.hunt:
        return dict(payload)
    lov = _tillatte_noekler(pref)
    return {k: v for k, v in payload.items() if k in lov}


def filtrer_tidspunkt(result_type: ResultType, captured_at: datetime,
                      pref: ResearchSharingPreference) -> datetime:
    """
    Uten datosamtykke beholdes bare AARET (1. januar).

    Kolonnen er obligatorisk, saa alternativet til aa grovkorne var aa avvise
    hele innsendingen. Aaret alene sier ikke naar noen var paa jakt, men lar
    materialet fortsatt grupperes per sesong - som er hele poenget med aa ta
    imot raden.
    """
    if result_type != ResultType.hunt or pref.share_date:
        return captured_at
    return captured_at.replace(month=1, day=1, hour=0, minute=0, second=0,
                               microsecond=0)
