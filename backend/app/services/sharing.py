"""
Utgaaende filtrering av vennedata (backend_spec §3).

Prinsippet: SERVEREN filtrerer etter delerens valg. Klienten skal aldri motta
et felt den ikke har lov til aa vise - da er «deaktivering nuller delte felt»
en garanti og ikke en klientdetalj som en modifisert app kan omgaa.

Visningsnavn deles alltid (§3) - men bare naar moderasjonen har godkjent det
(§3, sensur). Er navnet avvist eller til vurdering, eksponeres en noeytral
plassholder i stedet.
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from ..models import NameStatus, Series, SharingPreference, TeamMember, User

# Trendvinduet telles i SKUDD, ikke serier: en serie er 5-10 skudd, saa fem
# serier kunne bety alt fra 25 til 50 skudd og gjorde tallet uleselig paa tvers
# av brukere. Tjue skudd er stort nok til aa daempe stoey og lite nok til aa
# vaere ferskt.
TREND_SHOTS = 20
NAVN_UNDER_VURDERING = "Ukjent skytter"


def _totals(s: OrmSession, user_id: str) -> tuple[int, float]:
    shots, points = s.execute(
        select(func.coalesce(func.sum(Series.shot_count), 0),
               func.coalesce(func.sum(Series.sum_decimal), 0.0))
        .where(Series.user_id == user_id)).one()
    return int(shots), float(points)


def _season_shots(s: OrmSession, user_id: str) -> int:
    """Inneavaerende sesong = sesongnoekkelen paa den nyeste serien. Klienten
    eier definisjonen av en sesong (`season_key`); serveren teller bare."""
    season = s.scalar(select(Series.season_key).where(Series.user_id == user_id)
                      .order_by(Series.ts.desc()).limit(1))
    if not season:
        return 0
    return int(s.scalar(
        select(func.coalesce(func.sum(Series.shot_count), 0))
        .where(Series.user_id == user_id, Series.season_key == season)) or 0)


def _vindu(rader: list, start: int) -> tuple[float, int, int]:
    """
    Samler hele serier fra og med `start` til vinduet har minst TREND_SHOTS
    skudd. Returnerer (sum poeng, sum skudd, neste indeks).

    Hele serier med vilje: en serie er én oekt under de samme forholdene, og
    aa kutte den paa midten for aa treffe akkurat tjue skudd ville blandet to
    oekter i samme tall.
    """
    poeng, skudd = 0.0, 0
    i = start
    while i < len(rader) and skudd < TREND_SHOTS:
        poeng += rader[i][0]
        skudd += rader[i][1]
        i += 1
    return poeng, skudd, i


def _trend(s: OrmSession, user_id: str) -> float | None:
    """
    Utvikling: snitt per skudd i de siste ~20 skuddene minus de ~20
    foregaaende. Positivt tall = framgang.

    Vinduet telles i SKUDD og ikke i serier - se TREND_SHOTS. Returnerer None
    foer begge vinduene er fulle: et «trendtall» fra to serier ville vaert
    stoey presentert som innsikt.

    NB: dette er en TREND (en differanse, altsaa en retning), ikke et
    loepende snitt. Skal klienten vise nivaaet over de siste tjue skuddene, er
    det `avg_score` med et vindu - et annet felt og et annet delingsvalg.
    """
    rader = list(s.execute(
        select(Series.sum_decimal, Series.shot_count)
        .where(Series.user_id == user_id, Series.shot_count > 0)
        # En serie har minst ett skudd, saa 2 x TREND_SHOTS serier daekker
        # begge vinduene i verste fall. Uten grensen ville en bruker med tusen
        # serier lastet alle sammen for aa regne ut ett tall.
        .order_by(Series.ts.desc()).limit(TREND_SHOTS * 2)))

    nye_poeng, nye_skudd, i = _vindu(rader, 0)
    if nye_skudd < TREND_SHOTS:
        return None
    gamle_poeng, gamle_skudd, _ = _vindu(rader, i)
    if gamle_skudd < TREND_SHOTS:
        return None

    return round(nye_poeng / nye_skudd - gamle_poeng / gamle_skudd, 2)


def _team_ids(s: OrmSession, user_id: str) -> list[str]:
    return list(s.scalars(select(TeamMember.team_id)
                          .where(TeamMember.user_id == user_id)))


def friend_view(s: OrmSession, sharer: User,
                prefs: SharingPreference | None) -> dict:
    """Bygger den utgaaende representasjonen av `sharer`, filtrert paa
    hans/hennes egne delingsvalg."""
    navn = (sharer.display_name
            if sharer.display_name_status == NameStatus.approved
            else NAVN_UNDER_VURDERING)
    ut: dict = {
        "id": sharer.id,
        "public_id": sharer.public_id,
        "display_name": navn,
        "team_ids": _team_ids(s, sharer.id),
    }
    if prefs is None:
        return ut

    if prefs.share_phone and sharer.phone:
        # Klienten viser ring/SMS-ikon naar dette feltet finnes (§3).
        ut["phone"] = sharer.phone
    if prefs.share_home_kommune:
        ut["home_kommune"] = sharer.home_kommune

    # Regn bare ut det som faktisk skal deles.
    if prefs.share_shots_total or prefs.share_avg_score:
        shots, points = _totals(s, sharer.id)
        if prefs.share_shots_total:
            ut["shots_total"] = shots
        if prefs.share_avg_score:
            ut["avg_score"] = round(points / shots, 2) if shots else None
    if prefs.share_shots_season:
        ut["shots_season"] = _season_shots(s, sharer.id)
    if prefs.share_trend:
        ut["trend"] = _trend(s, sharer.id)

    # `kills` fra §3-modellen mangler med hensikt: jaktloggen ligger inne i den
    # klient-krypterte backup-bloben (§2), saa serveren kan ikke lese den. Skal
    # felte dyr deles med venner, maa jaktposter synkes som egne rader - en
    # spec-avklaring, ikke en implementasjonsdetalj.
    return ut
