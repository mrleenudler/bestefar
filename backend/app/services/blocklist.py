"""
Standard ordliste for moderasjon av visningsnavn (backend_spec §3).

Lista var tom fram til 2026-08-12 (B-37, B-43): en hardkodet ordliste er hverken
uttoemmende eller enkel aa vedlikeholde. Den er fortsatt ingen av delene - men
en tom liste fanger ingenting, og visningsnavnet er det ene feltet en bruker kan
skrive fritt til andre med. Dette er en fartsdump, ikke en loesning.
`DISPLAY_NAME_BLOCKLIST` legger ord TIL denne lista per miljoe.

## Regelen som avgjoer hva som kan staa her

`moderation.review` leter etter hvert ord som DELSTRENG i en foldet form av
navnet: smaa bokstaver, aksenter fjernet, alt som ikke er alfanumerisk strippet
bort. Foldingen er det som fanger «S-t-y-g-t» - og den er samtidig det som gjoer
korte ord farlige, for den fjerner ogsaa mellomrommene mellom fornavn og
etternavn. Et ord her maa derfor vaere langt nok og saerpreget nok til at det
ikke kan oppstaa inne i, eller paa tvers av, et ekte navn.

Kandidater som er VRAKET, og hvorfor - ikke legg dem inn igjen:

  hore    «Thoresen» folder til «thoresen». Vanlig norsk etternavn.
  fag     «Fagerli», «Fagermoen», og «fag» er et helt vanlig norsk ord.
  nazi    «Nazir», «Nazim», «Nazia» er ekte fornavn.
  sex     «Sexe» er et norsk etternavn (fra Voss).
  rape    «rape» er norsk for aa raape, og finnes i «Draper», «Grape».
  kkk     «Erik K. Kristiansen» folder til «erikkkristiansen».
  slut    «Slutten», og «Nils Lutken» folder til «nilslutken».
  dick    «Dickson», «Dickens».
  cock    «Hitchcock», «Cockburn».
  homo    «homofil» er et noeytralt ord, ikke et skjellsord.
  mongo   rammer «mongolsk» og folk fra Mongolia.

Og prinsippet bak vrakinga: **ord som beskriver hvem noen ER, staar ikke her.**
«jøde», «muslim», «same», «homofil», «innvandrer» er noeytrale ord. Det er
skjellsordene som blokkeres, ikke gruppene de rammer.

Norske tegn: foldingen dekomponerer «å» til «a», men lar «æ» og «ø» staa. Ord
med æ/ø trenger derfor en ASCII-variant ved siden av seg («ræva» + «raeva»),
ellers slipper den vanligste omskrivinga unna.

## Hva lista IKKE fanger

Ingen leet-normalisering («f4en»), ingen gjentatte bokstaver («fuuuck»), ingen
fonetiske varianter («phuck» er med, resten er ikke). Spec §3 aapner for en
manuell koe; den krever en admin-flate som ikke finnes ennaa.
"""

DEFAULT_BLOCKLIST: tuple[str, ...] = (
    # Grovt spraak, norsk
    "faen", "jævla", "jaevla", "jævel", "jaevel", "helvete", "satan",
    "dritt", "ræva", "raeva", "rasshøl", "rasshol", "rasshull",
    "kuk", "pikk", "fitte", "fitta", "knull", "ludder",
    # Grovt spraak, engelsk
    "fuck", "phuck", "shit", "bitch", "cunt", "asshole", "whore",
    # Seksualisert
    "porno", "penis", "voldtekt", "pedofil", "incest",
    # Hat og sjikane. Se docstringen: gruppenavn hoerer ikke hjemme her,
    # bare skjellsordene.
    "nigger", "nigga", "neger", "pakkis", "svartetryne",
    "faggot", "soper", "jødesvin", "jodesvin",
    "hitler", "heilhitler", "sieghail", "nazist", "nazisme", "hakekors",
    "1488", "retard",
)

# Unntak fra lista over. Et treff som ligger inne i et av disse uttrykkene,
# teller ikke - se `moderation.review`.
#
# «Anne Gerd» folder til «annegerd», som inneholder «neger». Navnet er vanlig
# nok til at det maa staa her fra dag én; det er ogsaa den beste illustrasjonen
# av hvorfor unntakslista finnes. Dukker det opp flere slike, settes de med
# `DISPLAY_NAME_ALLOWLIST` - en bruker som ikke faar bruke sitt eget navn skal
# ikke maatte vente paa en ny utrulling.
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "Anne Gerd",
    "Anne Gerda",
)
