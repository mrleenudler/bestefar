# Bestefar — Backend-spesifikasjon (utkast v0.1)

Underlag for backend som dekker funksjonene UI-laget (v0.6) bygger front-end for,
men som ikke kan fullføres uten konto + server. Bygger på `bestefar_CV-kjerne_spec.md`
§5–§6 og den eksisterende FastAPI-skissen i `backend/` (tre atskilte ansvar).

Status: **utkast** — feltdefinisjoner merket `TODO(eier)` der forskningsinnholdet
ikke er avklart.

## 0. Prinsipper (arvet fra spec)
- **Strukturell adskillelse:** treningsdata, feilanalyse-bilder og forskningsdata i
  separate tabeller/lagre. Forskning bruker pseudonym skytter-ID + samtykketabell.
- **Offline-først:** klienten fungerer uten backend; alt køes og synkes opportunistisk.
- **Samtykke styrer alt:** ingen deling uten eksplisitt, tilbaketrekkbart samtykke.

## 1. Konto og identitet
Trengs for backup, venner og lag.
- **Innlogging:** Google / Apple / e-post / telefonnummer (OAuth/OIDC + OTP for telefon).
- **Bruker-ID:** intern UUID. **Pseudonym forsknings-ID** avledet separat (ikke
  reversibelt koblet i forskningslageret).
- **Profil:** visningsnavn (≤ 24 tegn, latinske bokstaver inkl. æ/ø/å + tall/
  mellomrom/enkel tegnsetting — speiler klientens filter), fødselsår
  (egenrapportert), hjemkommune (valgfri), findable-flagg.
- **Endepunkter:** `POST /v1/auth/*`, `GET/PUT /v1/profile`.

## 2. Backup / dataoverføring (løser «mister loggen»)
Problem: appdata forsvinner ved avinstaller/reinstall uten konto.
- **Sync:** `PUT /v1/backup` (kryptert blob: serier + jaktlogg + innstillinger,
  klient-kryptert), `GET /v1/backup`. Konfliktløsning: last-write-wins per post-ID
  (postene har allerede UUID + `ts`).
- **«Flytt til ny telefon»:** kryptert eksportfil (klient) ELLER gjenoppretting fra
  konto-backup. Nøkkel avledet fra bruker-hemmelighet.
- **Android Auto Backup** dekker oppdateringer; konto-backup dekker reinstall/bytte.

## 3. Venner (front-end finnes i `VennerActivity`)
- **Modell:** `Friend { id, displayName, teamIds[], phone?, homeKommune?,
  shotsTotal?, shotsSeason?, avgScore?, trend?, kills[] }`. Klienten har `nickAlias`
  lokalt (redigert visningsnavn) — deles ikke.
- **Søk/legg til:** `GET /v1/users/search?q=` (kun findable-brukere), by bruker-ID,
  eller QR (QR = bruker-ID/redirect-URL). Vennskap krever **aksept** hos mottaker.
  `POST /v1/friends/request`, `POST /v1/friends/respond`.
- **Deling:** hver bruker velger felt som deles med venner (visningsnavn alltid).
  Server filtrerer utgående data etter delerens valg. Deaktivering nuller delte felt.
- **Telefonnummer delt** → klienten viser ring/SMS-ikon (håndteres klient-side når
  `phone` finnes).
- **Sensur av visningsnavn:** moderasjon server-side før navnet eksponeres for andre
  (regelsett + evt. manuell kø). Avvist navn deles ikke; brukeren varsles.

## 4. Lag (jaktlag/skytterlag)
- **Modell:** `Team { id, name, kind(jakt|skytter), memberCount, location?, leaders[] }`.
- **Nærliggende lag:** `GET /v1/teams/near?lat=&lon=&r=50000` → inntil 20, sortert
  etter avstand; de 3 nærmeste alltid med uansett avstand.
- **Oppretting + roller:** «jeg er leder» / «opprett for leder» / «be leder opprette».
  Flere ledere mulig.
- **Invitasjon:** ACTION_SEND-intent (ingen kontakttillatelse) med **redirect-URL** i
  EXTRA_TEXT; server leser User-Agent → riktig butikk (Play/App Store). Samme URL som
  QR for e-post/SMS-invitasjon. `POST /v1/teams/{id}/invite { emailOrPhone }` med
  validering (identifiser e-post vs telefon) og **leveringskvittering/-feil** tilbake.

## 5. Treningsresultater (utvidelse av `/v1/stats`)
- Per serie: `{ id, ts, weaponId, distanceM, position, modifier, shots[], corrected,
  seasonKey }`. Klienten køer usendte (uploaded=false).
- Brukes til venners delte statistikk (snitt/utvikling) og til brukerens egen backup.

## 6. Feilanalyse-bilder / OCR-donasjon (utvidelse av `/v1/failed-analyses`)
- Klienten køer i `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json` når brukeren har
  godtatt bildedeling (oppstartsvindu 2) eller per-innsending.
- **Endepunkt:** `POST /v1/failed-analyses` (multipart: bilde + JSON med detekterte
  poeng, OCR-poeng, `tag` ∈ {ocr_match, ocr_mismatch, rejected}).
- Formål: kalibrere OCR-heuristikken og CV-kjernen (bl.a. **over-deteksjon av treff**,
  se §8).

## 7. Forskningsdata (`/v1/research`, strukturelt adskilt)
- To resultattyper: **trening** og **jakt**, som separate modeller.
- Jakt-deling styres av brukerens valg: `{ vilt?, dato?, posisjon(grovhet)?,
  skuddsituasjon? }`. Skadedata private som standard.
- Samtykketabell: `{ pseudonymId, type(trening|jakt), granted_at, revoked_at? }`.
- `# TODO(eier): konkret feltinnhold for forskning ikke endelig avklart` (jf. kjerne-spec §6).

## 8. CV-kjerne-oppgaver (ikke backend, men avdekket i felt)
Notat til kjerne-repoet (versjonert bump av pinnen ved endring):
- **Over-deteksjon av treff** (flere merker enn reelt): undersøk om kamerabevegelse /
  multieksponering gir doble merker. Tiltak å vurdere: strammere auto-capture-
  stabilitet, og dedup av treff som ligger nærmere enn X ringavstander i `hits`/`overlap`.
- **OCR i kjernen:** UI-et bruker foreløpig ML Kit on-device. Vurder om skjerm-OCR bør
  flyttes til kjernen for konsistens og for å utnytte skjermens kjente layout.

## 9. Sikkerhet / personvern
- All PII kryptert i ro og i transitt. Forsknings-ID ikke reversibelt koblet til konto.
- Sletting: `DELETE /v1/account` (lokalt + sletteanmodning via pseudonym-ID for
  forskningslageret).

## 3.1 Bruker-ID og misbruksvern (musingsUI runde 5-spørsmål)
Designsvar på eierens spørsmål om venne-ID og søk:
- **Unik bruker-ID:** kort, håndskrivbar streng. Forslag: 8–10 tegn fra et
  forvekslingssikkert alfabet (Crockford base32: utelater I/L/O/U), f.eks.
  `BF-7Q4K-9F2M`. 9 base32-tegn ≈ 34 bit ≈ 68 mrd. ID-er — rikelig kapasitet for
  internasjonal spredning, samtidig som den er lett å lese opp/skrive ned. Vises
  også som QR (redirect-URL, jf. §4).
- **Feiltastingsvern:** ID-en har innebygd sjekksiffer (siste tegn), så åpenbare
  tastefeil avvises uten oppslag. Rommet er stort nok til at gjetting er
  upraktisk (forventet ~10⁴ reelle brukere mot 10¹⁰ mulige ID-er).
- **Søk etter bruker:** kun brukere med `findable=true` er søkbare.
  - Telefonsøk: hard rate-limit per konto/enhet — 5 mislykkede telefonsøk på én
    dag → karantene. Anbefaling: **1 dag** karantene ved første overtredelse,
    eskalerende til 7 dager ved gjentakelse. (Balanserer bruksvennlighet mot
    enumereringsangrep på telefonnumre, som er personsensitive.)
  - ID-søk: lavere risiko (ID er ikke PII), men samme prinsipp med mildere terskel.
  - **IP-heuristikk:** rate-limit også per IP/subnett for å stoppe automatiserte
    søk fra én kilde; CAPTCHA ved terskel. Logg kun aggregert, ikke per-søk-PII.
- **Personvern:** vennskap krever gjensidig aksept; ingen data deles før aksept.
  Telefonnummer deles kun hvis brukeren selv har krysset det av.

## 10. Direkte melding til utvikler (musingsUI runde 5)
Eier ønsket å slippe å åpne e-postapp. Krever backend:
- `POST /v1/feedback { subject, body, appVersion, deviceModel, userId? }` →
  serveren videresender til utviklerens innboks (eller sak-system).
- Klienten bruker foreløpig ACTION_SENDTO (mailto) med `subject` som e-post-Subject.
  Bytt til endepunktet når backend finnes.

## 11. Lag-medlemskap, lederskap og varsler (musingsUI runde 6)
UI-et (TeamPageActivity) bygger front-end for dette; alt reelt krever backend.
- **Medlemskap:** `Team.members[]`, `Team.leaders[]` (flere ledere mulig).
  Egen bruker vises alltid i lagets medlemsliste.
- **Invitasjoner:** «Inviter medlemmer» → kontaktliste/e-post/telefon (jf. §4).
- **Endre lagnavn:** varsel til alle medlemmer: «Navneendring — Lagleder for X har
  endret navnet på laget til Y». Vises som første melding ved neste app-åpning
  (klienten trenger en «pending messages»-kø hentet ved oppstart).
- **Fjern medlem:** varsel til den fjernede: «Lagleder har fjernet deg fra X».
- **Overfør lederskap:** velg eksisterende medlem → bekreftelse hos valgt leder.
- **Velg leder (ingen leder):** avstemning med 7-dagers nedtelling (vises som
  dager → timer < 24 → minutter < 120). Push til alle medlemmer. Stemmer kan
  endres til fristen; enstemmighet avslutter tidlig. `POST /v1/teams/{id}/vote`,
  `GET /v1/teams/{id}/vote-status`.
- **Fjern inaktiv lagleder:** hvis leder har brukt appen siste måned → «Lagleder
  er ikke inaktiv. Ta kontakt.» Ellers push til leder + 7-dagers timer; logger
  leder på, avbrytes; ellers mister leder lederstatus (forblir medlem), og laget
  kan velge ny leder.
- **Push-varsler:** krever FCM/APNs-registrering per enhet.
