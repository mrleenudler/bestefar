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
- **Profil:** visningsnavn (≤ 24 tegn, utskrivbar ASCII — speiler klientens filter),
  fødselsår (egenrapportert), hjemkommune (valgfri), findable-flagg.
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
