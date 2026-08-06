"""
Konfigurasjon fra miljoevariabler (backend_spec §0.1: secrets aldri i repoet).
Lokalt kan verdiene ligge i backend/.env - se .env.example. I produksjon settes
de som Fly secrets (`flyctl secrets set ...`).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")

    # --- Miljoe ---
    env: str = "dev"                      # dev | prod
    log_level: str = "INFO"

    # --- Database ---
    # Supabase gir en "Direct connection string" paa formen
    #   postgresql://postgres:<passord>@db.<ref>.supabase.co:5432/postgres
    # Den normaliseres til psycopg3-driveren i db.py.
    database_url: str = "sqlite:///./bestefar.db"
    # Skjemaet lages av Alembic fra fase 2. create_all() beholdes kun som
    # bekvemmelighet lokalt, og skal vaere AV i produksjon.
    auto_create_tables: bool = True

    # --- Lag og invitasjoner (§4) ---
    # Invitasjons-URL-en er den samme som QR-koden peker paa. Serveren leser
    # User-Agent og sender videre til riktig butikk.
    invite_base_url: str = "https://bestefar-api.fly.dev/i"
    play_store_url: str = ("https://play.google.com/store/apps/details"
                           "?id=no.bestefar.app")
    # iOS er ikke publisert ennaa; peker til Play-siden inntil videre.
    app_store_url: str = ("https://play.google.com/store/apps/details"
                          "?id=no.bestefar.app")

    # --- Innlogging (§1) ---
    # Noekkelen vaare EGNE tokens signeres med (HS256). Tom => alle
    # /v1/auth/*-endepunktene svarer 503: uten den kan vi ikke utstede noe som
    # helst, og aa falle tilbake paa en standardverdi ville gjort tokens
    # forfalskbare av hvem som helst som leser repoet.
    jwt_secret: str = ""
    # Kort levetid paa access-tokenet fordi det ikke kan tilbakekalles - det
    # verifiseres med signatur alene, uten oppslag i basen. Refresh-tokenet er
    # ugjenkjennelig for oss (vi lagrer bare hashen) og KAN tilbakekalles.
    access_token_ttl_minutes: int = 60
    refresh_token_ttl_days: int = 90

    # Gyldige `aud`-verdier fra leverandoerene, kommaseparert.
    # Tom liste => den leverandoeren svarer 503.
    #
    # GOOGLE: dette er WEB-klient-ID-en, ikke Android-klient-ID-en. Android
    # trenger sin egen klient (for SHA-1-bindingen), men Credential Manager
    # utsteder ID-tokens med web-klienten som `aud` - den samme verdien
    # klienten sender til setServerClientId(). Legger man inn Android-ID-en
    # her, avvises hvert eneste token med «Ugyldig Google-token».
    google_client_ids: str = ""
    apple_client_ids: str = ""          # bundle-ID / services-ID

    # E-postinnlogging: engangskode. Seks siffer er nok naar koden er
    # kortlivet, har faa forsoek og er ratebegrenset.
    email_code_ttl_minutes: int = 15
    email_code_max_attempts: int = 5
    # Per e-postadresse. Hevet fra 5 til 10: grensen teller FORESPOERSLER, ikke
    # leveranser, saa en bruker som ber om ny kode fordi e-posten er treg
    # brenner opp kvoten sin uten aa ha gjort noe galt. Ti forsoek er fortsatt
    # langt under det som trengs for aa gjette en sekssifret kode, som uansett
    # er vernet av email_code_max_attempts.
    email_code_rate_per_hour: int = 10

    @property
    def google_client_id_list(self) -> list[str]:
        return [s.strip() for s in self.google_client_ids.split(",") if s.strip()]

    @property
    def apple_client_id_list(self) -> list[str]:
        return [s.strip() for s in self.apple_client_ids.split(",") if s.strip()]

    # --- Push (§11, fase 8) ---
    # Tjenestekontoen fra Firebase, som RÅ JSON i én miljoevariabel. Tom =>
    # push logges bare, og meldingskoeen alene baerer varselet. Prosjekt-ID-en
    # leses fra JSON-en om den ikke settes eksplisitt.
    fcm_service_account_json: str = ""
    fcm_project_id: str = ""
    # Per HTTP-kall mot FCM. Kort med vilje: kallet skjer inne i
    # forespoerselen, og et tregt varsel skal ikke henge en lagoperasjon.
    push_timeout_seconds: float = 5.0
    # Samlet budsjett for én varsling. FCM HTTP v1 tar én mottaker per kall,
    # saa et stort lag betyr mange kall. Naar budsjettet er brukt opp, droppes
    # resten - meldingskoeen er garantien, push er bare det raske varselet.
    push_budget_seconds: float = 6.0

    # --- Moderasjon av visningsnavn (§3) ---
    # Kommaseparert. Tom som standard: en hardkodet norsk banneordliste ville
    # vaert baade ufullstendig og umulig aa vedlikeholde fra repoet.
    display_name_blocklist: str = ""

    @property
    def display_name_blocklist_list(self) -> list[str]:
        return [w.strip() for w in self.display_name_blocklist.split(",") if w.strip()]

    # --- Opplasting (§6) ---
    max_upload_bytes: int = 8 * 1024 * 1024

    # --- Backup (§2) ---
    # Bloben er serier + jaktlogg + innstillinger, komprimert og kryptert av
    # klienten. 16 MB er rikelig for mange aars logg og holder minnebruken nede
    # (hele bloben leses inn i minnet ved opp- og nedlasting).
    max_backup_bytes: int = 16 * 1024 * 1024

    # --- Forskning (§7) ---
    # Hemmelighet som forsknings-pseudonymet avledes med. Uten den svarer
    # /v1/research 503 - vi skal ikke lagre forskningsdata uten pseudonymisering.
    # Roteres ALDRI uten en plan for eksisterende forskningsdata.
    research_pseudonym_secret: str = ""
    # §7/§9: skal staa av til personvernerklaering og DPIA-vurdering er paa plass.
    research_enabled: bool = False

    # --- Feedback (§10) ---
    feedback_to: str = ""                 # utviklerens innboks; tom => kun logg
    feedback_from: str = "bestefar@bestefar.app"
    feedback_rate_per_hour: int = 5       # per IP
    resend_api_key: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in ("prod", "production")


@lru_cache
def settings() -> Settings:
    return Settings()
