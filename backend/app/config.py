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
