from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "trade"
    DB_USER: str = "trade"
    DB_PASSWORD: str = "trade_secret"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Data Collection
    START_DATE: str = "2010-01-01"
    YFINANCE_DELAY: float = 0.3
    BATCH_SIZE: int = 50

    # Notifications
    SLACK_WEBHOOK_URL: Optional[str] = None

    # External APIs
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def db_url_async(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()
