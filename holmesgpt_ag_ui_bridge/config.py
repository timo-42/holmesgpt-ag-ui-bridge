from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BRIDGE_", env_file=".env", extra="ignore")

    holmes_base_url: str = Field(default="http://localhost:8080")
    holmes_api_key: str | None = Field(default=None)
    request_timeout_seconds: float = Field(default=300.0)
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def chat_url(self) -> str:
        return f"{self.holmes_base_url.rstrip('/')}/api/chat"

    @property
    def model_url(self) -> str:
        return f"{self.holmes_base_url.rstrip('/')}/api/model"

    @property
    def health_url(self) -> str:
        return f"{self.holmes_base_url.rstrip('/')}/healthz"
