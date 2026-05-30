from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    ENVIRONMENT: str = Field(default="development")
    
    # Database configuration
    DATABASE_URL: str = Field(default="postgresql://postgres:1234@localhost:5432/edux_db")
    
    # JWT authentication configuration
    JWT_SECRET: str = Field(default="super-secret-key")
    
    # LLM Provider configuration
    OPENROUTER_API_KEY: str = Field(default="mock-or-key")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def async_database_url(self) -> str:
        """
        Converts the standard postgresql:// URL to postgresql+asyncpg:// for SQLAlchemy async engine,
        and ensures any query parameters like ?schema=public are handled properly or stripped if not supported by asyncpg.
        """
        url = self.DATABASE_URL
        # Remove schema=public query parameter as it causes issues for some postgres engines/drivers in asyncpg
        if "?" in url:
            base_url, query = url.split("?", 1)
            # Filter out schema=public if present, keeping others, or just clean it up
            params = [p for p in query.split("&") if not p.startswith("schema=")]
            url = f"{base_url}?{'&'.join(params)}" if params else base_url
            
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url

settings = Settings()
