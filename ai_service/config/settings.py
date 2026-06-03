from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    ENVIRONMENT: str = Field(default="development")
    
    # Database configuration
    DATABASE_URL: str = Field(default="postgresql://postgres:1234@localhost:5432/edux_db")
    DB_READONLY_USER: str = Field(default="postgres")
    DB_READONLY_PASSWORD: str = Field(default="1234")
    
    # JWT authentication configuration
    JWT_SECRET: str = Field(default="super-secret-key")
    
    # Redis configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    
    # LLM Provider configuration
    OPENROUTER_API_KEY: str = Field(default="mock-or-key")
    GROQ_API_KEY: str = Field(default="mock-groq-key")
    
    # Internal Debug API configuration
    INTERNAL_API_KEY: str = Field(default="internal-debug-key")
    
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

    @property
    def async_readonly_database_url(self) -> str:
        """
        Reconstructs the DATABASE_URL with DB_READONLY_USER and DB_READONLY_PASSWORD
        and ensures async driver prefixes are applied.
        """
        url = self.DATABASE_URL
        if "@" in url:
            prefix, rest = url.split("://", 1)
            credentials, host_db = rest.split("@", 1)
            rebuilt_url = f"{prefix}://{self.DB_READONLY_USER}:{self.DB_READONLY_PASSWORD}@{host_db}"
        else:
            rebuilt_url = url
            
        if "?" in rebuilt_url:
            base_url, query = rebuilt_url.split("?", 1)
            params = [p for p in query.split("&") if not p.startswith("schema=")]
            rebuilt_url = f"{base_url}?{'&'.join(params)}" if params else base_url
            
        if rebuilt_url.startswith("postgresql://"):
            return rebuilt_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif rebuilt_url.startswith("postgres://"):
            return rebuilt_url.replace("postgres://", "postgresql+asyncpg://", 1)
        return rebuilt_url

settings = Settings()
