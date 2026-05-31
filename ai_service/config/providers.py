from ai_service.config.settings import settings

PROVIDER_CONFIG = {
    "openrouter": {
        "model": "z-ai/glm-4.5-air:free",
        "api_key": settings.OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "timeout_seconds": 30.0,
        "max_retries": 2,
        "supports_tools": True,
        "supports_streaming": True,
    }
}
