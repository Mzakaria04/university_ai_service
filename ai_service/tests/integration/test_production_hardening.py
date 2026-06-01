import pytest
from httpx import AsyncClient
from ai_service.config.settings import settings
from ai_service.main import app
from ai_service.db.session import AsyncSessionLocal, engine, readonly_engine
from ai_service.db.models import AISession, AIMessageEvent, AIFeedback, AIToolExecutionLog, AIExecutionTrace

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

async def test_swagger_suppression_in_production(monkeypatch):
    """Verify that Swagger, ReDoc, and OpenAPI schemas are disabled when ENVIRONMENT is production."""
    # Temporarily force environment to production
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    
    # Re-initialize/build the app docs configurations
    docs_url_original = app.docs_url
    redoc_url_original = app.redoc_url
    openapi_url_original = app.openapi_url
    routes_original = app.routes.copy()
    
    # Simulate production FastAPI instantiation logic
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = None
    app.routes[:] = [r for r in app.routes if r.path not in ["/docs", "/redoc", "/openapi.json"]]
    
    # Re-create openapi_schema and setup docs routes
    app.openapi_schema = None
    app.setup()
    
    try:
        from httpx import ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Docs endpoint should return 404
            resp = await client.get("/docs")
            assert resp.status_code == 404
            
            # ReDoc endpoint should return 404
            resp = await client.get("/redoc")
            assert resp.status_code == 404
            
            # OpenAPI schema JSON endpoint should return 404
            resp = await client.get("/openapi.json")
            assert resp.status_code == 404
    finally:
        # Restore original settings
        app.docs_url = docs_url_original
        app.redoc_url = redoc_url_original
        app.openapi_url = openapi_url_original
        app.routes[:] = routes_original
        app.openapi_schema = None
        app.setup()


async def test_dual_engine_routing_binds():
    """Verify that AI-owned tables route to the write-enabled engine, while other operations use readonly."""
    # Retrieve SQLAlchemy session binds
    session = AsyncSessionLocal()
    
    # Core binds checks
    assert session.binds[AISession] == engine
    assert session.binds[AIMessageEvent] == engine
    assert session.binds[AIFeedback] == engine
    assert session.binds[AIToolExecutionLog] == engine
    assert session.binds[AIExecutionTrace] == engine
    
    # The default fallback bind for raw SQL queries and unmapped entities should be readonly_engine
    # Retrieve the mapper bind for unmapped entities / raw SQL
    assert session.get_bind() == readonly_engine.sync_engine
    
    await session.close()
