import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.observability.tracing import setup_tracing, get_tracer, get_in_memory_exporter, instrument_app
from ai_service.tools.base import ToolDefinition, ToolResult, ToolDomain
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS
from ai_service.models.user_context import UserContext, UserRole

# Set up testing tracer and FastAPI app
tracing_test_app = FastAPI()
setup_tracing(service_name="test-service", environment="test")
instrument_app(tracing_test_app)

@tracing_test_app.get("/dummy-route")
def dummy_route():
    return {"status": "ok"}

@pytest.fixture
def test_client():
    return TestClient(tracing_test_app)

@pytest.fixture(autouse=True)
def clear_spans():
    exporter = get_in_memory_exporter()
    if exporter:
        exporter.clear()
    yield

def test_fastapi_request_generates_span(test_client):
    """Verify that a request to the FastAPI app generates spans."""
    response = test_client.get("/dummy-route")
    assert response.status_code == 200

    exporter = get_in_memory_exporter()
    assert exporter is not None
    spans = exporter.get_finished_spans()
    
    # We should have at least one span corresponding to the HTTP request
    assert len(spans) > 0
    http_spans = [s for s in spans if "/dummy-route" in s.name or s.name == "GET /dummy-route"]
    assert len(http_spans) > 0
    assert "dummy-route" in http_spans[0].name

def test_tool_execution_generates_span():
    """Verify that executing a tool creates a trace span with attributes."""
    # Define a mock tool
    async def mock_handler(db, user_id):
        return ToolResult(success=True, data="mock-data")

    tool_def = ToolDefinition(
        name="test_mock_tool",
        description="A test mock tool",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler
    )
    
    # Temporarily authorize tool for STUDENT role and register it
    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("test_mock_tool")
    ToolRegistry.register(tool_def)
    
    try:
        user_ctx = UserContext(
            user_id="student-1",
            university_id="20260001",
            full_name="Alice Student",
            role=UserRole.STUDENT
        )
        
        executor = ToolExecutor(db=None)
        
        import asyncio
        result = asyncio.run(executor.execute("test_mock_tool", {}, user_ctx))
        assert result.success is True

        exporter = get_in_memory_exporter()
        spans = exporter.get_finished_spans()
        
        tool_spans = [s for s in spans if s.name == "tool_execution"]
        assert len(tool_spans) == 1
        
        span = tool_spans[0]
        assert span.attributes["tool.name"] == "test_mock_tool"
        assert span.attributes["user.id"] == "student-1"
        assert span.attributes["user.role"] == "STUDENT"
        assert span.attributes["tool.success"] is True
    finally:
        # Clean up registry and permissions
        if "test_mock_tool" in ToolRegistry._tools:
            del ToolRegistry._tools["test_mock_tool"]
        ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("test_mock_tool")

def test_provider_chat_generates_span(monkeypatch):
    """Verify that calling LLMProvider creates trace spans."""
    from ai_service.providers.openrouter import OpenRouterProvider
    
    provider = OpenRouterProvider()
    
    # Mock httpx AsyncClient post to avoid actual HTTP calls during testing
    import httpx
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "Hello! I am OpenRouter.",
                        "role": "assistant"
                    }
                }]
            }
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    import asyncio
    res = asyncio.run(provider.complete("Who are you?", max_tokens=10))
    assert res == "Hello! I am OpenRouter."

    exporter = get_in_memory_exporter()
    spans = exporter.get_finished_spans()
    
    provider_spans = [s for s in spans if s.name == "llm_provider_complete"]
    assert len(provider_spans) == 1
    assert provider_spans[0].attributes["provider.name"] == "openrouter"
    assert provider_spans[0].attributes["provider.model"] == OpenRouterProvider.MODEL
