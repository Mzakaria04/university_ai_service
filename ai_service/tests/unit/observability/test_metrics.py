import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_fastapi_instrumentator import Instrumentator

from ai_service.tools.base import ToolDefinition, ToolResult, ToolDomain
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS
from ai_service.models.user_context import UserContext, UserRole
from ai_service.observability.metrics import ai_tool_calls_total, ai_tokens_total

# Set up test app with Prometheus instrumentator
metrics_test_app = FastAPI()
Instrumentator().instrument(metrics_test_app).expose(metrics_test_app)

@metrics_test_app.get("/dummy-metrics-route")
def dummy_route():
    return {"status": "ok"}

@pytest.fixture
def metrics_client():
    return TestClient(metrics_test_app)

def test_http_request_increments_prometheus_metrics(metrics_client):
    """Verify that hitting endpoints generates Prometheus-formatted metrics."""
    response = metrics_client.get("/dummy-metrics-route")
    assert response.status_code == 200

    metrics_response = metrics_client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    
    assert "http_requests_total" in metrics_text or "http_request_duration_seconds" in metrics_text

def test_tool_execution_increments_custom_metrics(metrics_client):
    """Verify that executing a tool increments custom Prometheus counters."""
    async def mock_handler(db, user_id):
        return ToolResult(success=True, data="metrics-data")

    tool_def = ToolDefinition(
        name="test_metrics_tool",
        description="A metrics test mock tool",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler
    )
    
    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("test_metrics_tool")
    ToolRegistry.register(tool_def)
    
    try:
        user_ctx = UserContext(
            user_id="student-metrics-1",
            university_id="20260002",
            full_name="Bob Metrics",
            role=UserRole.STUDENT
        )
        
        executor = ToolExecutor(db=None)
        
        import asyncio
        # Get baseline values from Prometheus registry/labels
        try:
            initial_count = ai_tool_calls_total.labels(tool_name="test_metrics_tool", success="true")._value.get()
        except Exception:
            initial_count = 0
            
        result = asyncio.run(executor.execute("test_metrics_tool", {}, user_ctx))
        assert result.success is True
        
        # Verify metric increment
        final_count = ai_tool_calls_total.labels(tool_name="test_metrics_tool", success="true")._value.get()
        assert final_count == initial_count + 1
        
        # Verify exposed on /metrics endpoint
        metrics_response = metrics_client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_text = metrics_response.text
        assert 'ai_tool_calls_total{success="true",tool_name="test_metrics_tool"}' in metrics_text
        assert 'ai_tool_latency_seconds_count{tool_name="test_metrics_tool"}' in metrics_text
        
    finally:
        if "test_metrics_tool" in ToolRegistry._tools:
            del ToolRegistry._tools["test_metrics_tool"]
        ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("test_metrics_tool")
