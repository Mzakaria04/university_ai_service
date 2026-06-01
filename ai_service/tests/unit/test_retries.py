import pytest
import asyncio
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from ai_service.tools.base import ToolDefinition, ToolResult, ToolDomain
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS
from ai_service.models.user_context import UserContext, UserRole
from ai_service.errors import ToolAuthorizationError, ToolTimeoutError, ToolExecutionError
from ai_service.errors import ProviderTimeoutError, ProviderUnavailableError
from ai_service.providers.openrouter import OpenRouterProvider

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

async def test_tool_executor_retries_transient_failures():
    """Verify that the ToolExecutor retries transient exceptions up to max_retries."""
    calls = 0

    async def mock_handler(db, user_id):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("Transient error")
        return ToolResult(success=True, data="retry-success")

    tool_def = ToolDefinition(
        name="test_retry_tool",
        description="A tool that fails transiently",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler,
        max_retries=2
    )

    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("test_retry_tool")
    ToolRegistry.register(tool_def)

    try:
        user_ctx = UserContext(
            user_id="student-1",
            university_id="20260001",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        executor = ToolExecutor(db=None)
        
        result = await executor.execute("test_retry_tool", {}, user_ctx)
        assert result.success is True
        assert result.data == "retry-success"
        assert calls == 3  # Fails twice, succeeds on the 3rd attempt
    finally:
        if "test_retry_tool" in ToolRegistry._tools:
            del ToolRegistry._tools["test_retry_tool"]
        ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("test_retry_tool")

async def test_tool_executor_propagates_authorization_error():
    """Verify that ToolExecutor does not retry AuthorizationError."""
    calls = 0
    from ai_service.errors import AuthorizationError

    async def mock_handler(db, user_id):
        nonlocal calls
        calls += 1
        raise AuthorizationError("Access denied")

    tool_def = ToolDefinition(
        name="test_auth_retry_tool",
        description="A tool that raises AuthorizationError",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler,
        max_retries=2
    )

    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("test_auth_retry_tool")
    ToolRegistry.register(tool_def)

    try:
        user_ctx = UserContext(
            user_id="student-1",
            university_id="20260001",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        executor = ToolExecutor(db=None)
        
        with pytest.raises(AuthorizationError):
            await executor.execute("test_auth_retry_tool", {}, user_ctx)
        assert calls == 1  # Should only run once
    finally:
        if "test_auth_retry_tool" in ToolRegistry._tools:
            del ToolRegistry._tools["test_auth_retry_tool"]
        ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("test_auth_retry_tool")

async def test_tool_executor_timeout():
    """Verify that ToolExecutor raises ToolTimeoutError when execution exceeds timeout."""
    async def mock_handler(db, user_id):
        await asyncio.sleep(2.0)
        return ToolResult(success=True, data="should-timeout")

    tool_def = ToolDefinition(
        name="test_timeout_tool",
        description="A tool that hangs",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler,
        timeout_seconds=0.1,
        max_retries=0
    )

    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("test_timeout_tool")
    ToolRegistry.register(tool_def)

    try:
        user_ctx = UserContext(
            user_id="student-1",
            university_id="20260001",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        executor = ToolExecutor(db=None)
        
        result = await executor.execute("test_timeout_tool", {}, user_ctx)
        assert result.success is False
        assert "timed out" in result.error_message
    finally:
        if "test_timeout_tool" in ToolRegistry._tools:
            del ToolRegistry._tools["test_timeout_tool"]
        ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("test_timeout_tool")

async def test_provider_retries_transient_http_errors():
    """Verify that OpenRouterProvider retries transient HTTP errors up to 3 times."""
    provider = OpenRouterProvider()
    
    mock_post = AsyncMock()
    mock_post.side_effect = [
        httpx.TimeoutException("Timeout 1"),
        httpx.TimeoutException("Timeout 2"),
        MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Retry success"}}]})
    ]

    with patch("httpx.AsyncClient.post", mock_post):
        result = await provider.complete("hello")
        assert result == "Retry success"
        assert mock_post.call_count == 3
