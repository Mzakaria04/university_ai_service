import pytest
import json
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from ai_service.providers.openrouter import OpenRouterProvider
from ai_service.models.messages import Message
from ai_service.tools.base import ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole
from ai_service.errors import ProviderTimeoutError, ProviderUnavailableError, ProviderRateLimitError

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"


def mock_response(status_code: int, json_data: dict = None, content: bytes = b"") -> httpx.Response:
    """Helper to create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
        resp.content = json.dumps(json_data).encode("utf-8")
    else:
        resp.content = content
    
    # Mock status check raising HTTPStatusError on error codes
    def raise_for_status():
        if 400 <= status_code < 600:
            raise httpx.HTTPStatusError(message="HTTP Error", request=MagicMock(), response=resp)
    resp.raise_for_status.side_effect = raise_for_status
    return resp


async def test_complete_success():
    """Verify that complete() returns text response successfully."""
    provider = OpenRouterProvider()
    mock_resp = mock_response(200, {"choices": [{"message": {"content": "Summary text"}}]})
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        
        result = await provider.complete("Summarize this")
        assert result == "Summary text"
        mock_post.assert_called_once()
        # Verify model and payload
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "thudm/glm-4.5-air"
        assert kwargs["json"]["messages"][0]["content"] == "Summarize this"


async def test_chat_static_text_success():
    """Verify that non-streaming chat yields text content."""
    provider = OpenRouterProvider()
    mock_resp = mock_response(200, {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello student!"
            }
        }]
    })
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        
        response = await provider.chat([Message(role="user", content="Hi")], stream=False)
        assert response.content == "Hello student!"
        assert len(response.tool_calls) == 0


async def test_chat_static_tool_call_success():
    """Verify that non-streaming chat parses tool calls correctly."""
    provider = OpenRouterProvider()
    mock_resp = mock_response(200, {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_gpa_123",
                    "type": "function",
                    "function": {
                        "name": "get_my_gpa",
                        "arguments": "{}"
                    }
                }]
            }
        }]
    })
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        
        response = await provider.chat([Message(role="user", content="What is my GPA?")], stream=False)
        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_my_gpa"
        assert response.tool_calls[0].id == "call_gpa_123"


async def test_chat_streaming_text_success():
    """Verify that streaming text chunks yields content chunk-by-chunk."""
    provider = OpenRouterProvider()
    
    # Mocking standard HTTPX stream responses
    mock_client = MagicMock(spec=httpx.AsyncClient)
    
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    
    # Setup mock iterator yielding text lines in SSE format
    async def mock_aiter_lines():
        chunks = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: {"choices": [{"delta": {"content": "!"}}]}',
            "data: [DONE]"
        ]
        for c in chunks:
            yield c
            
    mock_resp.aiter_lines.side_effect = mock_aiter_lines
    mock_resp.aclose = AsyncMock()
    
    mock_client.build_request.return_value = MagicMock()
    mock_client.send.return_value = mock_resp
    mock_client.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await provider.chat([Message(role="user", content="Hi")], stream=True)
        assert len(response.tool_calls) == 0
        
        # Read from the stream iterator
        streamed_chunks = []
        async for chunk in response.stream():
            streamed_chunks.append(chunk)
            
        assert "".join(streamed_chunks) == "Hello world!"


async def test_chat_streaming_tool_calls():
    """Verify that streaming tool call chunks are accumulated and return ToolCall list."""
    provider = OpenRouterProvider()
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    
    # Setup mock iterator yielding tool call chunks
    async def mock_aiter_lines():
        chunks = [
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_my"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "_gpa"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"us"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "er_id\\\": 1}"}}]}}]}',
            "data: [DONE]"
        ]
        for c in chunks:
            yield c
            
    mock_resp.aiter_lines.side_effect = mock_aiter_lines
    mock_resp.aclose = AsyncMock()
    
    mock_client.build_request.return_value = MagicMock()
    mock_client.send.return_value = mock_resp
    mock_client.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await provider.chat([Message(role="user", content="What is my GPA?")], stream=True)
        
        # Tool calls should be resolved
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_my_gpa"
        assert response.tool_calls[0].id == "call_1"
        assert response.tool_calls[0].arguments == {"user_id": 1}


async def test_openrouter_timeout_exception():
    """Verify that httpx timeout maps to ProviderTimeoutError."""
    provider = OpenRouterProvider()
    
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(ProviderTimeoutError):
            await provider.complete("Prompt")


async def test_openrouter_rate_limit_exception():
    """Verify that HTTP 429 maps to ProviderRateLimitError."""
    provider = OpenRouterProvider()
    mock_resp = mock_response(429)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(ProviderRateLimitError):
            await provider.complete("Prompt")
