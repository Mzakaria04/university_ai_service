import pytest
from unittest.mock import patch, MagicMock

from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.registry import ToolRegistry
from ai_service.tools.rag.faculty_bylaw_search import bylaw_tool_definition

# Register the tool for unit testing context
ToolRegistry.register(bylaw_tool_definition)
from ai_service.tools.executor import ToolExecutor
from ai_service.errors import ToolAuthorizationError

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
async def cleanup_engine():
    """Ensure engine is disposed after each test to prevent event loop mismatch errors on Windows."""
    yield
    from ai_service.db.session import engine
    await engine.dispose()

async def test_rag_tool_registry_authorization():
    """Verify that all roles have authorization to search bylaws."""
    assert ToolRegistry.is_authorized("faculty_bylaw_search", UserRole.STUDENT) is True
    assert ToolRegistry.is_authorized("faculty_bylaw_search", UserRole.INSTRUCTOR) is True
    assert ToolRegistry.is_authorized("faculty_bylaw_search", UserRole.ADMIN) is True


async def test_rag_tool_executor_execution():
    """Verify that ToolExecutor successfully runs the RAG search tool and returns the mocked response."""
    student_ctx = UserContext(
        user_id="student-123",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    
    mock_db = MagicMock()
    executor = ToolExecutor(mock_db)

    # Patch the RAG pipeline ask function to prevent external Groq API calls in unit testing
    with patch("ai_service.tools.rag.faculty_bylaw_search.rag_ask") as mock_ask:
        mock_ask.return_value = "بموجب المادة 5، يتطلب التخرج الحصول على معدل تراكمي 2.0 على الأقل."
        
        result = await executor.execute(
            tool_name="faculty_bylaw_search",
            arguments={"query": "ما هي شروط التخرج؟"},
            user_context=student_ctx
        )
        
        assert result.success is True
        assert result.data == "بموجب المادة 5، يتطلب التخرج الحصول على معدل تراكمي 2.0 على الأقل."
        mock_ask.assert_called_once_with("ما هي شروط التخرج؟")
