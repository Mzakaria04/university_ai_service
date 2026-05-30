import pytest
import uuid
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.models.user_context import UserContext, UserRole
from ai_service.models.messages import Message, ToolCall
from ai_service.providers.base import LLMProvider, LLMResponse
from ai_service.orchestration.conversation_orchestrator import ConversationOrchestrator
from ai_service.sessions.manager import SessionManager
from ai_service.tools.registry import ToolRegistry
from ai_service.tools.student.gpa import gpa_tool_definition
ToolRegistry.register(gpa_tool_definition)

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

@pytest.fixture
async def db_session() -> AsyncSession:
    """Fixture that yields an async database session and always rolls back changes."""
    async with AsyncSessionLocal() as session:
        # Clean up database
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "Course"'))
        await session.execute(text('DELETE FROM "Term"'))
        await session.execute(text('DELETE FROM ai_message_events'))
        await session.execute(text('DELETE FROM ai_sessions'))
        await session.execute(text('DELETE FROM "User"'))
        await session.commit()

        # Populate dependencies for foreign key constraints
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'alice-student-111', '20261111', 'Alice Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.commit()
        
        yield session
        await session.rollback()

async def mock_text_stream(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk

async def test_orchestrator_direct_completion(db_session):
    """Verify that the orchestrator streams text directly if no tool calls are requested."""
    # 1. Setup mock provider
    provider = MagicMock(spec=LLMProvider)
    
    async def mock_stream_chunks():
        for chunk in ["Hello ", "Alice!"]:
            yield chunk

    # Return LLMResponse with a text stream
    mock_response = LLMResponse(content="", tool_calls=[], stream_iterator=mock_stream_chunks())
    provider.chat = AsyncMock(return_value=mock_response)

    orchestrator = ConversationOrchestrator(provider)

    # 2. Setup database session context
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    session_id = str(uuid.uuid4())
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # 3. Execute
    yielded_chunks = []
    async for chunk in orchestrator.orchestrate(db_session, session_id, user_context, "Hello assistant"):
        yielded_chunks.append(chunk)

    # 4. Assertions
    assert yielded_chunks == ["Hello ", "Alice!"]
    
    # Provider.chat should be called once with correct parameters
    provider.chat.assert_called_once()
    call_args = provider.chat.call_args[1]
    messages_sent = call_args["messages"]
    assert len(messages_sent) == 2
    assert messages_sent[0].role == "system"
    assert messages_sent[1].role == "user"
    assert messages_sent[1].content == "Hello assistant"

    # Confirm assistant response was written to database
    query = select(AIMessageEvent).where(
        AIMessageEvent.session_id == session_id,
        AIMessageEvent.role == "assistant"
    )
    res = await db_session.execute(query)
    messages_db = res.scalars().all()
    assert len(messages_db) == 1
    assert messages_db[0].content == "Hello Alice!"
    assert messages_db[0].message_type == "text"


async def test_orchestrator_with_tool_execution(db_session):
    """Verify that the orchestrator executes a tool, appends the result, and queries LLM again."""
    # 1. Setup DB dependencies for the GPA tool
    # Insert course and transcript
    await db_session.execute(text("""
        INSERT INTO "Course" (id, code, name, "creditHours", scope, "specialType", "isActive", "createdAt", "updatedAt")
        VALUES ('course-math-101', 'MATH101', 'Calculus I', 3, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW())
        ON CONFLICT DO NOTHING;
    """))
    await db_session.execute(text("""
        INSERT INTO "Term" (id, name, "startDate", "endDate", "isActive", "createdAt", "updatedAt")
        VALUES ('term-1', 'Fall 2026', NOW(), NOW(), true, NOW(), NOW())
        ON CONFLICT DO NOTHING;
    """))
    await db_session.execute(text("""
        INSERT INTO "Transcript" (
            id, "studentId", "courseId", "termId", grade, "isPassed", "letterGrade", "gradePoint", "includeInGpa", "isRetake", "createdAt"
        ) VALUES ('trans-1', 'alice-student-111', 'course-math-101', 'term-1', 95.0, true, 'A', 4.0, true, false, NOW())
        ON CONFLICT DO NOTHING;
    """))
    await db_session.commit()

    # 2. Setup mock provider to return tool call first, then text completion
    provider = MagicMock(spec=LLMProvider)
    
    # 1st response: Tool call
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_gpa_123", name="get_my_gpa", arguments={})]
    )
    
    # 2nd response: Text stream
    async def mock_stream_chunks():
        for chunk in ["Your ", "GPA ", "is ", "4.0."]:
            yield chunk
    text_response = LLMResponse(content="", tool_calls=[], stream_iterator=mock_stream_chunks())

    # We mock the chat calls sequentially
    provider.chat = AsyncMock(side_effect=[tool_response, text_response])

    orchestrator = ConversationOrchestrator(provider)

    # 3. Setup database session context
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    session_id = str(uuid.uuid4())
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # 4. Execute
    yielded_chunks = []
    async for chunk in orchestrator.orchestrate(db_session, session_id, user_context, "What is my GPA?"):
        yielded_chunks.append(chunk)

    # 5. Assertions
    assert yielded_chunks == ["Your ", "GPA ", "is ", "4.0."]
    
    # Provider.chat should be called exactly twice
    assert provider.chat.call_count == 2
    
    # Second call should include original user query, tool call, and tool execution result
    second_call_args = provider.chat.call_args_list[1][1]
    history = second_call_args["messages"]
    
    assert len(history) == 4 # System, User, Assistant (tool call), Tool (result)
    assert history[0].role == "system"
    assert history[1].role == "user"
    
    assert history[2].role == "assistant"
    assert history[2].message_type == "tool_call"
    
    assert history[3].role == "tool"
    assert history[3].message_type == "tool_result"
    assert "cumulative_gpa" in history[3].content
    assert "4.0" in history[3].content

    # Confirm assistant response was written to database
    query = select(AIMessageEvent).where(
        AIMessageEvent.session_id == session_id,
        AIMessageEvent.role == "assistant"
    )
    res = await db_session.execute(query)
    messages_db = res.scalars().all()
    assert len(messages_db) == 1
    assert messages_db[0].content == "Your GPA is 4.0."


async def test_orchestrator_max_iterations_bound(db_session):
    """Verify that the orchestrator aborts with RuntimeError if LLM enters an infinite tool-calling loop."""
    # 1. Setup mock provider to always return tool call
    provider = MagicMock(spec=LLMProvider)
    loop_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_loop", name="get_my_gpa", arguments={})]
    )
    provider.chat = AsyncMock(return_value=loop_response)

    orchestrator = ConversationOrchestrator(provider)

    # 2. Setup database session context
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    session_id = str(uuid.uuid4())
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # 3. Execute and verify it raises RuntimeError
    with pytest.raises(RuntimeError, match="exceeded maximum tool calling rounds"):
        async for _ in orchestrator.orchestrate(db_session, session_id, user_context, "Keep calling GPA"):
            pass
            
    # Confirm it ran exactly 5 iterations before raising
    assert provider.chat.call_count == 5
