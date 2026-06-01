import pytest
import uuid
import json
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIExecutionTrace
from ai_service.models.user_context import UserContext, UserRole
from ai_service.models.messages import Message, ToolCall
from ai_service.providers.base import LLMProvider, LLMResponse
from ai_service.orchestration.conversation_orchestrator import ConversationOrchestrator
from ai_service.persistence.message_writer import MessagePersistence

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
async def cleanup_engine():
    yield
    from ai_service.db.session import engine
    await engine.dispose()

@pytest.fixture
async def db_session() -> AsyncSession:
    """Fixture that yields an async database session, cleans DB tables, and rolls back changes."""
    async with AsyncSessionLocal() as session:
        # Clean tables in correct order to avoid FK violations
        await session.execute(text('DELETE FROM ai_execution_traces'))
        await session.execute(text('DELETE FROM ai_tool_execution_logs'))
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "Course"'))
        await session.execute(text('DELETE FROM "Term"'))
        await session.execute(text('DELETE FROM ai_message_events'))
        await session.execute(text('DELETE FROM ai_sessions'))
        await session.execute(text('DELETE FROM "User"'))
        await session.commit()

        # Insert mock student user
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'user-111', '20261111', 'Alice', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.commit()
        
        yield session
        await session.rollback()

async def test_save_execution_trace_persistence(db_session):
    """Verify that save_execution_trace directly writes record to ai_execution_traces."""
    session_id = str(uuid.uuid4())
    
    # Pre-create session to satisfy FK constraints
    session_rec = AISession(
        id=session_id,
        user_id="user-111",
        role="STUDENT",
        message_count=1
    )
    db_session.add(session_rec)
    await db_session.flush()
    await db_session.commit()

    # Call save_execution_trace
    trace_rec = await MessagePersistence.save_execution_trace(
        db=db_session,
        session_id=session_id,
        request_id="test-req-id",
        user_id="user-111",
        user_role="STUDENT",
        provider_used="openrouter",
        model_used="z-ai/glm-4.5-air:free",
        provider_fallback=False,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        tool_calls_count=0,
        tools_used=None,
        latency_ms=150,
        rag_chunks_retrieved=0,
        success=True,
        error_type=None
    )

    assert trace_rec is not None
    assert trace_rec.id is not None
    
    # Query from DB
    q = select(AIExecutionTrace).where(AIExecutionTrace.request_id == "test-req-id")
    res = await db_session.execute(q)
    db_trace = res.scalars().first()

    assert db_trace is not None
    assert db_trace.session_id == session_id
    assert db_trace.user_id == "user-111"
    assert db_trace.user_role == "STUDENT"
    assert db_trace.provider_used == "openrouter"
    assert db_trace.model_used == "z-ai/glm-4.5-air:free"
    assert db_trace.prompt_tokens == 10
    assert db_trace.completion_tokens == 20
    assert db_trace.total_tokens == 30
    assert db_trace.success is True
    assert db_trace.error_type is None
    assert db_trace.latency_ms == 150

async def test_orchestrator_execution_trace_on_success(db_session):
    """Verify that running orchestrate successfully writes a trace record with correct details."""
    provider = MagicMock(spec=LLMProvider)
    
    async def mock_stream_chunks():
        for chunk in ["Hello ", "Alice!"]:
            yield chunk

    # Return LLMResponse with provider details
    mock_response = LLMResponse(
        content="", 
        tool_calls=[], 
        stream_iterator=mock_stream_chunks(),
        provider_name="openrouter",
        model_name="z-ai/glm-4.5-air:free",
        provider_fallback=False,
        prompt_tokens=40,
        completion_tokens=80,
        total_tokens=120
    )
    provider.chat = AsyncMock(return_value=mock_response)

    orchestrator = ConversationOrchestrator(provider)

    user_context = UserContext(
        user_id="user-111",
        university_id="20261111",
        full_name="Alice",
        role=UserRole.STUDENT
    )
    session_id = str(uuid.uuid4())
    await db_session.execute(text(f"""
        INSERT INTO ai_sessions (id, user_id, role, message_count, created_at, updated_at)
        VALUES ('{session_id}', 'user-111', 'STUDENT', 0, NOW(), NOW())
    """))
    await db_session.commit()

    # Bind request_id to structlog contextvars
    import structlog
    structlog.contextvars.bind_contextvars(request_id="orchestrator-success-req-id")

    # Run orchestration
    chunks = []
    async for chunk in orchestrator.orchestrate(db_session, session_id, user_context, "Hello assistant"):
        chunks.append(chunk)

    # Check trace was persisted
    q = select(AIExecutionTrace).where(AIExecutionTrace.request_id == "orchestrator-success-req-id")
    res = await db_session.execute(q)
    db_trace = res.scalars().first()

    assert db_trace is not None
    assert db_trace.session_id == session_id
    assert db_trace.success is True
    assert db_trace.provider_used == "openrouter"
    assert db_trace.model_used == "z-ai/glm-4.5-air:free"
    assert db_trace.provider_fallback is False
    assert db_trace.prompt_tokens is not None
    assert db_trace.completion_tokens is not None
    assert db_trace.latency_ms >= 0
    assert db_trace.tool_calls_count == 0
    assert db_trace.error_type is None

async def test_orchestrator_execution_trace_on_failure(db_session):
    """Verify that running orchestrate with an exception writes a failed trace record."""
    provider = MagicMock(spec=LLMProvider)
    provider.chat = AsyncMock(side_effect=RuntimeError("Provider failure simulation"))

    orchestrator = ConversationOrchestrator(provider)

    user_context = UserContext(
        user_id="user-111",
        university_id="20261111",
        full_name="Alice",
        role=UserRole.STUDENT
    )
    session_id = str(uuid.uuid4())
    await db_session.execute(text(f"""
        INSERT INTO ai_sessions (id, user_id, role, message_count, created_at, updated_at)
        VALUES ('{session_id}', 'user-111', 'STUDENT', 0, NOW(), NOW())
    """))
    await db_session.commit()

    import structlog
    structlog.contextvars.bind_contextvars(request_id="orchestrator-fail-req-id")

    # Run orchestration and check that it raises
    with pytest.raises(RuntimeError, match="Provider failure simulation"):
        async for _ in orchestrator.orchestrate(db_session, session_id, user_context, "Hello assistant"):
            pass

    # Check trace was persisted with success=False
    q = select(AIExecutionTrace).where(AIExecutionTrace.request_id == "orchestrator-fail-req-id")
    res = await db_session.execute(q)
    db_trace = res.scalars().first()

    assert db_trace is not None
    assert db_trace.session_id == session_id
    assert db_trace.success is False
    assert db_trace.error_type == "RuntimeError"
