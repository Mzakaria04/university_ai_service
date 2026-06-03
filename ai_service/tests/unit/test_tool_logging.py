import pytest
import uuid
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent, AIToolExecutionLog
from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.base import ToolDefinition, ToolResult, ToolDomain
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS

# Enable asyncio testing
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

async def test_tool_execution_persists_log_to_db(db_session):
    """Verify that a successful tool call writes correct records to ai_tool_execution_logs."""
    session_id = str(uuid.uuid4())
    message_event_id = str(uuid.uuid4())

    # Pre-create session & message event to satisfy FK constraints
    session_rec = AISession(
        id=session_id,
        user_id="user-111",
        role="STUDENT",
        message_count=1
    )
    db_session.add(session_rec)
    await db_session.flush()

    msg_rec = AIMessageEvent(
        id=message_event_id,
        session_id=session_id,
        role="assistant",
        message_type="tool_call",
        content="",
        sequence_number=1
    )
    db_session.add(msg_rec)
    await db_session.flush()
    await db_session.commit()

    # Define tool
    async def mock_tool_handler(db, user_id, test_arg):
        import asyncio
        await asyncio.sleep(0.02)
        return ToolResult(success=True, data={"output_key": "output_val"})

    tool_def = ToolDefinition(
        name="test_logging_tool",
        description="A tool for testing persistence",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_tool_handler
    )

    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("test_logging_tool")
    ToolRegistry.register(tool_def)

    try:
        user_ctx = UserContext(
            user_id="user-111",
            university_id="20261111",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        executor = ToolExecutor(db_session)
        
        result = await executor.execute(
            tool_name="test_logging_tool",
            arguments={"test_arg": "test_val"},
            user_context=user_ctx,
            session_id=session_id,
            message_event_id=message_event_id
        )
        assert result.success is True

        # Query the log from DB
        q = select(AIToolExecutionLog).where(AIToolExecutionLog.tool_name == "test_logging_tool")
        res = await db_session.execute(q)
        log = res.scalars().first()

        assert log is not None
        assert log.session_id == session_id
        assert log.message_event_id == message_event_id
        assert log.tool_name == "test_logging_tool"
        assert log.user_id == "user-111"
        assert log.user_role == "STUDENT"
        assert log.arguments_json == {"test_arg": "test_val"}
        assert log.result_json == {"output_key": "output_val"}
        assert log.success is True
        assert log.error_message is None
        assert log.attempt_number == 1
        assert log.latency_ms > 0
    finally:
        if "test_logging_tool" in ToolRegistry._tools:
            del ToolRegistry._tools["test_logging_tool"]
        ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("test_logging_tool")
