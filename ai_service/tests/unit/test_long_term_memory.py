import pytest
import uuid
from unittest.mock import AsyncMock

from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.models.user_context import UserContext, UserRole
from ai_service.models.messages import Message
from ai_service.sessions.manager import SessionManager
from ai_service.persistence.message_writer import MessagePersistence
from ai_service.memory.long_term import LongTermMemory
from ai_service.memory.short_term import ShortTermMemory
from ai_service.providers.base import LLMProvider

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
async def db_session():
    """Yields a database session, clears target tables, and rolls back after test."""
    from sqlalchemy import text
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

        # Insert test user
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

async def test_memory_no_compression_under_threshold(db_session):
    """Verify that memory is not compressed when below threshold."""
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # Insert 5 message events (below 20 triggers)
    for i in range(5):
        await MessagePersistence.save_message(db_session, session_id, "user", f"message {i}", "text")
        await MessagePersistence.save_message(db_session, session_id, "assistant", f"reply {i}", "text")

    mock_provider = AsyncMock(spec=LLMProvider)
    ltm = LongTermMemory(mock_provider)

    await ltm.maybe_compress(session_id, db_session)

    # Assert complete was not called
    mock_provider.complete.assert_not_called()

    # Load recent history and assert all 10 messages are unsummarized
    short_mem = ShortTermMemory()
    messages = await short_mem.load(db_session, session_id)
    assert len(messages) == 10

async def test_memory_compression_triggers_and_persists(db_session):
    """Verify that memory compression triggers, generates summary, updates DB, and updates short-term visibility."""
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # Save 22 messages to exceed the 20 messages threshold
    for i in range(11):
        await MessagePersistence.save_message(db_session, session_id, "user", f"user message {i}", "text")
        await MessagePersistence.save_message(db_session, session_id, "assistant", f"assistant message {i}", "text")

    # Set up LLM Mock for summarization
    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.complete.return_value = "This is a summary of the dialogue."
    
    ltm = LongTermMemory(mock_provider)
    await ltm.maybe_compress(session_id, db_session)

    # Verify complete was called
    mock_provider.complete.assert_called_once()

    # Assert session is updated
    from sqlalchemy import select
    res = await db_session.execute(select(AISession).where(AISession.id == session_id))
    session_rec = res.scalar_one()
    assert session_rec.summary_text == "This is a summary of the dialogue."

    # Load summary via service
    loaded_summary = await ltm.load_summary(session_id, db_session)
    assert loaded_summary == "This is a summary of the dialogue."

    # Verify original messages are marked is_summarized=True
    msg_res = await db_session.execute(
        select(AIMessageEvent)
        .where(AIMessageEvent.session_id == session_id)
        .where(AIMessageEvent.message_type != "summary")
    )
    all_msgs = msg_res.scalars().all()
    assert len(all_msgs) == 22
    for m in all_msgs:
        assert m.is_summarized is True

    # Assert that short term memory load now returns 0 messages since all are summarized
    short_mem = ShortTermMemory()
    recent = await short_mem.load(db_session, session_id)
    assert len(recent) == 0
