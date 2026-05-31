import pytest
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select

from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.models.user_context import UserContext, UserRole
from ai_service.models.messages import Message
from ai_service.sessions.manager import SessionManager
from ai_service.persistence.message_writer import MessagePersistence
from ai_service.memory.short_term import ShortTermMemory
from ai_service.memory.composer import MemoryComposer

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
        # Clean up database in correct reverse-dependency order
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

async def test_empty_memory_window(db_session):
    """Verify that empty session message logs return empty memory states."""
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    await SessionManager.load_or_create(db_session, session_id, user_context)

    memory_service = ShortTermMemory()
    messages = await memory_service.load(db_session, session_id)
    assert len(messages) == 0

    context_block = MemoryComposer.compose_context_block(messages)
    assert context_block == ""


async def test_normal_memory_window_and_composer(db_session):
    """Verify that recent message events are retrieved chronologically and formatted correctly."""
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # Save multiple turns
    await MessagePersistence.save_message(db_session, session_id, "user", "Hello", "text")
    await MessagePersistence.save_message(db_session, session_id, "assistant", "Hi, how can I help?", "text")
    await MessagePersistence.save_message(db_session, session_id, "user", "What is my GPA?", "text")
    # Save a tool result as well
    await MessagePersistence.save_message(
        db_session, session_id, "tool", "{\"cumulative_gpa\": 3.5}", "tool_result",
        tool_call_id="call_gpa", tool_name="get_my_gpa"
    )

    memory_service = ShortTermMemory()
    messages = await memory_service.load(db_session, session_id)
    
    assert len(messages) == 4
    # Chronological ordering assertion
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi, how can I help?"
    assert messages[2].role == "user"
    assert messages[2].content == "What is my GPA?"
    assert messages[3].role == "tool"
    assert messages[3].content == "{\"cumulative_gpa\": 3.5}"
    assert messages[3].tool_name == "get_my_gpa"

    # Compose block and assert format
    context_block = MemoryComposer.compose_context_block(messages)
    assert "[Recent Conversation History]" in context_block
    assert "User: Hello" in context_block
    assert "Assistant: Hi, how can I help?" in context_block
    assert "User: What is my GPA?" in context_block
    assert "Tool (get_my_gpa): [Result: {\"cumulative_gpa\": 3.5}]" in context_block


async def test_memory_token_budget_trimming(db_session):
    """Verify that memory loads trim older messages if token count exceeds the budget."""
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    await SessionManager.load_or_create(db_session, session_id, user_context)

    # Save a sequence of large messages
    # Each of these represents approximately 1000 tokens of text
    large_text = "lorem ipsum " * 300 # ~300 words
    
    # Save 4 large turns (User -> Assistant -> User -> Assistant)
    await MessagePersistence.save_message(db_session, session_id, "user", f"Large User Message 1: {large_text}", "text")
    await MessagePersistence.save_message(db_session, session_id, "assistant", f"Large Assistant Message 1: {large_text}", "text")
    await MessagePersistence.save_message(db_session, session_id, "user", f"Large User Message 2: {large_text}", "text")
    await MessagePersistence.save_message(db_session, session_id, "assistant", f"Large Assistant Message 2: {large_text}", "text")

    # Set token budget artificially small to enforce trimming (e.g. 1500 tokens)
    # The 2 newest messages (Message 2 turns) should total ~1220 tokens and fit.
    # Adding Assistant Message 1 would exceed 1500, so it and older messages should be dropped.
    memory_service = ShortTermMemory(token_budget=1500)
    messages = await memory_service.load(db_session, session_id)

    # Assert that only the 2 newest messages are returned, in chronological order
    assert len(messages) == 2
    assert "Large User Message 2" in messages[0].content
    assert "Large Assistant Message 2" in messages[1].content
