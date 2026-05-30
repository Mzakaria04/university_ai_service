import pytest
import asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.models.user_context import UserContext, UserRole
from ai_service.sessions.manager import SessionManager
from ai_service.persistence.message_writer import MessagePersistence
from ai_service.errors import SessionOwnershipError

# Enable asyncio testing for pytest
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

from sqlalchemy import select, text

@pytest.fixture
async def db_session() -> AsyncSession:
    """Fixture that yields an async database session, cleans DB, populates dummy users, and always rolls back changes."""
    async with AsyncSessionLocal() as session:
        # Clean up database to avoid unique constraints / FK violations from other tests
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "Course"'))
        await session.execute(text('DELETE FROM "Term"'))
        await session.execute(text('DELETE FROM ai_message_events'))
        await session.execute(text('DELETE FROM ai_sessions'))
        await session.execute(text('DELETE FROM "User"'))
        await session.commit()

        # Insert mock student users to satisfy foreign key constraints
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'alice-uuid-111', '20261111', 'Alice', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'bob-uuid-222', '20262222', 'Bob', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '87654321', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.commit() # Commit users so they are visible, transaction rollback at end will clean them up!
        
        yield session
        await session.rollback()


async def test_session_lazy_creation_and_ownership(db_session):
    """Verify that SessionManager creates sessions lazily and enforces user ownership."""
    user_alice = UserContext(
        user_id="alice-uuid-111",
        university_id="20261111",
        full_name="Alice",
        role=UserRole.STUDENT
    )
    user_bob = UserContext(
        user_id="bob-uuid-222",
        university_id="20262222",
        full_name="Bob",
        role=UserRole.STUDENT
    )
    
    session_id = str(uuid.uuid4())

    # Step 1: Session does not exist yet. Lazy load should create it.
    session = await SessionManager.load_or_create(db_session, session_id, user_alice)
    assert session is not None
    assert session.id == session_id
    assert session.user_id == "alice-uuid-111"
    assert session.role == "STUDENT"
    assert session.message_count == 0

    # Step 2: Request the same session with Alice again. Should load it without issues.
    session_loaded = await SessionManager.load_or_create(db_session, session_id, user_alice)
    assert session_loaded.id == session_id

    # Step 3: Request the same session with Bob. Should raise SessionOwnershipError (403).
    with pytest.raises(SessionOwnershipError):
        await SessionManager.load_or_create(db_session, session_id, user_bob)


async def test_message_persistence_and_sequence_ordering(db_session):
    """Verify that MessagePersistence stores messages and auto-increments sequence numbers."""
    user_alice = UserContext(
        user_id="alice-uuid-111",
        university_id="20261111",
        full_name="Alice",
        role=UserRole.STUDENT
    )
    session_id = str(uuid.uuid4())
    
    # Pre-create session
    await SessionManager.load_or_create(db_session, session_id, user_alice)

    # Save User message
    msg1 = await MessagePersistence.save_message(
        db=db_session,
        session_id=session_id,
        role="user",
        content="Hello, what is my GPA?",
        message_type="text"
    )
    assert msg1.sequence_number == 1
    assert msg1.role == "user" # Transparently converted to lowercase python string
    assert msg1.message_type == "text"

    # Save Assistant message
    msg2 = await MessagePersistence.save_message(
        db=db_session,
        session_id=session_id,
        role="assistant",
        content="Let me lookup your GPA.",
        message_type="text"
    )
    assert msg2.sequence_number == 2
    assert msg2.role == "assistant"

    # Save Tool message (checking uppercase enum conversion and binding)
    msg3 = await MessagePersistence.save_message(
        db=db_session,
        session_id=session_id,
        role="tool",
        content='{"gpa": 3.75}',
        message_type="tool_result",
        tool_call_id="call_123",
        tool_name="get_my_gpa"
    )
    assert msg3.sequence_number == 3
    assert msg3.role == "tool"
    assert msg3.message_type == "tool_result"

    # Verify session message count was updated
    q = select(AISession).where(AISession.id == session_id)
    res = await db_session.execute(q)
    session_rec = res.scalars().first()
    assert session_rec.message_count == 3
