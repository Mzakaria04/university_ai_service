import pytest
import uuid
import json
from jose import jwt
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text, select

from ai_service.main import app
from ai_service.config.settings import settings
from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.sessions.manager import SessionManager
from ai_service.models.user_context import UserContext, UserRole
from ai_service.persistence.message_writer import MessagePersistence
from ai_service.errors import SessionNotFoundError

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

def create_token(user_id: str, role: str = "STUDENT") -> str:
    payload = {
        "id": user_id,
        "universityId": "20261111",
        "fullName": "Alice Student",
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

@pytest.fixture
async def setup_db():
    """Cleans and populates database with test data, committing transactions to make them visible globally."""
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

        # Insert Alice (student)
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'alice-student-111', '20261111', 'Alice Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        
        # Insert Bob (student) for cross-ownership testing
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'bob-student-222', '20262222', 'Bob Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '87654321', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.commit()

    yield

    async with AsyncSessionLocal() as session:
        # Clean up database again
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "Course"'))
        await session.execute(text('DELETE FROM "Term"'))
        await session.execute(text('DELETE FROM ai_message_events'))
        await session.execute(text('DELETE FROM ai_sessions'))
        await session.execute(text('DELETE FROM "User"'))
        await session.commit()


async def test_get_session_history_success(setup_db):
    token = create_token("alice-student-111")
    session_id = str(uuid.uuid4())
    
    # 1. Pre-create session and insert messages
    async with AsyncSessionLocal() as db:
        await SessionManager.load_or_create(
            db, session_id,
            user_context=UserContext(
                user_id="alice-student-111",
                university_id="20261111",
                full_name="Alice Student",
                role=UserRole.STUDENT
            )
        )
        await MessagePersistence.save_message(db, session_id, "user", "Message 1", "text")
        await MessagePersistence.save_message(db, session_id, "assistant", "Response 1", "text")
        await MessagePersistence.save_message(db, session_id, "user", "Message 2", "text")
        await db.commit()

    # 2. Get history via GET endpoint
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["message_count"] == 3
        messages = data["messages"]
        assert len(messages) == 3
        
        # Check chronological order
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Message 1"
        assert messages[0]["sequence_number"] == 1
        
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Response 1"
        assert messages[1]["sequence_number"] == 2
        
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Message 2"
        assert messages[2]["sequence_number"] == 3


async def test_get_session_history_ownership_violation(setup_db):
    alice_token = create_token("alice-student-111")
    bob_token = create_token("bob-student-222")
    session_id = str(uuid.uuid4())
    
    # Create session owned by Alice
    async with AsyncSessionLocal() as db:
        await SessionManager.load_or_create(
            db, session_id,
            user_context=UserContext(
                user_id="alice-student-111",
                university_id="20261111",
                full_name="Alice Student",
                role=UserRole.STUDENT
            )
        )
        await db.commit()

    # Bob attempts to GET the session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        assert response.status_code == 403
        assert "not authorized to access session" in response.json()["detail"]


async def test_get_session_history_not_found(setup_db):
    token = create_token("alice-student-111")
    non_existent_id = str(uuid.uuid4())
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/sessions/{non_existent_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


async def test_delete_session_success(setup_db):
    token = create_token("alice-student-111")
    session_id = str(uuid.uuid4())
    
    # Create session
    async with AsyncSessionLocal() as db:
        await SessionManager.load_or_create(
            db, session_id,
            user_context=UserContext(
                user_id="alice-student-111",
                university_id="20261111",
                full_name="Alice Student",
                role=UserRole.STUDENT
            )
        )
        await db.commit()

    # Delete session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        delete_response = await client.delete(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert delete_response.status_code == 200
        assert delete_response.json()["status"] == "deleted"

        # Verify in DB that is_deleted is true in metadata_json
        async with AsyncSessionLocal() as db:
            q = select(AISession).where(AISession.id == session_id)
            res = await db.execute(q)
            session_rec = res.scalars().first()
            assert session_rec.metadata_json is not None
            assert session_rec.metadata_json.get("is_deleted") is True

        # Calling GET should now return 404
        get_response = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert get_response.status_code == 404


async def test_delete_session_ownership_violation(setup_db):
    alice_token = create_token("alice-student-111")
    bob_token = create_token("bob-student-222")
    session_id = str(uuid.uuid4())
    
    # Create session owned by Alice
    async with AsyncSessionLocal() as db:
        await SessionManager.load_or_create(
            db, session_id,
            user_context=UserContext(
                user_id="alice-student-111",
                university_id="20261111",
                full_name="Alice Student",
                role=UserRole.STUDENT
            )
        )
        await db.commit()

    # Bob attempts to DELETE the session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        assert response.status_code == 403
        assert "not authorized to access session" in response.json()["detail"]


async def test_delete_session_not_found(setup_db):
    token = create_token("alice-student-111")
    non_existent_id = str(uuid.uuid4())
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(
            f"/api/v1/sessions/{non_existent_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


async def test_load_soft_deleted_session_fails(setup_db):
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    
    # Create session and manually mark soft deleted
    async with AsyncSessionLocal() as db:
        await SessionManager.load_or_create(db, session_id, user_context)
        
        q = select(AISession).where(AISession.id == session_id)
        res = await db.execute(q)
        session_rec = res.scalars().first()
        session_rec.metadata_json = {"is_deleted": True}
        await db.commit()

    # Attempt to load_or_create again, should raise SessionNotFoundError
    async with AsyncSessionLocal() as db:
        with pytest.raises(SessionNotFoundError):
            await SessionManager.load_or_create(db, session_id, user_context)
