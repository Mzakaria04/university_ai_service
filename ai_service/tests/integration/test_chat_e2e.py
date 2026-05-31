import pytest
import uuid
import json
from jose import jwt
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock

from ai_service.main import app
from ai_service.config.settings import settings
from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.providers.base import LLMResponse
from ai_service.models.messages import ToolCall
from ai_service.sessions.manager import SessionManager
from ai_service.models.user_context import UserContext, UserRole
from sqlalchemy import text, select

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
        # Clean up database
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
        
        # Insert Course, Term, Transcript for GPA query
        await session.execute(text("""
            INSERT INTO "Course" (id, code, name, "creditHours", scope, "specialType", "isActive", "createdAt", "updatedAt")
            VALUES ('course-math-101', 'MATH101', 'Calculus I', 3, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "Term" (id, name, "startDate", "endDate", "isActive", "createdAt", "updatedAt")
            VALUES ('term-1', 'Fall 2026', NOW(), NOW(), true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "Transcript" (
                id, "studentId", "courseId", "termId", grade, "isPassed", "letterGrade", "gradePoint", "includeInGpa", "isRetake", "createdAt"
            ) VALUES ('trans-1', 'alice-student-111', 'course-math-101', 'term-1', 95.0, true, 'A', 4.0, true, false, NOW())
            ON CONFLICT DO NOTHING;
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


async def test_create_session_e2e(setup_db):
    """Verify that a user can explicitly create a session via the sessions endpoint."""
    token = create_token("alice-student-111")
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        
        session_id = data["session_id"]
        
        # Verify it exists in database
        async with AsyncSessionLocal() as session:
            q = select(AISession).where(AISession.id == session_id)
            res = await session.execute(q)
            session_rec = res.scalars().first()
            assert session_rec is not None
            assert session_rec.user_id == "alice-student-111"
            assert session_rec.role == "STUDENT"


async def test_chat_streaming_e2e(setup_db):
    """Verify end-to-end chat endpoint that executes a tool call and streams the final answer."""
    token = create_token("alice-student-111")
    session_id = str(uuid.uuid4())
    
    # Pre-create session so it exists
    async with AsyncSessionLocal() as db:
        await SessionManager.load_or_create(
            db, session_id,
            user_context=UserContext(
                user_id="alice-student-111",
                university_id="20261111",
                full_name="Alice Student",
                role="STUDENT"
            )
        )
    
    # Mock OpenRouterProvider.chat behavior
    # Round 1 returns tool call. Round 2 returns text response.
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_1", name="get_my_gpa", arguments={})]
    )
    
    async def mock_stream_chunks():
        for chunk in ["Your ", "GPA ", "is ", "4.0."]:
            yield chunk
    text_response = LLMResponse(content="", tool_calls=[], stream_iterator=mock_stream_chunks())

    with patch("ai_service.providers.openrouter.OpenRouterProvider.chat", AsyncMock(side_effect=[tool_response, text_response])):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/chat/{session_id}",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "What is my GPA?"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            
            # Read SSE chunks line-by-line
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
            
            # Assert SSE formatting and contents
            assert len(chunks) > 0
            assert chunks[-1] == "data: [DONE]"
            
            # Parse text contents
            text_result = ""
            for c in chunks[:-1]:
                data = json.loads(c[6:])
                text_result += data["choices"][0]["delta"]["content"]
                
            assert text_result == "Your GPA is 4.0."
            
            # Assert message persistence in DB
            # We expect 4 messages in Phase 2: 1 user, 1 assistant (tool_call), 1 tool (tool_result), 1 assistant (text)
            async with AsyncSessionLocal() as session:
                q = select(AIMessageEvent).where(AIMessageEvent.session_id == session_id).order_by(AIMessageEvent.sequence_number)
                res = await session.execute(q)
                messages = res.scalars().all()
                
                assert len(messages) == 4
                assert messages[0].role == "user"
                assert messages[0].content == "What is my GPA?"
                assert messages[0].sequence_number == 1
                
                assert messages[1].role == "assistant"
                assert messages[1].message_type == "tool_call"
                assert messages[1].sequence_number == 2
                
                assert messages[2].role == "tool"
                assert messages[2].message_type == "tool_result"
                assert messages[2].sequence_number == 3
                
                assert messages[3].role == "assistant"
                assert messages[3].message_type == "text"
                assert messages[3].content == "Your GPA is 4.0."
                assert messages[3].sequence_number == 4


async def test_chat_ownership_validation_e2e(setup_db):
    """Verify that Bob cannot chat in Alice's session and receives an SSE error."""
    alice_token = create_token("alice-student-111")
    bob_token = create_token("bob-student-222")
    session_id = str(uuid.uuid4())
    
    # 1. Alice creates the session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        
        # Manually create session in DB matching the UUID
        async with AsyncSessionLocal() as db:
            await SessionManager.load_or_create(
                db, session_id,
                user_context=UserContext(
                    user_id="alice-student-111",
                    university_id="20261111",
                    full_name="Alice Student",
                    role="STUDENT"
                )
            )

        # 2. Bob attempts to write in Alice's session
        response = await client.post(
            f"/api/v1/chat/{session_id}",
            headers={"Authorization": f"Bearer {bob_token}"},
            json={"message": "Hack GPA?"}
        )
        assert response.status_code == 200
        
        # Read the error chunk yielded from the SSE stream
        chunks = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line)
                
        assert len(chunks) == 1
        error_data = json.loads(chunks[0][6:])
        assert "error" in error_data
        assert "not authorized to access session" in error_data["error"]["message"]
