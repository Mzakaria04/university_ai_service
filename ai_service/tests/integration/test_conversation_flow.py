import pytest
import uuid
import json
from jose import jwt
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from sqlalchemy import text, select

from ai_service.main import app
from ai_service.config.settings import settings
from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent, AIFeedback
from ai_service.providers.base import LLMResponse
from ai_service.models.messages import ToolCall
from ai_service.sessions.manager import SessionManager
from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.registry import ToolRegistry

# Register tools
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

def create_token(user_id: str, full_name: str = "Alice Student", role: str = "STUDENT") -> str:
    payload = {
        "id": user_id,
        "universityId": "20261111",
        "fullName": full_name,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

@pytest.fixture
async def setup_db():
    """Cleans and populates database with test data, committing transactions to make them visible globally."""
    async with AsyncSessionLocal() as session:
        # Clean up database in reverse-dependency order
        await session.execute(text('DELETE FROM ai_feedback'))
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "CourseInstructor"'))
        await session.execute(text('DELETE FROM "Attendance"'))
        await session.execute(text('DELETE FROM "AttendanceSession"'))
        await session.execute(text('DELETE FROM "SessionSchedule"'))
        await session.execute(text('DELETE FROM "Session"'))
        await session.execute(text('DELETE FROM "Room"'))
        await session.execute(text('DELETE FROM "Building"'))
        await session.execute(text('DELETE FROM "Enrollment"'))
        await session.execute(text('DELETE FROM "CourseOffering"'))
        await session.execute(text('DELETE FROM "Course"'))
        await session.execute(text('DELETE FROM "Term"'))
        await session.execute(text('DELETE FROM ai_message_events'))
        await session.execute(text('DELETE FROM ai_sessions'))
        await session.execute(text('DELETE FROM "User"'))
        await session.commit()

        # Insert Alice (student) & Bob (student)
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES 
                ('alice-student-111', '20261111', 'Alice Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()),
                ('bob-student-222', '20262222', 'Bob Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '87654321', 'hash', false, NOW(), NOW())
            ON CONFLICT DO NOTHING;
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
        # Clean up database in reverse-dependency order
        await session.execute(text('DELETE FROM ai_feedback'))
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "CourseInstructor"'))
        await session.execute(text('DELETE FROM "Attendance"'))
        await session.execute(text('DELETE FROM "AttendanceSession"'))
        await session.execute(text('DELETE FROM "SessionSchedule"'))
        await session.execute(text('DELETE FROM "Session"'))
        await session.execute(text('DELETE FROM "Room"'))
        await session.execute(text('DELETE FROM "Building"'))
        await session.execute(text('DELETE FROM "Enrollment"'))
        await session.execute(text('DELETE FROM "CourseOffering"'))
        await session.execute(text('DELETE FROM "Course"'))
        await session.execute(text('DELETE FROM "Term"'))
        await session.execute(text('DELETE FROM ai_message_events'))
        await session.execute(text('DELETE FROM ai_sessions'))
        await session.execute(text('DELETE FROM "User"'))
        await session.commit()


async def test_full_conversational_and_feedback_flow(setup_db):
    """
    Verifies full student session flow:
    1. Explicit session creation.
    2. Multi-turn dialogue (Greeting turn -> GPA query turn).
    3. Session restoration via GET.
    4. Successful feedback submission.
    5. Feedback access control (403 for unauthorized, 404 for missing events).
    """
    alice_token = create_token("alice-student-111", "Alice Student", "STUDENT")
    bob_token = create_token("bob-student-222", "Bob Student", "STUDENT")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Alice creates a session
        session_response = await client.post(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        # 2. Turn 1: Conversational Greeting (No tool calls)
        async def mock_stream_greeting():
            for chunk in ["Hello! ", "How ", "can ", "I ", "help ", "you?"]:
                yield chunk
        greeting_response = LLMResponse(content="", tool_calls=[], stream_iterator=mock_stream_greeting())

        with patch("ai_service.providers.openrouter.OpenRouterProvider.chat", AsyncMock(return_value=greeting_response)) as mock_chat:
            chat1_response = await client.post(
                f"/api/v1/chat/{session_id}",
                headers={"Authorization": f"Bearer {alice_token}"},
                json={"message": "Hi assistant"}
            )
            assert chat1_response.status_code == 200
            
            # Read greeting stream
            greeting_text = ""
            async for line in chat1_response.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    greeting_text += json.loads(line[6:])["choices"][0]["delta"]["content"]
            assert greeting_text == "Hello! How can I help you?"

        # 3. Turn 2: GPA query (with tool calling and short-term memory loaded)
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="call_gpa", name="get_my_gpa", arguments={})]
        )
        async def mock_stream_gpa():
            for chunk in ["Your ", "GPA ", "is ", "4.0."]:
                yield chunk
        gpa_final_response = LLMResponse(content="", tool_calls=[], stream_iterator=mock_stream_gpa())

        with patch("ai_service.providers.openrouter.OpenRouterProvider.chat", AsyncMock(side_effect=[tool_call_response, gpa_final_response])) as mock_chat:
            chat2_response = await client.post(
                f"/api/v1/chat/{session_id}",
                headers={"Authorization": f"Bearer {alice_token}"},
                json={"message": "What is my GPA?"}
            )
            assert chat2_response.status_code == 200
            
            # Read gpa stream
            gpa_text = ""
            async for line in chat2_response.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    gpa_text += json.loads(line[6:])["choices"][0]["delta"]["content"]
            assert gpa_text == "Your GPA is 4.0."

            # Verify that in Turn 2, short-term memory was loaded and composed in the chat completion call!
            # The second call to openrouter (index 1) should have the greeting turn in system prompt or message list
            assert mock_chat.call_count == 2
            first_chat_messages = mock_chat.call_args_list[0][1]["messages"]
            # First message in Turn 2 (which is the tool recall call, index 0 of the call)
            # should contain the system prompt (which has memory block containing greeting text)
            system_msg = first_chat_messages[0]
            assert "Hello! How can I help you?" in system_msg.content

        # 4. Restoration: Retrieve session history
        history_response = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert history_data["session_id"] == session_id
        
        db_messages = history_data["messages"]
        # Expected messages:
        # 1. user: "Hi assistant"
        # 2. assistant: "Hello! How can I help you?"
        # 3. user: "What is my GPA?"
        # 4. assistant: tool_call (intermediary)
        # 5. tool: tool_result (intermediary)
        # 6. assistant: "Your GPA is 4.0."
        assert len(db_messages) == 6
        
        # Save the assistant response ID to submit feedback
        assistant_gpa_message_id = db_messages[5]["id"]
        assert db_messages[5]["role"] == "assistant"
        assert db_messages[5]["content"] == "Your GPA is 4.0."

        # 5. Feedback Submission: Alice rates her GPA response (thumbs up + comment)
        feedback_response = await client.post(
            "/api/v1/feedback",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "message_event_id": assistant_gpa_message_id,
                "rating": 1,
                "comment": "Perfect answer!"
            }
        )
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        assert feedback_data["status"] == "success"
        assert feedback_data["is_positive"] is True
        
        # Confirm feedback was written to the database
        async with AsyncSessionLocal() as session:
            q = select(AIFeedback).where(AIFeedback.id == feedback_data["feedback_id"])
            res = await session.execute(q)
            fb_record = res.scalars().first()
            assert fb_record is not None
            assert fb_record.message_event_id == assistant_gpa_message_id
            assert fb_record.user_id == "alice-student-111"
            assert fb_record.is_positive is True
            assert fb_record.comment == "Perfect answer!"

        # 6. Security check: Bob attempts to submit feedback for Alice's message (expects 403)
        bob_feedback_response = await client.post(
            "/api/v1/feedback",
            headers={"Authorization": f"Bearer {bob_token}"},
            json={
                "message_event_id": assistant_gpa_message_id,
                "rating": 0,
                "comment": "Try to hack rating"
            }
        )
        assert bob_feedback_response.status_code == 403
        assert "Cannot submit feedback for message events of another user" in bob_feedback_response.json()["detail"]

        # 7. Non-existent check: Alice submits feedback on a fake message ID (expects 404)
        fake_uuid = str(uuid.uuid4())
        fake_feedback_response = await client.post(
            "/api/v1/feedback",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "message_event_id": fake_uuid,
                "rating": 0,
                "comment": "Fake id"
            }
        )
        assert fake_feedback_response.status_code == 404
        assert "Message event not found" in fake_feedback_response.json()["detail"]
