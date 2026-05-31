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
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.providers.base import LLMResponse
from ai_service.models.messages import ToolCall
from ai_service.sessions.manager import SessionManager
from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.registry import ToolRegistry

# Explicitly register the student tools
from ai_service.tools.student.gpa import gpa_tool_definition
from ai_service.tools.student.schedule import schedule_tool_definition
from ai_service.tools.student.transcript import transcript_tool_definition
from ai_service.tools.student.attendance import attendance_tool_definition

ToolRegistry.register(gpa_tool_definition)
ToolRegistry.register(schedule_tool_definition)
ToolRegistry.register(transcript_tool_definition)
ToolRegistry.register(attendance_tool_definition)

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
        # Clean up database in reverse-dependency order
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

        # 1. Insert Alice
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES 
                ('alice-student-111', '20261111', 'Alice Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()),
                ('bob-instructor-222', '20262222', 'Bob Instructor', 'INSTRUCTOR'::"Role", 'NATIONAL_ID'::"IdentityType", '87654321', 'hash', false, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 1.5. Insert Building and Rooms
        await session.execute(text("""
            INSERT INTO "Building" (id, name, code, "campusName", "isActive", "createdAt")
            VALUES ('build-1', 'Main Building', 'MAIN', 'Main Campus', true, NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "Room" (id, "buildingId", name, code, capacity, "roomType", "isActive", "createdAt")
            VALUES 
                ('Room 101', 'build-1', 'Room 101', 'R101', 50, 'LECTURE_HALL'::"RoomType", true, NOW()),
                ('Room 202', 'build-1', 'Room 202', 'R202', 40, 'LAB'::"RoomType", true, NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 2. Insert Active Term
        await session.execute(text("""
            INSERT INTO "Term" (id, name, "startDate", "endDate", "isActive", "createdAt", "updatedAt")
            VALUES ('term-1', 'Fall 2026', NOW() - INTERVAL '1 month', NOW() + INTERVAL '3 months', true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 3. Insert Courses
        await session.execute(text("""
            INSERT INTO "Course" (id, code, name, "creditHours", scope, "specialType", "isActive", "createdAt", "updatedAt")
            VALUES 
                ('course-math-101', 'MATH101', 'Calculus I', 3, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW()),
                ('course-cs-101', 'CS101', 'Intro to CS', 4, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 4. Insert CourseOfferings
        await session.execute(text("""
            INSERT INTO "CourseOffering" (id, "courseId", "termId", "passingScore", "finalExamMinPercent", "isActive", "createdAt", "updatedAt")
            VALUES 
                ('co-math-1', 'course-math-101', 'term-1', 50, 30, true, NOW(), NOW()),
                ('co-cs-1', 'course-cs-101', 'term-1', 50, 30, true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 5. Insert Enrollments (Active status)
        await session.execute(text("""
            INSERT INTO "Enrollment" (id, "studentId", "courseOfferingId", status, "createdAt", "updatedAt")
            VALUES 
                ('enr-math-1', 'alice-student-111', 'co-math-1', 'ACTIVE', NOW(), NOW()),
                ('enr-cs-1', 'alice-student-111', 'co-cs-1', 'ACTIVE', NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 6. Insert Sessions
        await session.execute(text("""
            INSERT INTO "Session" (id, "courseOfferingId", name, type, "primaryTeacherId", "maxCapacity", "createdAt", "updatedAt")
            VALUES 
                ('sess-math-1', 'co-math-1', 'Math Lecture 1', 'LECTURE', 'bob-instructor-222', 30, NOW(), NOW()),
                ('sess-cs-1', 'co-cs-1', 'CS Lab 1', 'SECTION', 'bob-instructor-222', 30, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 7. Insert SessionSchedules
        await session.execute(text("""
            INSERT INTO "SessionSchedule" (id, "sessionId", "roomId", "dayOfWeek", "startTime", "endTime", location, "createdAt", "updatedAt")
            VALUES 
                ('ss-math-1', 'sess-math-1', 'Room 101', 1, '2026-05-30 09:00:00'::timestamp, '2026-05-30 10:30:00'::timestamp, 'Building A', NOW(), NOW()),
                ('ss-cs-1', 'sess-cs-1', 'Room 202', 3, '2026-05-30 14:00:00'::timestamp, '2026-05-30 16:00:00'::timestamp, 'Building B', NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 8. Insert AttendanceSessions & Attendances
        await session.execute(text("""
            INSERT INTO "AttendanceSession" (id, "sessionId", "openedById", "weekNumber", type, radius, token, "expiresAt", "isActive", "createdAt")
            VALUES 
                ('atts-math-1', 'sess-math-1', 'bob-instructor-222', 1, 'LECTURE', 10.0, 'token-math', NOW() + INTERVAL '1 hour', true, NOW()),
                ('atts-cs-1', 'sess-cs-1', 'bob-instructor-222', 1, 'SECTION', 10.0, 'token-cs', NOW() + INTERVAL '1 hour', true, NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "Attendance" (id, "studentId", "sessionId", "attendanceSessionId", status, "deviceId", "createdAt")
            VALUES 
                ('att-1', 'alice-student-111', 'sess-math-1', 'atts-math-1', 'PRESENT', 'device-1', NOW()),
                ('att-2', 'alice-student-111', 'sess-cs-1', 'atts-cs-1', 'ABSENT', 'device-1', NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.commit()

    yield

    async with AsyncSessionLocal() as session:
        # Clean up database in reverse-dependency order
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


async def test_multi_tool_execution_e2e(setup_db):
    """Verify that a single turn can execute multiple tools sequentially and persist them."""
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
    # Round 1 returns TWO tool calls (schedule AND attendance).
    # Round 2 returns text response summarizing both.
    multi_tool_response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="call_sched_1", name="get_my_schedule", arguments={}),
            ToolCall(id="call_att_1", name="get_my_attendance", arguments={})
        ]
    )
    
    async def mock_stream_chunks():
        for chunk in ["Here is your schedule and attendance summary."]:
            yield chunk
    text_response = LLMResponse(content="", tool_calls=[], stream_iterator=mock_stream_chunks())

    with patch("ai_service.providers.openrouter.OpenRouterProvider.chat", AsyncMock(side_effect=[multi_tool_response, text_response])):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/chat/{session_id}",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "Show me my schedule and attendance"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            
            # Read SSE chunks line-by-line
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
            
            assert len(chunks) > 0
            assert chunks[-1] == "data: [DONE]"
            
            # Parse text contents
            text_result = ""
            for c in chunks[:-1]:
                data = json.loads(c[6:])
                text_result += data["choices"][0]["delta"]["content"]
                
            assert text_result == "Here is your schedule and attendance summary."
            
            # Verify messages stored in the database.
            # We expect:
            # 1. User message (What is my schedule and attendance)
            # 2. Assistant message representing BOTH tool calls
            # 3. First Tool result (get_my_schedule result)
            # 4. Second Tool result (get_my_attendance result)
            # 5. Assistant text message response
            async with AsyncSessionLocal() as session:
                q = select(AIMessageEvent).where(AIMessageEvent.session_id == session_id).order_by(AIMessageEvent.sequence_number)
                res = await session.execute(q)
                messages = res.scalars().all()
                
                assert len(messages) == 5
                
                assert messages[0].role == "user"
                assert messages[0].content == "Show me my schedule and attendance"
                
                # Check the assistant tool_call message
                assert messages[1].role == "assistant"
                assert messages[1].message_type == "tool_call"
                # Make sure both tool calls are in metadata_json
                tcs = messages[1].metadata_json["tool_calls"]
                assert len(tcs) == 2
                assert tcs[0]["function"]["name"] == "get_my_schedule"
                assert tcs[1]["function"]["name"] == "get_my_attendance"
                
                # Check tool result messages
                assert messages[2].role == "tool"
                assert messages[2].message_type == "tool_result"
                assert messages[2].tool_name == "get_my_schedule"
                assert "schedule" in messages[2].content
                
                assert messages[3].role == "tool"
                assert messages[3].message_type == "tool_result"
                assert messages[3].tool_name == "get_my_attendance"
                assert "attendance" in messages[3].content
                
                # Final response
                assert messages[4].role == "assistant"
                assert messages[4].message_type == "text"
                assert messages[4].content == "Here is your schedule and attendance summary."
