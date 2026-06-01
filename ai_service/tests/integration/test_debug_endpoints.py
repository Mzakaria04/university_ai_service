import pytest
import uuid
from httpx import AsyncClient, ASGITransport

from ai_service.main import app
from ai_service.config.settings import settings
from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent, AIExecutionTrace
from ai_service.models.user_context import UserContext, UserRole
from ai_service.sessions.manager import SessionManager
from ai_service.persistence.message_writer import MessagePersistence

# Pre-register tools for test execution
from ai_service.tools.registry import ToolRegistry
from ai_service.tools.student.gpa import gpa_tool_definition
from ai_service.tools.rag.faculty_bylaw_search import bylaw_tool_definition
from ai_service.tools.student.schedule import schedule_tool_definition
from ai_service.tools.student.transcript import transcript_tool_definition
from ai_service.tools.student.attendance import attendance_tool_definition
from ai_service.tools.instructor.course_students import course_students_tool_definition
from ai_service.tools.instructor.student_progress import student_progress_tool_definition
from ai_service.tools.instructor.course_attendance import course_attendance_tool_definition
from ai_service.tools.admin.registration_statistics import registration_statistics_tool_definition
from ai_service.tools.admin.all_students import all_students_tool_definition

ToolRegistry.register(gpa_tool_definition)
ToolRegistry.register(bylaw_tool_definition)
ToolRegistry.register(schedule_tool_definition)
ToolRegistry.register(transcript_tool_definition)
ToolRegistry.register(attendance_tool_definition)
ToolRegistry.register(course_students_tool_definition)
ToolRegistry.register(student_progress_tool_definition)
ToolRegistry.register(course_attendance_tool_definition)
ToolRegistry.register(registration_statistics_tool_definition)
ToolRegistry.register(all_students_tool_definition)

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
    from sqlalchemy import text
    async with AsyncSessionLocal() as session:
        # Clean tables
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
                'user-999', '20269999', 'Debug User', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '99999999', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.commit()

        yield session
        await session.rollback()

async def test_debug_endpoints_gate_protection(db_session):
    """Verify that debug endpoints reject unauthorized requests with 403."""
    session_id = str(uuid.uuid4())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test without header
        r1 = await ac.get(f"/internal/debug/session/{session_id}/memory")
        assert r1.status_code == 403

        # Test with invalid header
        r2 = await ac.get(
            f"/internal/debug/session/{session_id}/memory",
            headers={"X-Internal-Key": "wrong-key"}
        )
        assert r2.status_code == 403

async def test_debug_get_tools_success(db_session):
    """Verify list of registered tools is returned with valid key."""
    from ai_service.tools.registry import ToolRegistry
    print("DEBUG TOOLS REGISTRY:", list(ToolRegistry._tools.keys()))
    transport = ASGITransport(app=app)
    headers = {"X-Internal-Key": settings.INTERNAL_API_KEY}
    
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/internal/debug/tools", headers=headers)
        assert res.status_code == 200
        data = res.json()
        print("DEBUG TOOLS DATA:", data)
        assert "registered_tools_count" in data
        assert "tools" in data
        assert any(t["name"] == "get_my_gpa" for t in data["tools"])

async def test_debug_providers_health_success(db_session):
    """Verify health endpoint returns circuit breaker and recent latencies."""
    transport = ASGITransport(app=app)
    headers = {"X-Internal-Key": settings.INTERNAL_API_KEY}
    
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/internal/debug/providers/health", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert "circuit_breaker" in data
        assert "recent_latencies" in data
        assert data["circuit_breaker"]["state"] == "CLOSED"

async def test_debug_session_memory_and_messages_success(db_session):
    """Verify session memory, message history, and traces can be inspected."""
    session_id = str(uuid.uuid4())
    user_context = UserContext(
        user_id="user-999",
        university_id="20269999",
        full_name="Debug User",
        role=UserRole.STUDENT
    )
    # Pre-create session and events
    await SessionManager.load_or_create(db_session, session_id, user_context)
    await MessagePersistence.save_message(db_session, session_id, "user", "Hello debug", "text")
    await MessagePersistence.save_message(db_session, session_id, "assistant", "Response debug", "text")
    
    # Save a mock trace
    await MessagePersistence.save_execution_trace(
        db=db_session,
        session_id=session_id,
        request_id="req-123",
        user_id="user-999",
        user_role="STUDENT",
        provider_used="openrouter",
        model_used="glm-4.5",
        provider_fallback=False,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        tool_calls_count=0,
        tools_used=[],
        latency_ms=100,
        rag_chunks_retrieved=0,
        success=True,
        error_type=None
    )
    await db_session.commit()

    transport = ASGITransport(app=app)
    headers = {"X-Internal-Key": settings.INTERNAL_API_KEY}

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test memory inspection
        r_mem = await ac.get(f"/internal/debug/session/{session_id}/memory", headers=headers)
        assert r_mem.status_code == 200
        mem_data = r_mem.json()
        assert mem_data["unsummarized_message_count"] == 2
        assert mem_data["summary_present"] is False

        # Test traces inspection
        r_trace = await ac.get(f"/internal/debug/session/{session_id}/trace?last_n=5", headers=headers)
        assert r_trace.status_code == 200
        traces = r_trace.json()
        assert len(traces) == 1
        assert traces[0]["request_id"] == "req-123"

        # Test messages inspection
        r_msg = await ac.get(f"/internal/debug/session/{session_id}/messages?include_tools=true", headers=headers)
        assert r_msg.status_code == 200
        messages = r_msg.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello debug"
        assert messages[1]["content"] == "Response debug"
