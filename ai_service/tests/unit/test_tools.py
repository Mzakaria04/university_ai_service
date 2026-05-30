import pytest
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from ai_service.db.session import AsyncSessionLocal
from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.registry import ToolRegistry
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.base import ToolResult
from ai_service.errors import ToolAuthorizationError

# Enable asyncio testing for pytest
pytestmark = pytest.mark.anyio

# Register the tool for test execution
from ai_service.tools.student.gpa import gpa_tool_definition
ToolRegistry.register(gpa_tool_definition)

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
        # Clean up database to avoid unique constraints / FK violations from other tests
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
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES (
                'bob-instructor-222', '20262222', 'Bob Instructor', 'INSTRUCTOR'::"Role", 'NATIONAL_ID'::"IdentityType", '87654321', 'hash', false, NOW(), NOW()
            ) ON CONFLICT DO NOTHING;
        """))
        await session.commit()
        
        yield session
        await session.rollback()


async def test_tool_registry_authorization():
    """Verify that roles have correct tool permissions in registry."""
    # Student has access to get_my_gpa
    assert ToolRegistry.is_authorized("get_my_gpa", UserRole.STUDENT) is True
    # Instructor does not
    assert ToolRegistry.is_authorized("get_my_gpa", UserRole.INSTRUCTOR) is False
    # Admin does not in Phase 1
    assert ToolRegistry.is_authorized("get_my_gpa", UserRole.ADMIN) is False


async def test_tool_executor_role_enforcement(db_session):
    """Verify that ToolExecutor raises ToolAuthorizationError for unauthorized roles."""
    student_ctx = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    instructor_ctx = UserContext(
        user_id="bob-instructor-222",
        university_id="20262222",
        full_name="Bob Instructor",
        role=UserRole.INSTRUCTOR
    )

    executor = ToolExecutor(db_session)

    # Student should proceed to execution (returns successful mock, or queries empty DB)
    result = await executor.execute("get_my_gpa", {}, student_ctx)
    assert result.success is True

    # Instructor should fail immediately with ToolAuthorizationError
    with pytest.raises(ToolAuthorizationError):
        await executor.execute("get_my_gpa", {}, instructor_ctx)


async def test_gpa_calculation_query(db_session):
    """Verify that the GPA tool correctly calculates metrics from Course and Transcript tables."""
    # 1. Insert dummy courses
    await db_session.execute(text("""
        INSERT INTO "Course" (id, code, name, "creditHours", scope, "specialType", "isActive", "createdAt", "updatedAt")
        VALUES 
            ('course-math-101', 'MATH101', 'Calculus I', 3, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW()),
            ('course-cs-101', 'CS101', 'Intro to CS', 4, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW()),
            ('course-phy-101', 'PHY101', 'Physics I', 3, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW())
        ON CONFLICT DO NOTHING;
    """))

    # 1.5. Insert dummy term
    await db_session.execute(text("""
        INSERT INTO "Term" (id, name, "startDate", "endDate", "isActive", "createdAt", "updatedAt")
        VALUES ('term-1', 'Fall 2026', NOW(), NOW(), true, NOW(), NOW())
        ON CONFLICT DO NOTHING;
    """))

    # 2. Insert transcript records for Alice
    # Physics is failed (isPassed = false), Math (A, 4.0 GP) and CS (B, 3.0 GP) are passed
    await db_session.execute(text("""
        INSERT INTO "Transcript" (
            id, "studentId", "courseId", "termId", grade, "isPassed", "letterGrade", "gradePoint", "includeInGpa", "isRetake", "createdAt"
        ) VALUES 
            ('trans-1', 'alice-student-111', 'course-math-101', 'term-1', 95.0, true, 'A', 4.0, true, false, NOW()),
            ('trans-2', 'alice-student-111', 'course-cs-101', 'term-1', 85.0, true, 'B', 3.0, true, false, NOW()),
            ('trans-3', 'alice-student-111', 'course-phy-101', 'term-1', 45.0, false, 'F', 0.0, true, false, NOW())
        ON CONFLICT DO NOTHING;
    """))
    await db_session.commit()

    student_ctx = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )

    executor = ToolExecutor(db_session)
    result = await executor.execute("get_my_gpa", {}, student_ctx)

    assert result.success is True
    data = result.data
    
    # Cumulative GPA = (4.0 * 3 + 3.0 * 4 + 0.0 * 3) / (3 + 4 + 3) = (12 + 12) / 10 = 2.4
    assert data["cumulative_gpa"] == 2.4
    
    # Total Credits passed = 3 (Math) + 4 (CS) = 7 credits
    assert data["total_credits"] == 7
    
    # Completed courses = 2 (Math & CS passed)
    assert data["courses_completed"] == 2
