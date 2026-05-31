import pytest
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ai_service.db.session import AsyncSessionLocal
from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.registry import ToolRegistry
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.student.schedule import schedule_tool_definition
from ai_service.tools.student.transcript import transcript_tool_definition
from ai_service.tools.student.attendance import attendance_tool_definition

# Enable asyncio testing for pytest
pytestmark = pytest.mark.anyio

# Explicitly register the student tools
ToolRegistry.register(schedule_tool_definition)
ToolRegistry.register(transcript_tool_definition)
ToolRegistry.register(attendance_tool_definition)

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

        # 1. Insert Users (Student Alice & Instructor Bob)
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

        # 9. Insert Transcript Grade Record
        await session.execute(text("""
            INSERT INTO "Transcript" (
                id, "studentId", "courseId", "termId", grade, "isPassed", "letterGrade", "gradePoint", "includeInGpa", "isRetake", "createdAt"
            ) VALUES 
                ('trans-math-1', 'alice-student-111', 'course-math-101', 'term-1', 95.0, true, 'A', 4.0, true, false, NOW())
            ON CONFLICT DO NOTHING;
        """))

        await session.commit()
        yield session
        
        # Clean up database in correct reverse-dependency order
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


async def test_get_my_schedule(db_session):
    """Verify that get_my_schedule retrieves student classes successfully."""
    student_ctx = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute("get_my_schedule", {}, student_ctx)

    assert result.success is True
    data = result.data
    assert "schedule" in data
    schedule = data["schedule"]
    assert len(schedule) == 2

    # Assert alphabetical ordering by day of week or day order (MONDAY before WEDNESDAY)
    assert schedule[0]["course_code"] == "MATH101"
    assert schedule[0]["day_of_week"] == "Monday"
    assert schedule[0]["location"] == "Building A"
    assert schedule[0]["room_id"] == "Room 101"
    assert "09:00:00" in schedule[0]["start_time"]

    assert schedule[1]["course_code"] == "CS101"
    assert schedule[1]["day_of_week"] == "Wednesday"
    assert "14:00:00" in schedule[1]["start_time"]


async def test_get_my_transcript(db_session):
    """Verify that get_my_transcript retrieves past grades successfully."""
    student_ctx = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute("get_my_transcript", {}, student_ctx)

    assert result.success is True
    data = result.data
    assert "transcript" in data
    transcript = data["transcript"]
    assert len(transcript) == 1
    
    entry = transcript[0]
    assert entry["course_code"] == "MATH101"
    assert entry["term_name"] == "Fall 2026"
    assert entry["grade"] == 95.0
    assert entry["letter_grade"] == "A"
    assert entry["grade_point"] == 4.0
    assert entry["credit_hours"] == 3
    assert entry["is_passed"] is True


async def test_get_my_attendance(db_session):
    """Verify that get_my_attendance retrieves attendance summary metrics correctly."""
    student_ctx = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute("get_my_attendance", {}, student_ctx)

    assert result.success is True
    data = result.data
    assert "attendance" in data
    attendance = data["attendance"]
    assert len(attendance) == 2

    # Math details
    math_att = next(x for x in attendance if x["course_code"] == "MATH101")
    assert math_att["total_sessions"] == 1
    assert math_att["present_count"] == 1
    assert math_att["absent_count"] == 0
    assert math_att["absence_percentage"] == 0.0

    # CS details
    cs_att = next(x for x in attendance if x["course_code"] == "CS101")
    assert cs_att["total_sessions"] == 1
    assert cs_att["present_count"] == 0
    assert cs_att["absent_count"] == 1
    assert cs_att["absence_percentage"] == 100.0
