import pytest
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ai_service.db.session import AsyncSessionLocal
from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.registry import ToolRegistry
from ai_service.tools.executor import ToolExecutor
from ai_service.errors import ToolAuthorizationError

# Import tool definitions
from ai_service.tools.instructor.course_students import course_students_tool_definition
from ai_service.tools.instructor.student_progress import student_progress_tool_definition
from ai_service.tools.instructor.course_attendance import course_attendance_tool_definition
from ai_service.tools.instructor.get_my_schedule import schedule_tool_definition as instructor_schedule_tool_definition
from ai_service.tools.admin.registration_statistics import registration_statistics_tool_definition
from ai_service.tools.admin.all_students import all_students_tool_definition

# Enable asyncio testing for pytest
pytestmark = pytest.mark.anyio

# Explicitly register the tools to ensure they are available in the registry
ToolRegistry.register(course_students_tool_definition)
ToolRegistry.register(student_progress_tool_definition)
ToolRegistry.register(course_attendance_tool_definition)
ToolRegistry.register(instructor_schedule_tool_definition)
ToolRegistry.register(registration_statistics_tool_definition)
ToolRegistry.register(all_students_tool_definition)

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
        await session.execute(text('DELETE FROM "GradeRecord"'))
        await session.execute(text('DELETE FROM "GradeItem"'))
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "regulations"'))
        await session.execute(text('DELETE FROM "Program"'))
        await session.execute(text('DELETE FROM "Faculty"'))
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

        # 1. Insert Users
        await session.execute(text("""
            INSERT INTO "User" (
                id, "universityId", "fullName", role, "identityType", "identityNumber", "passwordHash", "isBanned", "createdAt", "updatedAt"
            ) VALUES 
                ('alice-student-111', '20261111', 'Alice Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '12345678', 'hash', false, NOW(), NOW()),
                ('charlie-student-333', '20263333', 'Charlie Student', 'STUDENT'::"Role", 'NATIONAL_ID'::"IdentityType", '11223344', 'hash', false, NOW(), NOW()),
                ('bob-instructor-222', '20262222', 'Bob Instructor', 'INSTRUCTOR'::"Role", 'NATIONAL_ID'::"IdentityType", '87654321', 'hash', false, NOW(), NOW()),
                ('dave-instructor-444', '20264444', 'Dave Instructor', 'INSTRUCTOR'::"Role", 'NATIONAL_ID'::"IdentityType", '44332211', 'hash', false, NOW(), NOW()),
                ('eve-admin-555', '20265555', 'Eve Admin', 'ADMIN'::"Role", 'NATIONAL_ID'::"IdentityType", '55555555', 'hash', false, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 2. Insert Building and Rooms
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

        # 3. Insert Active Term
        await session.execute(text("""
            INSERT INTO "Term" (id, name, "startDate", "endDate", "isActive", "createdAt", "updatedAt")
            VALUES ('term-1', 'Fall 2026', NOW() - INTERVAL '1 month', NOW() + INTERVAL '3 months', true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 4. Insert Courses
        await session.execute(text("""
            INSERT INTO "Course" (id, code, name, "creditHours", scope, "specialType", "isActive", "createdAt", "updatedAt")
            VALUES 
                ('course-math-101', 'MATH101', 'Calculus I', 3, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW()),
                ('course-cs-101', 'CS101', 'Intro to CS', 4, 'UNIVERSITY'::"CourseScope", 'REGULAR'::"CourseSpecialType", true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 5. Insert CourseOfferings
        await session.execute(text("""
            INSERT INTO "CourseOffering" (id, "courseId", "termId", "passingScore", "finalExamMinPercent", "isActive", "createdAt", "updatedAt")
            VALUES 
                ('co-math-1', 'course-math-101', 'term-1', 50, 30, true, NOW(), NOW()),
                ('co-cs-1', 'course-cs-101', 'term-1', 50, 30, true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 6. Insert Enrollments
        await session.execute(text("""
            INSERT INTO "Enrollment" (id, "studentId", "courseOfferingId", status, "createdAt", "updatedAt")
            VALUES 
                ('enr-math-alice', 'alice-student-111', 'co-math-1', 'ACTIVE', NOW(), NOW()),
                ('enr-math-charlie', 'charlie-student-333', 'co-math-1', 'ACTIVE', NOW(), NOW()),
                ('enr-cs-alice', 'alice-student-111', 'co-cs-1', 'ACTIVE', NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 7. Insert CourseInstructors
        await session.execute(text("""
            INSERT INTO "CourseInstructor" (
                id, "courseOfferingId", "instructorId", "roleInCourse", "isOwner", "createdAt"
            ) VALUES 
                ('ci-bob-math', 'co-math-1', 'bob-instructor-222', 'LECTURER'::"CourseInstructorRole", true, NOW()),
                ('ci-bob-cs', 'co-cs-1', 'bob-instructor-222', 'LECTURER'::"CourseInstructorRole", true, NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 8. Insert Sessions
        await session.execute(text("""
            INSERT INTO "Session" (id, "courseOfferingId", name, type, "primaryTeacherId", "maxCapacity", "createdAt", "updatedAt")
            VALUES 
                ('sess-math-1', 'co-math-1', 'Math Lecture 1', 'LECTURE', 'bob-instructor-222', 30, NOW(), NOW()),
                ('sess-cs-1', 'co-cs-1', 'CS Lab 1', 'SECTION', 'bob-instructor-222', 30, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 9. Insert SessionSchedules
        await session.execute(text("""
            INSERT INTO "SessionSchedule" (id, "sessionId", "roomId", "dayOfWeek", "startTime", "endTime", location, "createdAt", "updatedAt")
            VALUES 
                ('ss-math-1', 'sess-math-1', 'Room 101', 1, '2026-05-30 09:00:00'::timestamp, '2026-05-30 10:30:00'::timestamp, 'Building A', NOW(), NOW()),
                ('ss-cs-1', 'sess-cs-1', 'Room 202', 3, '2026-05-30 14:00:00'::timestamp, '2026-05-30 16:00:00'::timestamp, 'Building B', NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 10. Insert AttendanceSessions & Attendances
        await session.execute(text("""
            INSERT INTO "AttendanceSession" (id, "sessionId", "openedById", "weekNumber", type, radius, token, "expiresAt", "isActive", "createdAt")
            VALUES 
                ('atts-math-1', 'sess-math-1', 'bob-instructor-222', 1, 'LECTURE', 10.0, 'token-math', NOW() + INTERVAL '1 hour', true, NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "Attendance" (id, "studentId", "sessionId", "attendanceSessionId", status, "deviceId", "createdAt")
            VALUES 
                ('att-alice', 'alice-student-111', 'sess-math-1', 'atts-math-1', 'PRESENT', 'device-1', NOW()),
                ('att-charlie', 'charlie-student-333', 'sess-math-1', 'atts-math-1', 'ABSENT', 'device-2', NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 11. Insert Faculty, Program, regulations, student_profiles
        await session.execute(text("""
            INSERT INTO "Faculty" (id, name, code, "createdAt")
            VALUES ('fac-1', 'Engineering', 'ENG', NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "Program" (
                id, name, code, "facultyId", "totalCreditsForGraduation", description, "durationYears", "entryMode", "specializationStartLevel", "minGpaForJoin", "sortOrder", "isActive", "createdAt", "updatedAt"
            ) VALUES 
                ('prog-1', 'Computer Engineering', 'CCE', 'fac-1', 150, 'CCE program', 5, 'DIRECT_ENTRY'::"ProgramEntryMode", 3, 2.0, 1, true, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "regulations" (
                id, name, "facultyId", "programId", "durationYears", "totalCreditsRequired", "startBatchYear", "endBatchYear", status, description, "createdAt", "updatedAt"
            ) VALUES 
                ('reg-1', '2026 Regulation', 'fac-1', 'prog-1', 5, 150, 2026, NULL, 'ACTIVE'::"RegulationStatus", 'regulation desc', NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO student_profiles (
                id, "userId", "facultyId", "programId", "specializationStatus", "declaredAt", "regulationId", "entryYear", "currentLevel", "passedCredits", "currentGPA", "expectedGraduationYear", "advisorId", "academicStatus", "createdAt", "updatedAt"
            ) VALUES 
                ('sp-alice', 'alice-student-111', 'fac-1', 'prog-1', 'DECLARED'::"SpecializationStatus", NOW(), 'reg-1', 2026, 1, 3, 4.0, 2031, 'bob-instructor-222', 'ACTIVE'::"AcademicStatus", NOW(), NOW()),
                ('sp-charlie', 'charlie-student-333', 'fac-1', 'prog-1', 'DECLARED'::"SpecializationStatus", NOW(), 'reg-1', 2026, 2, 30, 3.5, 2030, 'bob-instructor-222', 'ACTIVE'::"AcademicStatus", NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """))

        # 12. Insert GradeItems and GradeRecords
        await session.execute(text("""
            INSERT INTO "GradeItem" (id, "courseOfferingId", name, type, "maxScore", weight, "createdAt")
            VALUES 
                ('gi-math-mid', 'co-math-1', 'Midterm Exam', 'MIDTERM'::"GradeItemType", 100.0, 30.0, NOW()),
                ('gi-math-final', 'co-math-1', 'Final Exam', 'FINAL'::"GradeItemType", 100.0, 70.0, NOW())
            ON CONFLICT DO NOTHING;
        """))
        await session.execute(text("""
            INSERT INTO "GradeRecord" (id, "studentId", "gradeItemId", score, "attemptId", "createdAt")
            VALUES 
                ('gr-alice-mid', 'alice-student-111', 'gi-math-mid', 85.0, 'att-1', NOW()),
                ('gr-alice-final', 'alice-student-111', 'gi-math-final', 90.0, 'att-2', NOW())
            ON CONFLICT DO NOTHING;
        """))

        await session.commit()
        yield session

        # Clean up database in correct reverse-dependency order
        await session.execute(text('DELETE FROM "GradeRecord"'))
        await session.execute(text('DELETE FROM "GradeItem"'))
        await session.execute(text('DELETE FROM "Transcript"'))
        await session.execute(text('DELETE FROM student_profiles'))
        await session.execute(text('DELETE FROM "regulations"'))
        await session.execute(text('DELETE FROM "Program"'))
        await session.execute(text('DELETE FROM "Faculty"'))
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


async def test_get_my_schedule_instructor(db_session):
    """Verify that get_my_schedule retrieves instructor sessions correctly."""
    instructor_ctx = UserContext(
        user_id="bob-instructor-222",
        university_id="20262222",
        full_name="Bob Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute("get_my_schedule", {}, instructor_ctx)

    assert result.success is True
    assert "schedule" in result.data
    schedule = result.data["schedule"]
    assert len(schedule) == 2
    
    # Assert schedule details
    math_sess = next(s for s in schedule if s["course_code"] == "MATH101")
    assert math_sess["session_name"] == "Math Lecture 1"
    assert math_sess["session_type"] == "LECTURE"
    assert math_sess["day_of_week"] == "Monday"


async def test_get_course_students_authorized_instructor(db_session):
    """Verify that an authorized instructor can retrieve students roster."""
    instructor_ctx = UserContext(
        user_id="bob-instructor-222",
        university_id="20262222",
        full_name="Bob Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute(
        "get_course_students",
        {"course_offering_id": "co-math-1"},
        instructor_ctx
    )

    assert result.success is True
    assert "students" in result.data
    students = result.data["students"]
    assert len(students) == 2
    student_names = [s["student_name"] for s in students]
    assert "Alice Student" in student_names
    assert "Charlie Student" in student_names


async def test_get_course_students_unauthorized_instructor(db_session):
    """Verify that an unauthorized instructor cannot retrieve students roster and gets 403."""
    instructor_ctx = UserContext(
        user_id="dave-instructor-444",
        university_id="20264444",
        full_name="Dave Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)
    
    with pytest.raises(ToolAuthorizationError):
        await executor.execute(
            "get_course_students",
            {"course_offering_id": "co-math-1"},
            instructor_ctx
        )


async def test_get_course_students_admin(db_session):
    """Verify that an admin can retrieve students roster for any course offering."""
    admin_ctx = UserContext(
        user_id="eve-admin-555",
        university_id="20265555",
        full_name="Eve Admin",
        role=UserRole.ADMIN
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute(
        "get_course_students",
        {"course_offering_id": "co-math-1"},
        admin_ctx
    )

    assert result.success is True
    assert len(result.data["students"]) == 2


async def test_get_student_progress_authorized_instructor(db_session):
    """Verify that an authorized instructor can query student progress details."""
    instructor_ctx = UserContext(
        user_id="bob-instructor-222",
        university_id="20262222",
        full_name="Bob Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute(
        "get_student_progress",
        {"student_id": "alice-student-111", "course_offering_id": "co-math-1"},
        instructor_ctx
    )

    assert result.success is True
    assert result.data["student_name"] == "Alice Student"
    grades = result.data["grades"]
    assert len(grades) == 2
    
    midterm = next(g for g in grades if g["item_name"] == "Midterm Exam")
    assert midterm["score"] == 85.0
    assert midterm["max_score"] == 100.0
    assert midterm["weight"] == 30.0


async def test_get_student_progress_unauthorized_instructor(db_session):
    """Verify that an unauthorized instructor cannot query student progress."""
    instructor_ctx = UserContext(
        user_id="dave-instructor-444",
        university_id="20264444",
        full_name="Dave Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)

    with pytest.raises(ToolAuthorizationError):
        await executor.execute(
            "get_student_progress",
            {"student_id": "alice-student-111", "course_offering_id": "co-math-1"},
            instructor_ctx
        )


async def test_get_student_progress_admin(db_session):
    """Verify that an admin can query student progress."""
    admin_ctx = UserContext(
        user_id="eve-admin-555",
        university_id="20265555",
        full_name="Eve Admin",
        role=UserRole.ADMIN
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute(
        "get_student_progress",
        {"student_id": "alice-student-111", "course_offering_id": "co-math-1"},
        admin_ctx
    )

    assert result.success is True
    assert len(result.data["grades"]) == 2


async def test_get_course_attendance_authorized_instructor(db_session):
    """Verify that an authorized instructor can retrieve course attendance stats."""
    instructor_ctx = UserContext(
        user_id="bob-instructor-222",
        university_id="20262222",
        full_name="Bob Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute(
        "get_course_attendance",
        {"course_offering_id": "co-math-1"},
        instructor_ctx
    )

    assert result.success is True
    attendance = result.data["students_attendance"]
    assert len(attendance) == 2
    
    alice = next(a for a in attendance if a["student_name"] == "Alice Student")
    assert alice["total_sessions"] == 1
    assert alice["present_count"] == 1
    assert alice["absent_count"] == 0
    assert alice["absence_percentage"] == 0.0

    charlie = next(a for a in attendance if a["student_name"] == "Charlie Student")
    assert charlie["total_sessions"] == 1
    assert charlie["present_count"] == 0
    assert charlie["absent_count"] == 1
    assert charlie["absence_percentage"] == 100.0


async def test_get_course_attendance_unauthorized_instructor(db_session):
    """Verify that an unauthorized instructor cannot retrieve course attendance."""
    instructor_ctx = UserContext(
        user_id="dave-instructor-444",
        university_id="20264444",
        full_name="Dave Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)

    with pytest.raises(ToolAuthorizationError):
        await executor.execute(
            "get_course_attendance",
            {"course_offering_id": "co-math-1"},
            instructor_ctx
        )


async def test_get_registration_statistics_admin(db_session):
    """Verify that an admin can retrieve registration statistics."""
    admin_ctx = UserContext(
        user_id="eve-admin-555",
        university_id="20265555",
        full_name="Eve Admin",
        role=UserRole.ADMIN
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute("get_registration_statistics", {}, admin_ctx)

    assert result.success is True
    assert result.data["total_students"] == 2
    assert "ACTIVE" in result.data["enrollments_by_status"]
    assert result.data["enrollments_by_status"]["ACTIVE"] == 3
    
    active_courses = result.data["active_courses_enrollment"]
    assert len(active_courses) == 2
    math_stat = next(c for c in active_courses if c["course_code"] == "MATH101")
    assert math_stat["enrollment_count"] == 2


async def test_get_registration_statistics_unauthorized(db_session):
    """Verify that students/instructors cannot retrieve registration stats."""
    student_ctx = UserContext(
        user_id="alice-student-111",
        university_id="20261111",
        full_name="Alice Student",
        role=UserRole.STUDENT
    )
    executor = ToolExecutor(db_session)

    with pytest.raises(ToolAuthorizationError):
        await executor.execute("get_registration_statistics", {}, student_ctx)


async def test_get_all_students_admin(db_session):
    """Verify that an admin can retrieve all students list with pagination."""
    admin_ctx = UserContext(
        user_id="eve-admin-555",
        university_id="20265555",
        full_name="Eve Admin",
        role=UserRole.ADMIN
    )
    executor = ToolExecutor(db_session)
    result = await executor.execute("get_all_students", {"limit": 1, "offset": 0}, admin_ctx)

    assert result.success is True
    assert result.data["total_count"] == 2
    assert len(result.data["students"]) == 1
    assert result.data["limit"] == 1
    assert result.data["offset"] == 0

    student = result.data["students"][0]
    assert student["full_name"] == "Alice Student"
    assert student["current_gpa"] == 4.0
    assert student["passed_credits"] == 3
    assert student["academic_status"] == "ACTIVE"


async def test_get_all_students_unauthorized(db_session):
    """Verify that student/instructor cannot retrieve all students."""
    instructor_ctx = UserContext(
        user_id="bob-instructor-222",
        university_id="20262222",
        full_name="Bob Instructor",
        role=UserRole.INSTRUCTOR
    )
    executor = ToolExecutor(db_session)

    with pytest.raises(ToolAuthorizationError):
        await executor.execute("get_all_students", {}, instructor_ctx)
