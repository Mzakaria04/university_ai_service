from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole
from ai_service.errors import ToolAuthorizationError

logger = logging.getLogger("ai_service.tools.course_students")

class GetCourseStudentsTool:
    """
    Tool implementation to retrieve a student roster for a course section.
    Enforces instructor assignment checks; ADMIN role has broad access.
    """

    async def execute(self, db: AsyncSession, user_id: str, course_offering_id: str) -> ToolResult:
        try:
            # 1. Determine user role in database
            role_query = text('SELECT role FROM "User" WHERE id = :user_id')
            role_res = await db.execute(role_query, {"user_id": user_id})
            role = role_res.scalar()

            if not role:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"User {user_id} not found."
                )

            role_str = str(role).upper()

            # 2. Contextual authorization check for instructors
            if role_str == "INSTRUCTOR":
                ci_query = text("""
                    SELECT 1 FROM "CourseInstructor" 
                    WHERE "courseOfferingId" = :course_offering_id 
                      AND "instructorId" = :user_id
                """)
                ci_res = await db.execute(ci_query, {"course_offering_id": course_offering_id, "user_id": user_id})
                if not ci_res.scalar():
                    raise ToolAuthorizationError(
                        f"Instructor {user_id} is not authorized to access course offering {course_offering_id}."
                    )
            elif role_str != "ADMIN":
                raise ToolAuthorizationError(f"Role {role_str} is not authorized to use get_course_students.")

            # 3. Retrieve student roster
            query = text("""
                SELECT 
                    u.id AS student_id,
                    u."fullName" AS student_name,
                    u."universityId" AS university_id,
                    e.status AS enrollment_status
                FROM "Enrollment" e
                JOIN "User" u ON e."studentId" = u.id
                WHERE e."courseOfferingId" = :course_offering_id
                ORDER BY u."fullName"
                LIMIT 100
            """)
            
            result = await db.execute(query, {"course_offering_id": course_offering_id})
            rows = result.mappings().all()

            students = []
            for r in rows:
                students.append({
                    "student_id": r["student_id"],
                    "student_name": r["student_name"],
                    "university_id": r["university_id"],
                    "enrollment_status": r["enrollment_status"]
                })

            return ToolResult(
                success=True,
                data={"course_offering_id": course_offering_id, "students": students}
            )

        except ToolAuthorizationError as e:
            raise e
        except Exception as e:
            logger.error(f"Error executing get_course_students: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query course roster: {str(e)}"
            )

# Tool definition mapping
course_students_tool_instance = GetCourseStudentsTool()

course_students_tool_definition = ToolDefinition(
    name="get_course_students",
    description=(
        "Retrieve the roster of students enrolled in a specific course offering section, "
        "including their names, student IDs, and enrollment statuses. "
        "Instructors can only retrieve rosters for sections they are assigned to teach; "
        "Admins can retrieve rosters for any section."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.INSTRUCTOR, UserRole.ADMIN},
    parameters=[
        {
            "name": "course_offering_id",
            "type": "string",
            "description": "The unique course offering ID to query student roster for.",
            "required": True
        }
    ],
    handler=course_students_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["instructor", "admin", "roster", "students"]
)
