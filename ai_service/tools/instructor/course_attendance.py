from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole
from ai_service.errors import ToolAuthorizationError

logger = logging.getLogger("ai_service.tools.course_attendance")

class GetCourseAttendanceTool:
    """
    Tool implementation to retrieve a section's attendance summary roster.
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
                raise ToolAuthorizationError(f"Role {role_str} is not authorized to use get_course_attendance.")

            # 3. Retrieve section students' attendance summaries
            query = text("""
                SELECT 
                    u.id AS student_id,
                    u."fullName" AS student_name,
                    COUNT(a.id) AS total_sessions,
                    SUM(CASE WHEN a.status = 'PRESENT' THEN 1 ELSE 0 END) AS present_count,
                    SUM(CASE WHEN a.status = 'ABSENT' THEN 1 ELSE 0 END) AS absent_count,
                    COALESCE(
                        (SUM(CASE WHEN a.status = 'ABSENT' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(a.id), 0)) * 100.0,
                        0.0
                    ) AS absence_percentage
                FROM "Enrollment" e
                JOIN "User" u ON e."studentId" = u.id
                LEFT JOIN "Session" s ON s."courseOfferingId" = e."courseOfferingId"
                LEFT JOIN "Attendance" a ON a."sessionId" = s.id AND a."studentId" = e."studentId"
                WHERE e."courseOfferingId" = :course_offering_id
                  AND e.status = 'ACTIVE'
                GROUP BY u.id, u."fullName"
                ORDER BY u."fullName"
                LIMIT 100
            """)
            
            result = await db.execute(query, {"course_offering_id": course_offering_id})
            rows = result.mappings().all()

            students_attendance = []
            for r in rows:
                students_attendance.append({
                    "student_id": r["student_id"],
                    "student_name": r["student_name"],
                    "total_sessions": int(r["total_sessions"]),
                    "present_count": int(r["present_count"]),
                    "absent_count": int(r["absent_count"]),
                    "absence_percentage": round(float(r["absence_percentage"]), 2)
                })

            return ToolResult(
                success=True,
                data={
                    "course_offering_id": course_offering_id,
                    "students_attendance": students_attendance
                }
            )

        except ToolAuthorizationError as e:
            raise e
        except Exception as e:
            logger.error(f"Error executing get_course_attendance: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query course attendance roster: {str(e)}"
            )

# Tool definition mapping
course_attendance_tool_instance = GetCourseAttendanceTool()

course_attendance_tool_definition = ToolDefinition(
    name="get_course_attendance",
    description=(
        "Retrieve attendance summary statistics for all students enrolled in a specific course offering section, "
        "including total sessions, presence count, absence count, and absence percentage. "
        "Instructors can only query sections they teach; "
        "Admins can query any section."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.INSTRUCTOR, UserRole.ADMIN},
    parameters=[
        {
            "name": "course_offering_id",
            "type": "string",
            "description": "The unique course offering ID to query student attendance summary for.",
            "required": True
        }
    ],
    handler=course_attendance_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["instructor", "admin", "attendance", "absences"]
)
