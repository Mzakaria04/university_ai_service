from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole
from ai_service.errors import ToolAuthorizationError

logger = logging.getLogger("ai_service.tools.registration_statistics")

class GetRegistrationStatisticsTool:
    """
    Tool implementation to retrieve university-wide enrollment and registration metrics.
    Only authorized for ADMIN role.
    """

    async def execute(self, db: AsyncSession, user_id: str) -> ToolResult:
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

            # 2. Authorization check
            if role_str != "ADMIN":
                raise ToolAuthorizationError(f"Role {role_str} is not authorized to use get_registration_statistics.")

            # 3. Query total student count
            students_count_query = text('SELECT COUNT(*) FROM "User" WHERE role = \'STUDENT\'')
            students_count_res = await db.execute(students_count_query)
            total_students = students_count_res.scalar() or 0

            # 4. Query enrollment counts by status
            status_query = text('SELECT status, COUNT(*) AS count FROM "Enrollment" GROUP BY status')
            status_res = await db.execute(status_query)
            status_rows = status_res.mappings().all()
            enrollments_by_status = {r["status"]: int(r["count"]) for r in status_rows}

            # 5. Query active enrollment counts per course offering (active term)
            course_query = text("""
                SELECT 
                    c.code AS course_code, 
                    c.name AS course_name, 
                    COUNT(e.id) AS enrollment_count
                FROM "Enrollment" e
                JOIN "CourseOffering" co ON e."courseOfferingId" = co.id
                JOIN "Course" c ON co."courseId" = c.id
                JOIN "Term" t ON co."termId" = t.id
                WHERE e.status = 'ACTIVE' AND t."isActive" = true
                GROUP BY c.code, c.name
                ORDER BY enrollment_count DESC
                LIMIT 20
            """)
            course_res = await db.execute(course_query)
            course_rows = course_res.mappings().all()
            
            courses_stats = []
            for r in course_rows:
                courses_stats.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "enrollment_count": int(r["enrollment_count"])
                })

            return ToolResult(
                success=True,
                data={
                    "total_students": total_students,
                    "enrollments_by_status": enrollments_by_status,
                    "active_courses_enrollment": courses_stats
                }
            )

        except ToolAuthorizationError as e:
            raise e
        except Exception as e:
            logger.error(f"Error executing get_registration_statistics: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query registration statistics: {str(e)}"
            )

# Tool definition mapping
registration_statistics_tool_instance = GetRegistrationStatisticsTool()

registration_statistics_tool_definition = ToolDefinition(
    name="get_registration_statistics",
    description=(
        "Retrieve university-wide registration and enrollment statistics, including the total number of students, "
        "enrollment counts grouped by status, and course enrollment rankings for the active semester. "
        "Only Admins are authorized to use this tool."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.ADMIN},
    parameters=[],  # user_id is injected contextually
    handler=registration_statistics_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["admin", "statistics", "enrollment", "registration"]
)
