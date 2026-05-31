from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole

logger = logging.getLogger("ai_service.tools.attendance")

class GetMyAttendanceTool:
    """
    Tool implementation to retrieve a student's attendance summary and absence percentages.
    Queries the 'Enrollment', 'CourseOffering', 'Course', 'Session', and 'Attendance' tables.
    """

    async def execute(self, db: AsyncSession, user_id: str) -> ToolResult:
        query = text("""
            SELECT 
                c.code AS course_code,
                c.name AS course_name,
                COUNT(a.id) AS total_sessions,
                SUM(CASE WHEN a.status = 'PRESENT' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN a.status = 'ABSENT' THEN 1 ELSE 0 END) AS absent_count,
                COALESCE(
                    (SUM(CASE WHEN a.status = 'ABSENT' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(a.id), 0)) * 100.0,
                    0.0
                ) AS absence_percentage
            FROM "Enrollment" e
            JOIN "CourseOffering" co ON e."courseOfferingId" = co.id
            JOIN "Course" c ON co."courseId" = c.id
            JOIN "Term" t ON co."termId" = t.id AND t."isActive" = true
            LEFT JOIN "Session" s ON s."courseOfferingId" = co.id
            LEFT JOIN "Attendance" a ON a."sessionId" = s.id AND a."studentId" = e."studentId"
            WHERE e."studentId" = :user_id
              AND e.status = 'ACTIVE'
            GROUP BY c.id, c.code, c.name
            ORDER BY c.code
            LIMIT 50
        """)
        
        try:
            result = await db.execute(query, {"user_id": user_id})
            rows = result.mappings().all()
            
            attendance_list = []
            for r in rows:
                attendance_list.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "total_sessions": int(r["total_sessions"]),
                    "present_count": int(r["present_count"]),
                    "absent_count": int(r["absent_count"]),
                    "absence_percentage": round(float(r["absence_percentage"]), 2)
                })
                
            return ToolResult(
                success=True,
                data={"attendance": attendance_list}
            )
            
        except Exception as e:
            logger.error(f"Error executing get_my_attendance: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query attendance: {str(e)}"
            )

# Tool definition mapping
attendance_tool_instance = GetMyAttendanceTool()

attendance_tool_definition = ToolDefinition(
    name="get_my_attendance",
    description=(
        "Retrieve the authenticated student's attendance summary for current active term courses, "
        "including total sessions, presence count, absence count, and absence percentage. "
        "Use this when the student asks about their class attendance, absences, or if they have attendance warnings."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.STUDENT},
    parameters=[],  # user_id is injected contextually
    handler=attendance_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["attendance", "academic", "absences"]
)
