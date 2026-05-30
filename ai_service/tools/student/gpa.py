from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole

logger = logging.getLogger("ai_service.tools.gpa")

class GetMyGpaTool:
    """
    Tool implementation to retrieve a student's GPA, completed credits, and courses from database.
    Queries the case-sensitive 'Transcript' and 'Course' tables.
    """

    async def execute(self, db: AsyncSession, user_id: str) -> ToolResult:
        query = text("""
            SELECT 
                COALESCE(SUM(t."gradePoint" * c."creditHours") / NULLIF(SUM(CASE WHEN t."includeInGpa" = true THEN c."creditHours" ELSE 0 END), 0), 0.0) AS gpa,
                COALESCE(SUM(CASE WHEN t."isPassed" = true THEN c."creditHours" ELSE 0 END), 0) AS total_credits,
                COUNT(CASE WHEN t."isPassed" = true THEN 1 END)::integer AS courses_completed
            FROM "Transcript" t
            JOIN "Course" c ON t."courseId" = c.id
            WHERE t."studentId" = :user_id
        """)
        
        try:
            result = await db.execute(query, {"user_id": user_id})
            row = result.mappings().first()
            
            if not row:
                return ToolResult(
                    success=True,
                    data={
                        "cumulative_gpa": 0.0,
                        "total_credits": 0,
                        "courses_completed": 0,
                        "message": "No academic transcript records found."
                    }
                )

            # Round GPA to 2 decimal places
            gpa_val = round(float(row["gpa"]), 2)
            total_credits_val = int(row["total_credits"])
            courses_completed_val = int(row["courses_completed"])

            return ToolResult(
                success=True,
                data={
                    "cumulative_gpa": gpa_val,
                    "total_credits": total_credits_val,
                    "courses_completed": courses_completed_val
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing get_my_gpa query: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query GPA database: {str(e)}"
            )

# Tool definition mapping get_my_gpa
gpa_tool_instance = GetMyGpaTool()

gpa_tool_definition = ToolDefinition(
    name="get_my_gpa",
    description=(
        "Retrieve the authenticated student's current cumulative GPA, total earned credits, "
        "and number of completed courses from the live database. "
        "Use this when the student asks about their own GPA, academic standing, or credit progress. "
        "Do NOT use this for GPA rules or policy questions."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.STUDENT},
    parameters=[],  # Parameters are empty because user_id is injected contextually
    handler=gpa_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["student", "gpa", "academic"]
)
