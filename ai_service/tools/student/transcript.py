from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole

logger = logging.getLogger("ai_service.tools.transcript")

class GetMyTranscriptTool:
    """
    Tool implementation to retrieve a student's final grades and credit hours.
    Queries the case-sensitive 'Transcript', 'Course', and 'Term' tables.
    """

    async def execute(self, db: AsyncSession, user_id: str) -> ToolResult:
        query = text("""
            SELECT 
                term.name AS term_name,
                c.code AS course_code,
                c.name AS course_name,
                t.grade AS grade,
                t."letterGrade" AS letter_grade,
                t."gradePoint" AS grade_point,
                c."creditHours" AS credit_hours,
                t."isPassed" AS is_passed
            FROM "Transcript" t
            JOIN "Course" c ON t."courseId" = c.id
            JOIN "Term" term ON t."termId" = term.id
            WHERE t."studentId" = :user_id
            ORDER BY term."startDate" ASC, c.code ASC
            LIMIT 100
        """)
        
        try:
            result = await db.execute(query, {"user_id": user_id})
            rows = result.mappings().all()
            
            transcript_list = []
            for r in rows:
                transcript_list.append({
                    "term_name": r["term_name"],
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "grade": float(r["grade"]) if r["grade"] is not None else None,
                    "letter_grade": r["letter_grade"],
                    "grade_point": float(r["grade_point"]) if r["grade_point"] is not None else None,
                    "credit_hours": int(r["credit_hours"]),
                    "is_passed": bool(r["is_passed"])
                })
                
            return ToolResult(
                success=True,
                data={"transcript": transcript_list}
            )
            
        except Exception as e:
            logger.error(f"Error executing get_my_transcript: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query transcript: {str(e)}"
            )

# Tool definition mapping
transcript_tool_instance = GetMyTranscriptTool()

transcript_tool_definition = ToolDefinition(
    name="get_my_transcript",
    description=(
        "Retrieve the authenticated student's academic transcript containing all final grades, "
        "letter grades, grade points, and course credit hours sorted chronologically by term. "
        "Use this when the student asks for their grades, transcript, past performance, or course passing status."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.STUDENT},
    parameters=[],  # user_id is injected contextually
    handler=transcript_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["transcript", "academic", "grades"]
)
