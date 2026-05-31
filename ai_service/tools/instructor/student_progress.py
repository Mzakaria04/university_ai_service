from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole
from ai_service.errors import ToolAuthorizationError

logger = logging.getLogger("ai_service.tools.student_progress")

class GetStudentProgressTool:
    """
    Tool implementation to retrieve a student's grade records and academic progress.
    Enforces instructor assignment checks; ADMIN role has broad access.
    """

    async def execute(self, db: AsyncSession, user_id: str, student_id: str, course_offering_id: str) -> ToolResult:
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
                raise ToolAuthorizationError(f"Role {role_str} is not authorized to use get_student_progress.")

            # 3. Retrieve student full name
            s_query = text('SELECT "fullName" FROM "User" WHERE id = :student_id')
            s_res = await db.execute(s_query, {"student_id": student_id})
            student_name = s_res.scalar()

            if not student_name:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Student {student_id} not found."
                )

            # 4. Retrieve student grade records
            query = text("""
                SELECT 
                    gi.name AS item_name,
                    gi.type AS item_type,
                    gr.score AS score,
                    gi."maxScore" AS max_score,
                    gi.weight AS weight
                FROM "GradeItem" gi
                LEFT JOIN "GradeRecord" gr ON gi.id = gr."gradeItemId" AND gr."studentId" = :student_id
                WHERE gi."courseOfferingId" = :course_offering_id
                ORDER BY gi.name
            """)
            
            result = await db.execute(query, {"course_offering_id": course_offering_id, "student_id": student_id})
            rows = result.mappings().all()

            grades = []
            for r in rows:
                grades.append({
                    "item_name": r["item_name"],
                    "item_type": r["item_type"],
                    "score": float(r["score"]) if r["score"] is not None else None,
                    "max_score": float(r["max_score"]),
                    "weight": float(r["weight"])
                })

            return ToolResult(
                success=True,
                data={
                    "student_id": student_id,
                    "student_name": student_name,
                    "course_offering_id": course_offering_id,
                    "grades": grades
                }
            )

        except ToolAuthorizationError as e:
            raise e
        except Exception as e:
            logger.error(f"Error executing get_student_progress: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query student progress: {str(e)}"
            )

# Tool definition mapping
student_progress_tool_instance = GetStudentProgressTool()

student_progress_tool_definition = ToolDefinition(
    name="get_student_progress",
    description=(
        "Retrieve academic progress and scores of a specific student for a course offering, "
        "including scores on individual assignments, quizzes, and exams. "
        "Instructors can only query students in sections they teach; "
        "Admins can query any student."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.INSTRUCTOR, UserRole.ADMIN},
    parameters=[
        {
            "name": "student_id",
            "type": "string",
            "description": "The unique student ID (UUID) to query progress for.",
            "required": True
        },
        {
            "name": "course_offering_id",
            "type": "string",
            "description": "The unique course offering ID to query grades for.",
            "required": True
        }
    ],
    handler=student_progress_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["instructor", "admin", "progress", "grades"]
)
