from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain, ToolParameter
from ai_service.models.user_context import UserRole
from ai_service.errors import ToolAuthorizationError

logger = logging.getLogger("ai_service.tools.student_progress")

class GetStudentProgressTool:
    """
    Tool implementation to retrieve a student's grades and progress details in a course section.
    Instructors can only query sections they teach. Admins have global access.
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

            # 2. Authorization & Context check
            if role_str == "INSTRUCTOR":
                check_query = text("""
                    SELECT 1 FROM "CourseInstructor" 
                    WHERE "instructorId" = :user_id AND "courseOfferingId" = :course_offering_id
                """)
                check_res = await db.execute(check_query, {"user_id": user_id, "course_offering_id": course_offering_id})
                if not check_res.scalar():
                    raise ToolAuthorizationError(
                        f"Instructor {user_id} is not assigned to course offering {course_offering_id}."
                    )
            elif role_str == "ADMIN":
                # Admins have global access
                pass
            else:
                raise ToolAuthorizationError(f"Role {role_str} is not authorized to retrieve student progress.")

            # 3. Retrieve student's full name
            student_name_query = text('SELECT "fullName" FROM "User" WHERE id = :student_id')
            student_name_res = await db.execute(student_name_query, {"student_id": student_id})
            student_name = student_name_res.scalar() or "Unknown Student"

            # 4. Retrieve student grades/record
            query = text("""
                SELECT 
                    gi.name AS assessment_name,
                    gi.weight AS assessment_weight,
                    gi."maxScore" AS max_possible_grade,
                    gr.score AS grade_earned
                FROM "GradeRecord" gr
                JOIN "GradeItem" gi ON gr."gradeItemId" = gi.id
                WHERE gr."studentId" = :student_id AND gi."courseOfferingId" = :course_offering_id
            """)
            
            result = await db.execute(query, {"student_id": student_id, "course_offering_id": course_offering_id})
            rows = result.mappings().all()

            grades = []
            for r in rows:
                grades.append({
                    "assessment_name": r["assessment_name"],
                    "weight_percentage": float(r["assessment_weight"]) if r["assessment_weight"] is not None else None,
                    "max_grade": float(r["max_possible_grade"]) if r["max_possible_grade"] is not None else None,
                    "grade_earned": float(r["grade_earned"]) if r["grade_earned"] is not None else None,
                    "feedback": None
                })

            return ToolResult(
                success=True,
                data={
                    "student_id": student_id,
                    "student_name": student_name,
                    "course_offering_id": course_offering_id,
                    "assessments": grades,
                    "grades": [
                        {
                            "item_name": g["assessment_name"],
                            "score": g["grade_earned"],
                            "max_score": g["max_grade"],
                            "weight": g["weight_percentage"],
                            "feedback": g["feedback"]
                        }
                        for g in grades
                    ]
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
        "Retrieve detailed coursework, midterm, final, and overall grades for a student in a specific course offering. "
        "Instructors can only query students in sections they teach; "
        "Admins can query any student."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.INSTRUCTOR, UserRole.ADMIN},
    parameters=[
        ToolParameter(
            name="student_id",
            type="string",
            description="The unique student ID (UUID) to query progress for.",
            required=True
        ),
        ToolParameter(
            name="course_offering_id",
            type="string",
            description="The unique course offering ID to query grades for.",
            required=True
        )
    ],
    handler=student_progress_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["instructor", "admin", "progress", "grades"]
)
