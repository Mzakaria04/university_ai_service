from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole
from ai_service.errors import ToolAuthorizationError

logger = logging.getLogger("ai_service.tools.all_students")

class GetAllStudentsTool:
    """
    Tool implementation to retrieve a paginated list of all students.
    Only authorized for ADMIN role.
    """

    async def execute(self, db: AsyncSession, user_id: str, limit: int = 50, offset: int = 0) -> ToolResult:
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
                raise ToolAuthorizationError(f"Role {role_str} is not authorized to use get_all_students.")

            # 3. Parse and sanitize limit/offset
            try:
                limit_val = int(limit)
            except (ValueError, TypeError):
                limit_val = 50
            
            try:
                offset_val = int(offset)
            except (ValueError, TypeError):
                offset_val = 0

            if limit_val <= 0:
                limit_val = 50
            elif limit_val > 100:
                limit_val = 100

            if offset_val < 0:
                offset_val = 0

            # 4. Query total count of students
            count_query = text('SELECT COUNT(*) FROM "User" WHERE role = \'STUDENT\'')
            count_res = await db.execute(count_query)
            total_count = count_res.scalar() or 0

            # 5. Query students list
            query = text("""
                SELECT 
                    u.id AS student_id,
                    u."universityId" AS university_id,
                    u."fullName" AS full_name,
                    u.email AS email,
                    u.phone AS phone,
                    sp."entryYear" AS entry_year,
                    sp."currentLevel" AS current_level,
                    sp."passedCredits" AS passed_credits,
                    sp."currentGPA" AS current_gpa,
                    sp."academicStatus" AS academic_status,
                    u."isBanned" AS is_banned
                FROM "User" u
                LEFT JOIN student_profiles sp ON u.id = sp."userId"
                WHERE u.role = 'STUDENT'
                ORDER BY u."fullName"
                LIMIT :limit OFFSET :offset
            """)
            
            result = await db.execute(query, {"limit": limit_val, "offset": offset_val})
            rows = result.mappings().all()

            students = []
            for r in rows:
                students.append({
                    "student_id": r["student_id"],
                    "university_id": r["university_id"],
                    "full_name": r["full_name"],
                    "email": r["email"],
                    "phone": r["phone"],
                    "entry_year": int(r["entry_year"]) if r["entry_year"] is not None else None,
                    "current_level": int(r["current_level"]) if r["current_level"] is not None else None,
                    "passed_credits": int(r["passed_credits"]) if r["passed_credits"] is not None else None,
                    "current_gpa": float(r["current_gpa"]) if r["current_gpa"] is not None else None,
                    "academic_status": str(r["academic_status"]) if r["academic_status"] is not None else None,
                    "is_banned": bool(r["is_banned"])
                })

            return ToolResult(
                success=True,
                data={
                    "students": students,
                    "total_count": total_count,
                    "limit": limit_val,
                    "offset": offset_val
                }
            )

        except ToolAuthorizationError as e:
            raise e
        except Exception as e:
            logger.error(f"Error executing get_all_students: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query all students: {str(e)}"
            )

# Tool definition mapping
all_students_tool_instance = GetAllStudentsTool()

all_students_tool_definition = ToolDefinition(
    name="get_all_students",
    description=(
        "Retrieve a paginated list of all registered students in the system with their general and academic profiles. "
        "Use this when listing students or searching/browsing student profiles. "
        "Only Admins are authorized to use this tool."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.ADMIN},
    parameters=[
        {
            "name": "limit",
            "type": "integer",
            "description": "Maximum number of students to return (default 50, max 100).",
            "required": False
        },
        {
            "name": "offset",
            "type": "integer",
            "description": "Number of students to skip for pagination (default 0).",
            "required": False
        }
    ],
    handler=all_students_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["admin", "students", "list", "directory"]
)
