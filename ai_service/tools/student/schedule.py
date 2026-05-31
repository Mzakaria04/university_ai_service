from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.models.user_context import UserRole

logger = logging.getLogger("ai_service.tools.schedule")

class GetMyScheduleTool:
    """
    Tool implementation to retrieve a student's or instructor's current semester schedule.
    Queries the database to fetch sessions, rooms, times, and locations for the active term.
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
                    error_message=f"User {user_id} not found in database."
                )

            # PostgresUpperCaseEnum or raw uppercase check
            role_str = str(role).upper()

            if role_str == "STUDENT":
                # Query schedule for active term student is enrolled in
                query = text("""
                    SELECT 
                        c.code AS course_code,
                        c.name AS course_name,
                        s.name AS session_name,
                        s.type AS session_type,
                        ss."dayOfWeek" AS day_of_week,
                        ss."startTime" AS start_time,
                        ss."endTime" AS end_time,
                        ss.location AS location,
                        ss."roomId" AS room_id
                    FROM "Enrollment" e
                    JOIN "CourseOffering" co ON e."courseOfferingId" = co.id
                    JOIN "Term" t ON co."termId" = t.id
                    JOIN "Course" c ON co."courseId" = c.id
                    JOIN "Session" s ON s."courseOfferingId" = co.id
                    JOIN "SessionSchedule" ss ON ss."sessionId" = s.id
                    WHERE e."studentId" = :user_id
                      AND e.status = 'ACTIVE'
                      AND t."isActive" = true
                    ORDER BY ss."dayOfWeek", ss."startTime"
                    LIMIT 50
                """)
            elif role_str == "INSTRUCTOR":
                # Query schedule for active term course instructor teaches
                query = text("""
                    SELECT 
                        c.code AS course_code,
                        c.name AS course_name,
                        s.name AS session_name,
                        s.type AS session_type,
                        ss."dayOfWeek" AS day_of_week,
                        ss."startTime" AS start_time,
                        ss."endTime" AS end_time,
                        ss.location AS location,
                        ss."roomId" AS room_id
                    FROM "CourseInstructor" ci
                    JOIN "CourseOffering" co ON ci."courseOfferingId" = co.id
                    JOIN "Term" t ON co."termId" = t.id
                    JOIN "Course" c ON co."courseId" = c.id
                    JOIN "Session" s ON s."courseOfferingId" = co.id
                    JOIN "SessionSchedule" ss ON ss."sessionId" = s.id
                    WHERE ci."instructorId" = :user_id
                      AND t."isActive" = true
                    ORDER BY ss."dayOfWeek", ss."startTime"
                    LIMIT 50
                """)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Role {role_str} is not supported for schedules."
                )

            result = await db.execute(query, {"user_id": user_id})
            rows = result.mappings().all()

            DAYS_MAP = {
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday",
                0: "Sunday"
            }
            schedule_list = []
            for r in rows:
                start_time_val = r["start_time"]
                end_time_val = r["end_time"]
                day_val = r["day_of_week"]
                day_str = DAYS_MAP.get(day_val, str(day_val)) if isinstance(day_val, int) else str(day_val)

                schedule_list.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "session_name": r["session_name"],
                    "session_type": r["session_type"],
                    "day_of_week": day_str,
                    "start_time": start_time_val.isoformat() if hasattr(start_time_val, "isoformat") else str(start_time_val),
                    "end_time": end_time_val.isoformat() if hasattr(end_time_val, "isoformat") else str(end_time_val),
                    "location": r["location"],
                    "room_id": r["room_id"]
                })

            return ToolResult(
                success=True,
                data={"schedule": schedule_list}
            )

        except Exception as e:
            logger.error(f"Error executing get_my_schedule: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query schedule: {str(e)}"
            )

# Tool definition mapping
schedule_tool_instance = GetMyScheduleTool()

schedule_tool_definition = ToolDefinition(
    name="get_my_schedule",
    description=(
        "Retrieve the authenticated user's (student or instructor) class schedule for the current active semester, "
        "including days, times, locations, and rooms. "
        "Use this whenever the user asks for their course schedule, class times, or where they should be."
    ),
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.STUDENT, UserRole.INSTRUCTOR},
    parameters=[],  # user_id is injected contextually
    handler=schedule_tool_instance.execute,
    timeout_seconds=5.0,
    max_retries=2,
    tags=["schedule", "academic", "class"]
)
