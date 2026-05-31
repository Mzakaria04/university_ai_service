from ai_service.models.user_context import UserRole
from ai_service.tools.base import ToolDefinition

ROLE_TOOL_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.STUDENT: {
        "get_my_gpa",
        "get_my_schedule",
        "get_my_transcript",
        "get_my_attendance",
        "faculty_bylaw_search",
    },
    UserRole.INSTRUCTOR: {
        "get_course_students",
        "get_student_progress",
        "get_course_attendance",
        "get_my_schedule",
        "faculty_bylaw_search",
    },
    UserRole.ADMIN: {
        "get_registration_statistics",
        "get_all_students",
        "get_course_students",
        "get_student_progress",
        "get_course_attendance",
        "faculty_bylaw_search",
    },
}

class ToolRegistry:
    _tools: dict[str, ToolDefinition] = {}

    @classmethod
    def register(cls, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        cls._tools[tool.name] = tool

    @classmethod
    def get_authorized_tools(cls, role: UserRole) -> list[ToolDefinition]:
        """Get all tool definitions authorized for the specified role."""
        allowed_names = ROLE_TOOL_PERMISSIONS.get(role, set())
        return [t for name, t in cls._tools.items() if name in allowed_names]

    @classmethod
    def is_authorized(cls, tool_name: str, role: UserRole) -> bool:
        """Check if a specific tool is authorized for the given role."""
        return tool_name in ROLE_TOOL_PERMISSIONS.get(role, set())

    @classmethod
    def get(cls, tool_name: str) -> ToolDefinition | None:
        """Retrieve a registered tool definition by name."""
        return cls._tools.get(tool_name)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (useful for unit tests)."""
        cls._tools.clear()
