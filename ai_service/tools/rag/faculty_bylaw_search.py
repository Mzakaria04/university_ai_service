import asyncio
import logging
from typing import Any

from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain, ToolParameter
from ai_service.models.user_context import UserRole
from ai_service.rag.pipeline_adapter import ask as rag_ask

logger = logging.getLogger("ai_service.tools.faculty_bylaw_search")

class FacultyBylawSearchTool:
    """
    Tool wrapping the university RAG pipeline to search regulations, policies, and bylaws.
    """

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """
        Executes bylaw search query.
        Runs the synchronous RAG search inside a worker thread to keep the event loop unblocked.
        """
        logger.info(f"Executing RAG bylaw search for query: {query}")
        try:
            # Execute synchronous rag.ask inside loop executor thread pool
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, rag_ask, query)
            
            return ToolResult(
                success=True,
                data=answer
            )
        except Exception as e:
            logger.error(f"RAG execution failed for query '{query}': {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to query bylaws database: {str(e)}"
            )

# Tool definition mapping faculty_bylaw_search
bylaw_tool_instance = FacultyBylawSearchTool()

bylaw_tool_definition = ToolDefinition(
    name="faculty_bylaw_search",
    description=(
        "Search and retrieve general university regulations, guidelines, course registration rules, "
        "exam bylaws, and grading policies from the faculty bylaws RAG knowledge base. "
        "Use this for questions about rules, credits, warnings, GPA requirements, graduation regulations, "
        "and administrative bylaws. Do NOT use this for student-specific personal data (such as requesting "
        "a specific student's schedule, grades, or attendance)."
    ),
    domain=ToolDomain.RAG,
    allowed_roles={UserRole.STUDENT, UserRole.INSTRUCTOR, UserRole.ADMIN},
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The natural language question or search query related to bylaws and university regulations.",
            required=True
        )
    ],
    handler=bylaw_tool_instance.execute,
    timeout_seconds=15.0,
    max_retries=1,
    tags=["rag", "bylaw", "policy", "regulations"]
)
