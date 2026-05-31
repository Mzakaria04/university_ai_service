from typing import List, Optional
from ai_service.models.user_context import UserContext
from ai_service.tools.base import ToolDefinition

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant for {institution_name} university management system.

You are speaking with a {role} named {user_display_name}.

Your capabilities:
{tool_capability_summary}

Guidelines:
- Always answer in the user's language.
- Use tools to retrieve live data whenever the question involves personal or institutional records.
- Use regulation/policy tools for questions about rules, policies, and handbook content.
- For general questions or greetings, respond conversationally without tools.
- Never reveal raw database values beyond what was asked.
- If a tool fails, tell the user the data is temporarily unavailable and suggest they try again.
- Never fabricate data. If you don't have the information, say so.
- when you use faculty_bylaw_search tool you must use the user's original question exactly as written. 

{memory_context_block}"""

class PromptBuilder:
    @staticmethod
    def build_system_prompt(
        user_context: UserContext,
        authorized_tools: Optional[List[ToolDefinition]] = None,
        memory_context: str = "",
    ) -> str:
        """
        Builds a role-aware system prompt for the AI assistant.
        """
        if authorized_tools is None:
            authorized_tools = []
            
        tool_summary = "\n".join(
            f"- {t.name}: {t.description.split('.')[0]}"  # First sentence only
            for t in authorized_tools
        )
        if not tool_summary:
            tool_summary = "- No tools available."
            
        role_name = user_context.role.value if hasattr(user_context.role, "value") else str(user_context.role)
        role_display = role_name.title()
        
        memory_block = f"Conversation history summary:\n{memory_context}" if memory_context else ""
        
        return SYSTEM_PROMPT_TEMPLATE.format(
            institution_name="Your University",
            role=role_display,
            user_display_name=user_context.full_name,
            tool_capability_summary=tool_summary,
            memory_context_block=memory_block,
        ).strip()

