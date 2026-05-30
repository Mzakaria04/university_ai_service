from ai_service.models.user_context import UserContext

class PromptBuilder:
    @staticmethod
    def build_system_prompt(user_context: UserContext) -> str:
        """
        Builds a role-aware system prompt for the AI assistant.
        """
        role_str = user_context.role.name if hasattr(user_context.role, "name") else str(user_context.role)
        
        prompt = (
            f"You are the University AI Assistant. You help students, instructors, and staff with academic queries.\n"
            f"Currently interacting with User: {user_context.full_name} (Role: {role_str}, University ID: {user_context.university_id}).\n\n"
            f"Guidelines:\n"
            f"1. You must answer questions professionally and politely.\n"
            f"2. If the user asks about their GPA, credit hours, or completed courses, you MUST use the `get_my_gpa` tool to get their real data from the database. Do not hallucinate or make up any academic numbers.\n"
            f"3. Only discuss data retrieved from the tools. If a tool fails or returns no data, explain that you cannot retrieve it right now.\n"
            f"4. Refuse to perform actions or answer queries not authorized for the user's role."
        )
        return prompt
