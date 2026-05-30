import json
import logging
from typing import AsyncIterator, Any
from sqlalchemy.ext.asyncio import AsyncSession

from ai_service.models.messages import Message
from ai_service.models.user_context import UserContext
from ai_service.providers.base import LLMProvider
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.registry import ToolRegistry
from ai_service.persistence.message_writer import MessagePersistence
from ai_service.orchestration.prompt_builder import PromptBuilder

logger = logging.getLogger("ai_service.orchestration.conversation_orchestrator")

class ConversationOrchestrator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def orchestrate(
        self,
        db: AsyncSession,
        session_id: str,
        user_context: UserContext,
        user_message_content: str,
    ) -> AsyncIterator[str]:
        """
        Runs the bounded agentic execution loop for a chat session.
        Yields raw text chunks to stream.
        At the end of the loop, persists the final assistant response to the database.
        """
        # 1. Build the system prompt
        system_prompt = PromptBuilder.build_system_prompt(user_context)
        
        # 2. Set up initial messages list for LLM context
        # (Only the system prompt and the current user query, as per Phase 1 requirements:
        # "no memory summarization, only current user message and system prompt")
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message_content)
        ]
        
        # 3. Retrieve authorized tools for user's role
        authorized_tools = ToolRegistry.get_authorized_tools(user_context.role)
        
        # 4. Agentic execution loop (max 5 iterations)
        MAX_ITERATIONS = 5
        tool_executor = ToolExecutor(db)
        final_assistant_content = ""
        
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Orchestration loop: session={session_id}, iteration={iteration + 1}")
            
            # Call LLM provider (stream mode is used)
            response = await self.provider.chat(
                messages=messages,
                tools=authorized_tools,
                stream=True
            )
            
            # Check if the LLM returned tool calls
            if response.tool_calls:
                # Append assistant message with tool calls to context history
                assistant_msg = response.as_assistant_message()
                messages.append(assistant_msg)
                
                # Execute tool calls
                for tool_call in response.tool_calls:
                    logger.info(f"Executing tool {tool_call.name} (id={tool_call.id}) in loop")
                    tool_result = await tool_executor.execute(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        user_context=user_context
                    )
                    
                    # Serialize result content as JSON
                    if tool_result.success:
                        content_str = json.dumps(tool_result.data)
                    else:
                        content_str = json.dumps({"error": tool_result.error_message})
                    
                    # Append tool response message to context history
                    tool_msg = Message(
                        role="tool",
                        content=content_str,
                        message_type="tool_result",
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name
                    )
                    messages.append(tool_msg)
                
                # Continue loop to feed the tool responses back to the LLM
                continue
            
            # If no tool calls, this is the final response.
            # We yield text chunks dynamically
            async for chunk in response.stream():
                final_assistant_content += chunk
                yield chunk
            
            # Exit loop since final text was streamed
            break
        else:
            # If loop completed without breaking, it means we exceeded max iterations (5 tool call rounds)
            error_msg = "Agentic execution loop exceeded maximum tool calling rounds (5)."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 5. Persist the final response text to database (only if we got content)
        if final_assistant_content:
            try:
                await MessagePersistence.save_message(
                    db=db,
                    session_id=session_id,
                    role="assistant",
                    content=final_assistant_content,
                    message_type="text"
                )
            except Exception as e:
                logger.error(f"Failed to persist assistant response: {e}")
