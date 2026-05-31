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
        # 1. Load history from database
        from ai_service.memory.short_term import ShortTermMemory
        from ai_service.memory.composer import MemoryComposer
        
        memory_service = ShortTermMemory()
        history_messages = await memory_service.load(db, session_id)
        
        # Check if the last message in history is the current user query to prevent duplication or omission
        if not history_messages or history_messages[-1].role != "user" or history_messages[-1].content != user_message_content:
            history_messages.append(Message(role="user", content=user_message_content))
            
        # Get past history (everything except the current user message) for the system prompt
        past_history = history_messages[:-1]
        memory_context = MemoryComposer.compose_context_block(past_history)
        
        # 2. Retrieve authorized tools for user's role
        authorized_tools = ToolRegistry.get_authorized_tools(user_context.role)
        
        # 3. Build the system prompt
        system_prompt = PromptBuilder.build_system_prompt(
            user_context=user_context,
            authorized_tools=authorized_tools,
            memory_context=memory_context
        )
        
        # 4. Set up messages list for LLM context (system prompt + history/current user messages)
        messages = [Message(role="system", content=system_prompt)] + history_messages
        
        # 5. Agentic execution loop (max 5 iterations)
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
                
                # Persist tool_call message event to DB
                await MessagePersistence.save_message(
                    db=db,
                    session_id=session_id,
                    role=assistant_msg.role,
                    content=assistant_msg.content,
                    message_type=assistant_msg.message_type,
                    metadata_json=assistant_msg.metadata_json
                )
                
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
                        content_str = json.dumps(tool_result.data, ensure_ascii=False)
                    else:
                        content_str = json.dumps({"error": tool_result.error_message}, ensure_ascii=False)
                    
                    # Append tool response message to context history
                    tool_msg = Message(
                        role="tool",
                        content=content_str,
                        message_type="tool_result",
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name
                    )
                    messages.append(tool_msg)
                    
                    # Persist tool_result message event to DB
                    await MessagePersistence.save_message(
                        db=db,
                        session_id=session_id,
                        role=tool_msg.role,
                        content=tool_msg.content,
                        message_type=tool_msg.message_type,
                        tool_call_id=tool_msg.tool_call_id,
                        tool_name=tool_msg.tool_name
                    )
                
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
            
        # 6. Persist the final response text to database (only if we got content)
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
