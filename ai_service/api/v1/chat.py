import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_service.db.session import AsyncSessionLocal
from ai_service.sessions.manager import SessionManager
from ai_service.persistence.message_writer import MessagePersistence
from ai_service.providers.openrouter import OpenRouterProvider
from ai_service.orchestration.conversation_orchestrator import ConversationOrchestrator
from ai_service.orchestration.streaming import format_sse_chunk, format_sse_done, format_sse_error

logger = logging.getLogger("ai_service.api.chat")
router = APIRouter()

class ChatPayload(BaseModel):
    message: str = Field(..., description="The user chat message")

@router.post("/chat/{session_id}")
async def chat_endpoint(
    session_id: str,
    payload: ChatPayload,
    request: Request
):
    """
    POST /api/v1/chat/{session_id}
    Streams the assistant's final response after any internal tool execution as SSE.
    """
    user_context = request.state.user_context
    user_message = payload.message
    
    logger.info(f"Received chat request for session {session_id} from user {user_context.user_id}")

    async def sse_event_generator():
        async with AsyncSessionLocal() as db:
            try:
                # 1. Load session & verify ownership
                await SessionManager.load_or_create(db, session_id, user_context)
                
                # 2. Save user message to database
                await MessagePersistence.save_message(
                    db=db,
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    message_type="text"
                )
                
                # 3. Instantiate provider and orchestrator
                provider = OpenRouterProvider()
                orchestrator = ConversationOrchestrator(provider)
                
                # 4. Stream assistant chunks
                async for chunk in orchestrator.orchestrate(
                    db=db,
                    session_id=session_id,
                    user_context=user_context,
                    user_message_content=user_message
                ):
                    yield format_sse_chunk(chunk)
                
                # 5. Terminate the SSE stream
                yield format_sse_done()
                
            except Exception as e:
                logger.error(f"Error in chat SSE stream: {e}", exc_info=True)
                yield format_sse_error(str(e))

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
