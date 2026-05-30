import uuid
import logging
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ai_service.db.session import get_db
from ai_service.db.models import AISession

logger = logging.getLogger("ai_service.api.sessions")
router = APIRouter()

@router.post("/sessions", response_model=dict)
async def create_session(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Explicitly creates a new chat session for the authenticated user.
    """
    user_context = request.state.user_context
    session_id = str(uuid.uuid4())
    
    logger.info(f"Creating new session {session_id} for user {user_context.user_id}")
    
    # Store role as uppercase string matching database constraint format (STUDENT, INSTRUCTOR, etc.)
    role_str = user_context.role.name if hasattr(user_context.role, "name") else str(user_context.role)
    
    session_record = AISession(
        id=session_id,
        user_id=user_context.user_id,
        role=role_str.upper(),
        title="New Chat Session",
        message_count=0
    )
    db.add(session_record)
    await db.commit()
    
    return {"session_id": session_id}
