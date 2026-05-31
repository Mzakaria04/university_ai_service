import uuid
import logging
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ai_service.db.session import get_db
from ai_service.db.models import AISession, AIMessageEvent
from ai_service.errors import SessionNotFoundError, SessionOwnershipError

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


@router.get("/sessions/{session_id}", response_model=dict)
async def get_session_history(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the full active history of a chat session, restricted to the owner.
    """
    user_context = request.state.user_context
    logger.info(f"Retrieving session history for {session_id} by user {user_context.user_id}")
    
    query = select(AISession).where(AISession.id == session_id)
    result = await db.execute(query)
    session_record = result.scalars().first()
    
    if not session_record:
        raise SessionNotFoundError(f"Session {session_id} not found")
        
    if session_record.user_id != user_context.user_id:
        raise SessionOwnershipError(f"User {user_context.user_id} is not authorized to access session {session_id}")
        
    # Check if soft deleted
    if session_record.metadata_json and session_record.metadata_json.get("is_deleted"):
        raise SessionNotFoundError(f"Session {session_id} not found")
        
    # Retrieve messages ordered by sequence_number
    msg_query = (
        select(AIMessageEvent)
        .where(AIMessageEvent.session_id == session_id)
        .order_by(AIMessageEvent.sequence_number.asc())
    )
    msg_result = await db.execute(msg_query)
    messages = msg_result.scalars().all()
    
    message_list = []
    for msg in messages:
        message_list.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "message_type": msg.message_type,
            "tool_call_id": msg.tool_call_id,
            "tool_name": msg.tool_name,
            "metadata_json": msg.metadata_json,
            "sequence_number": msg.sequence_number,
            "created_at": msg.created_at.isoformat() if msg.created_at else None
        })
        
    return {
        "session_id": session_record.id,
        "message_count": session_record.message_count,
        "messages": message_list
    }


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Soft-deletes the chat session by setting is_deleted flag in metadata_json.
    """
    user_context = request.state.user_context
    logger.info(f"Soft-deleting session {session_id} by user {user_context.user_id}")
    
    query = select(AISession).where(AISession.id == session_id)
    result = await db.execute(query)
    session_record = result.scalars().first()
    
    if not session_record:
        raise SessionNotFoundError(f"Session {session_id} not found")
        
    if session_record.user_id != user_context.user_id:
        raise SessionOwnershipError(f"User {user_context.user_id} is not authorized to access session {session_id}")
        
    # Check if already soft deleted
    if session_record.metadata_json and session_record.metadata_json.get("is_deleted"):
        raise SessionNotFoundError(f"Session {session_id} not found")
        
    metadata = session_record.metadata_json or {}
    new_metadata = dict(metadata)
    new_metadata["is_deleted"] = True
    session_record.metadata_json = new_metadata
    
    await db.commit()
    return {"status": "deleted"}

