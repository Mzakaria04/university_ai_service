import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ai_service.db.session import get_db
from ai_service.db.models import AIMessageEvent, AISession, AIFeedback

logger = logging.getLogger("ai_service.api.feedback")
router = APIRouter()

class FeedbackPayload(BaseModel):
    message_event_id: str = Field(..., description="The unique ID of the message event to rate")
    rating: int = Field(..., ge=0, le=1, description="Rating rating: 1 for positive, 0 for negative")
    comment: str | None = Field(None, description="Optional text feedback or explanation")

@router.post("/feedback")
async def submit_feedback(
    payload: FeedbackPayload,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    POST /api/v1/feedback
    Submits feedback (thumbs up/down rating and comment) for a specific AI message event.
    Validates ownership of the session associated with the message event.
    """
    user_context = request.state.user_context
    user_id = user_context.user_id

    logger.info(f"User {user_id} submitting feedback for message event {payload.message_event_id}")

    # 1. Retrieve the message event and join session to check ownership
    query = (
        select(AIMessageEvent, AISession)
        .join(AISession, AIMessageEvent.session_id == AISession.id)
        .where(AIMessageEvent.id == payload.message_event_id)
    )
    res = await db.execute(query)
    row = res.first()

    if not row:
        logger.warning(f"Feedback submission failed: message event {payload.message_event_id} not found.")
        raise HTTPException(status_code=404, detail="Message event not found.")

    message_event, session = row

    # 2. Enforce session ownership validation
    if session.user_id != user_id:
        logger.warning(f"Unauthorized feedback attempt: user {user_id} tried to rate message in session {session.id} owned by {session.user_id}")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Cannot submit feedback for message events of another user's session."
        )

    # 3. Save feedback record to the database
    is_positive = (payload.rating == 1)
    feedback_record = AIFeedback(
        id=str(uuid.uuid4()),
        message_event_id=payload.message_event_id,
        user_id=user_id,
        is_positive=is_positive,
        comment=payload.comment,
        created_at=datetime.utcnow()
    )
    db.add(feedback_record)
    await db.commit()

    logger.info(f"Feedback successfully saved with ID {feedback_record.id}")

    return {
        "status": "success",
        "feedback_id": feedback_record.id,
        "message_event_id": feedback_record.message_event_id,
        "is_positive": feedback_record.is_positive
    }
