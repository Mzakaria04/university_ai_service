from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from ai_service.db.models import AISession
from ai_service.models.user_context import UserContext
from ai_service.errors import SessionOwnershipError

class SessionManager:
    @staticmethod
    async def load_or_create(
        db: AsyncSession,
        session_id: str,
        user_context: UserContext
    ) -> AISession:
        """
        Loads an existing session from the database or creates a new one lazily.
        Ensures the requesting user owns the session if it already exists.
        """
        # Query for session by ID
        query = select(AISession).where(AISession.id == session_id)
        result = await db.execute(query)
        session_record = result.scalars().first()

        if session_record:
            # Check ownership (JWT user_id must match session's user_id)
            if session_record.user_id != user_context.user_id:
                raise SessionOwnershipError(
                    f"User {user_context.user_id} is not authorized to access session {session_id}"
                )
            
            # Update last active timestamp and commit
            session_record.last_active_at = datetime.utcnow()
            await db.commit()
            return session_record

        # Create session lazily if it doesn't exist
        new_session = AISession(
            id=session_id,
            user_id=user_context.user_id,
            role=user_context.role,
            title="New Chat Session",
            message_count=0
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session
