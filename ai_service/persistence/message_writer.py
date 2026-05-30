from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
import uuid

from ai_service.db.models import AIMessageEvent, AISession

class MessagePersistence:
    @staticmethod
    async def save_message(
        db: AsyncSession,
        session_id: str,
        role: str,              # user | assistant | tool | system
        content: str,
        message_type: str = "text", # text | tool_call | tool_result | summary
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        metadata_json: dict | None = None
    ) -> AIMessageEvent:
        """
        Persists a new message event to the database.
        Automatically calculates the next monotonic sequence number for the session
        and updates the message count in the AISession table.
        """
        # Determine the next sequence number in a thread-safe way for this session
        query = select(func.coalesce(func.max(AIMessageEvent.sequence_number), 0)).where(
            AIMessageEvent.session_id == session_id
        )
        result = await db.execute(query)
        max_seq = result.scalar() or 0
        next_seq = max_seq + 1

        # Instantiate message event ORM record
        event = AIMessageEvent(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            message_type=message_type,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            metadata_json=metadata_json,
            is_summarized=False,
            sequence_number=next_seq
        )
        db.add(event)

        # Update the AISession message_count and last_active_at
        session_query = select(AISession).where(AISession.id == session_id)
        session_result = await db.execute(session_query)
        session_record = session_result.scalars().first()
        if session_record:
            session_record.message_count = next_seq
            session_record.last_active_at = func.now()

        await db.commit()
        await db.refresh(event)
        return event
