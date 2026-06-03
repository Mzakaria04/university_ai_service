from typing import Any
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

        try:
            await db.commit()
        except Exception as e:
            import logging
            debug_logger = logging.getLogger("ai_service.debug")
            try:
                # Query all sessions in DB using raw SQL to bypass ORM state issues
                from sqlalchemy import text
                res = await db.execute(text("SELECT id, user_id FROM ai_sessions"))
                sessions_in_db = res.mappings().all()
                debug_logger.error(f"DEBUG SAVE_MESSAGE ERROR: sessions in DB: {sessions_in_db}, trying to insert session_id: {session_id}")
            except Exception as query_err:
                debug_logger.error(f"DEBUG SAVE_MESSAGE ERROR: failed to query sessions: {query_err}")
            raise e
        await db.refresh(event)
        return event

    @staticmethod
    async def save_execution_trace(
        db: AsyncSession,
        session_id: str,
        request_id: str,
        user_id: str,
        user_role: str,
        provider_used: str,
        model_used: str,
        provider_fallback: bool,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        tool_calls_count: int,
        tools_used: list | None,
        latency_ms: int,
        rag_chunks_retrieved: int | None,
        success: bool,
        error_type: str | None
    ) -> Any:
        """
        Persists a new execution trace to the database.
        """
        from ai_service.db.models import AIExecutionTrace
        trace_record = AIExecutionTrace(
            id=str(uuid.uuid4()),
            session_id=session_id,
            request_id=request_id,
            user_id=user_id,
            user_role=user_role,
            provider_used=provider_used,
            model_used=model_used,
            provider_fallback=provider_fallback,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls_count=tool_calls_count,
            tools_used=tools_used,
            latency_ms=latency_ms,
            rag_chunks_retrieved=rag_chunks_retrieved,
            success=success,
            error_type=error_type
        )
        db.add(trace_record)
        await db.commit()
        await db.refresh(trace_record)
        return trace_record
