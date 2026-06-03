import logging
import uuid
import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ai_service.db.models import AIMessageEvent, AISession
from ai_service.providers.base import LLMProvider

logger = logging.getLogger("ai_service.memory.long_term")

class LongTermMemory:
    COMPRESSION_TRIGGER_COUNT = 20  # messages
    COMPRESSION_TRIGGER_TOKENS = 6000  # tokens
    MAX_SUMMARY_WORDS = 300

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = tiktoken.encoding_for_model("gpt-4")

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    async def maybe_compress(self, session_id: str, db: AsyncSession) -> None:
        """
        Triggers summarization if unsummarized messages exceed 20 messages or 6000 tokens.
        Generates a summary via the LLM, saves it, and marks old messages as summarized.
        """
        # Fetch all unsummarized message events sorted chronologically
        query = (
            select(AIMessageEvent)
            .where(AIMessageEvent.session_id == session_id)
            .where(AIMessageEvent.is_summarized == False)
            .where(AIMessageEvent.message_type != "summary")
            .order_by(AIMessageEvent.sequence_number.asc())
        )
        result = await db.execute(query)
        unsummarized = result.scalars().all()

        if not unsummarized:
            logger.debug(f"Session {session_id}: No unsummarized messages found.")
            return

        total_tokens = sum(self._estimate_tokens(m.content) for m in unsummarized)
        total_count = len(unsummarized)

        if total_count < self.COMPRESSION_TRIGGER_COUNT and total_tokens < self.COMPRESSION_TRIGGER_TOKENS:
            logger.info(
                f"Session {session_id}: Skip compression. "
                f"Unsummarized count={total_count} (<{self.COMPRESSION_TRIGGER_COUNT}), "
                f"tokens={total_tokens} (<{self.COMPRESSION_TRIGGER_TOKENS})"
            )
            return

        logger.info(f"Session {session_id}: Compressing memory. Count={total_count}, Tokens={total_tokens}")

        # Format conversation text for prompt
        conversation_text_lines = []
        for m in unsummarized:
            role_label = m.role.upper()
            if m.message_type == "tool_call":
                content = f"[Requested Tool: {m.tool_name or 'unknown'}]"
            elif m.message_type == "tool_result":
                content = f"[Tool Result: {m.content}]"
            else:
                content = m.content
            conversation_text_lines.append(f"{role_label}: {content}")
        
        conversation_text = "\n".join(conversation_text_lines)

        # Generate summary
        prompt = f"""Summarize the following university assistant conversation.
Focus on: what data was retrieved, what policies were discussed, 
any unresolved questions, and the student/instructor's apparent goals.
Be concise. Max {self.MAX_SUMMARY_WORDS} words.

Conversation:
{conversation_text}"""

        try:
            summary_text = await self.provider.complete(prompt, max_tokens=512)
            summary_text = summary_text.strip()
        except Exception as e:
            logger.error(f"Failed to generate memory summary via LLM: {e}", exc_info=True)
            return

        if not summary_text:
            logger.warning(f"Session {session_id}: LLM returned an empty summary text.")
            return

        # Save summary message event and mark messages as summarized
        try:
            # Query max sequence number
            seq_query = (
                select(AIMessageEvent.sequence_number)
                .where(AIMessageEvent.session_id == session_id)
                .order_by(AIMessageEvent.sequence_number.desc())
                .limit(1)
            )
            seq_result = await db.execute(seq_query)
            max_seq = seq_result.scalar() or 0
            new_seq = max_seq + 1

            # Persist summary event
            summary_event = AIMessageEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="system",
                message_type="summary",
                content=summary_text,
                is_summarized=False,
                sequence_number=new_seq
            )
            db.add(summary_event)

            # Update summarized messages
            message_ids = [m.id for m in unsummarized]
            await db.execute(
                update(AIMessageEvent)
                .where(AIMessageEvent.id.in_(message_ids))
                .values(is_summarized=True)
            )

            # Update AISession's summary_text and message_count
            session_query = select(AISession).where(AISession.id == session_id)
            session_res = await db.execute(session_query)
            session_record = session_res.scalar_one_or_none()
            if session_record:
                session_record.summary_text = summary_text
                session_record.message_count += 1
            
            await db.commit()
            logger.info(f"Session {session_id}: Memory compression complete. Saved summary and marked {len(message_ids)} messages as summarized.")
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to save compressed memory to DB: {e}", exc_info=True)

    async def load_summary(self, session_id: str, db: AsyncSession) -> str | None:
        """
        Loads the latest summary text for the session.
        """
        query = (
            select(AIMessageEvent)
            .where(AIMessageEvent.session_id == session_id)
            .where(AIMessageEvent.message_type == "summary")
            .order_by(AIMessageEvent.sequence_number.desc())
            .limit(1)
        )
        result = await db.execute(query)
        event = result.scalars().first()
        return event.content if event else None
