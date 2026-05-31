import logging
import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ai_service.db.models import AIMessageEvent
from ai_service.models.messages import Message

logger = logging.getLogger("ai_service.memory.short_term")

class ShortTermMemory:
    TOKEN_BUDGET = 3000
    MAX_MESSAGES = 12

    def __init__(self, token_budget: int = TOKEN_BUDGET, max_messages: int = MAX_MESSAGES):
        self.token_budget = token_budget
        self.max_messages = max_messages
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to general model encoding if offline/issues
            self.encoding = tiktoken.encoding_for_model("gpt-4")

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    async def load(self, db: AsyncSession, session_id: str) -> list[Message]:
        """
        Loads the last N messages for the session from the database,
        trims them to fit within the token budget (keeping the newest messages),
        and returns them as Message domain models in chronological order.
        """
        # Fetch the newest N messages first (ordered by sequence_number descending)
        query = (
            select(AIMessageEvent)
            .where(AIMessageEvent.session_id == session_id)
            .order_by(AIMessageEvent.sequence_number.desc())
            .limit(self.max_messages)
        )
        result = await db.execute(query)
        db_messages = result.scalars().all()
        
        allowed_messages = []
        cumulative_tokens = 0
        
        for db_msg in db_messages:
            msg_tokens = self._estimate_tokens(db_msg.content)
            
            # Check if adding this message exceeds the soft token budget
            if cumulative_tokens + msg_tokens > self.token_budget:
                logger.info(
                    f"Session {session_id}: Token budget of {self.token_budget} exceeded. "
                    f"Dropping older messages starting from sequence_number={db_msg.sequence_number}"
                )
                break
                
            cumulative_tokens += msg_tokens
            
            # Convert database model (with transparent uppercase conversion) to domain model
            allowed_messages.append(Message(
                role=db_msg.role,
                content=db_msg.content,
                message_type=db_msg.message_type,
                tool_call_id=db_msg.tool_call_id,
                tool_name=db_msg.tool_name,
                metadata_json=db_msg.metadata_json
            ))
            
        # Re-order to chronological order (oldest first) for context mapping
        allowed_messages.reverse()
        logger.info(f"Session {session_id}: Loaded {len(allowed_messages)} messages (total tokens: {cumulative_tokens})")
        return allowed_messages
