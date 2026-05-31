import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ENUM
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy.types as types

Base = declarative_base()

class PostgresUpperCaseEnum(types.TypeDecorator):
    """
    Custom SQLAlchemy TypeDecorator to transparently handle uppercase PostgreSQL enums.
    Binds lowercase python strings as uppercase to the database,
    and returns database uppercase values as lowercase python strings.
    """
    impl = types.String
    cache_ok = True

    def __init__(self, enum_name: str, values: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum_impl = ENUM(
            *values,
            name=enum_name,
            create_type=False
        )

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(self.enum_impl)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # E.g. 'user' -> 'USER'
        if hasattr(value, "value"):
            return value.value.upper()
        return str(value).upper()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # E.g. 'USER' -> 'user'
        return str(value).lower()


class AISession(Base):
    __tablename__ = "ai_sessions"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Text, nullable=False)  # References User.id
    role = Column(String, nullable=False)   # E.g. STUDENT, INSTRUCTOR, ADMIN
    title = Column(String(255), nullable=True)
    summary_text = Column(Text, nullable=True)
    last_active_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, nullable=False, default=0)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship(
        "AIMessageEvent",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AIMessageEvent.sequence_number"
    )


class AIMessageEvent(Base):
    __tablename__ = "ai_message_events"

    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(Text, ForeignKey("ai_sessions.id", ondelete="CASCADE"), nullable=False)
    
    # role can be user, assistant, tool, system (database expects USER, ASSISTANT, etc.)
    role = Column(
        PostgresUpperCaseEnum("ai_message_role", ["USER", "ASSISTANT", "TOOL", "SYSTEM"]),
        nullable=False
    )
    
    # message_type can be text, tool_call, tool_result, summary, system_instruction
    message_type = Column(
        PostgresUpperCaseEnum("ai_message_type", ["TEXT", "TOOL_CALL", "TOOL_RESULT", "SUMMARY", "SYSTEM_INSTRUCTION"]),
        nullable=False
    )
    
    content = Column(Text, nullable=False)
    tool_call_id = Column(String(100), nullable=True)
    tool_name = Column(String(100), nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    is_summarized = Column(Boolean, nullable=False, default=False)
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    session = relationship("AISession", back_populates="messages")


class AIFeedback(Base):
    __tablename__ = "ai_feedback"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_event_id = Column(Text, ForeignKey("ai_message_events.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Text, nullable=False)  # References User.id
    is_positive = Column(Boolean, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    message_event = relationship("AIMessageEvent")
