from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class Session:
    id: str
    user_id: str
    role: str
    title: str | None = None
    summary_text: str | None = None
    last_active_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    metadata_json: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
