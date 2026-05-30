from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from ai_service.models.user_context import UserRole

class ToolDomain(str, Enum):
    DATABASE = "database"
    RAG = "rag"
    UTILITY = "utility"

@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "integer", "boolean"
    description: str
    required: bool = True
    enum_values: list[str] | None = None

@dataclass
class ToolResult:
    success: bool
    data: Any
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolDefinition:
    name: str
    description: str
    domain: ToolDomain
    allowed_roles: set[UserRole]
    parameters: list[ToolParameter]
    handler: Callable[..., Coroutine[Any, Any, ToolResult]]
    timeout_seconds: float = 10.0
    max_retries: int = 2
    tags: list[str] = field(default_factory=list)

    def to_llm_schema(self) -> dict[str, Any]:
        """Serialize to OpenAI-compatible tool schema."""
        properties = {}
        for p in self.parameters:
            prop = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum_values:
                prop["enum"] = p.enum_values
            properties[p.name] = prop

        required = [p.name for p in self.parameters if p.required]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
