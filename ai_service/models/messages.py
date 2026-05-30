from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": str(self.arguments)
            }
        }

@dataclass
class Message:
    role: str               # user | assistant | tool | system
    content: str
    message_type: str = "text" # text | tool_call | tool_result | summary
    tool_call_id: str | None = None
    tool_name: str | None = None
    metadata_json: dict[str, Any] | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert standard Message to OpenAI API format."""
        d = {
            "role": self.role,
            "content": self.content
        }
        if self.role == "tool" and self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
            d["name"] = self.tool_name or ""
        elif self.role == "assistant" and self.metadata_json and "tool_calls" in self.metadata_json:
            d["tool_calls"] = self.metadata_json["tool_calls"]
            if not d["content"]:
                d["content"] = None
        return d
