from typing import Any
from ai_service.models.messages import Message

class MemoryComposer:
    @staticmethod
    def compose_context_block(messages: list[Message]) -> str:
        """
        Formats a list of Message objects into a readable, chronological conversation history text block
        to be injected into the system prompt.
        """
        if not messages:
            return ""
        
        lines = ["[Recent Conversation History]"]
        for msg in messages:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                # Check if it was an intermediary tool call event
                if msg.message_type == "tool_call" and msg.metadata_json and "tool_calls" in msg.metadata_json:
                    for tc in msg.metadata_json["tool_calls"]:
                        func = tc.get("function", {})
                        lines.append(f"Assistant: [Requested Tool: {func.get('name')} with arguments {func.get('arguments')}]")
                else:
                    lines.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                lines.append(f"Tool ({msg.tool_name or 'unknown'}): [Result: {msg.content}]")
            else:
                role_label = msg.role.capitalize()
                lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(lines)
