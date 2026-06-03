from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from ai_service.models.messages import Message, ToolCall
from ai_service.tools.base import ToolDefinition

class LLMResponse:
    def __init__(
        self,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        stream_iterator: AsyncIterator[str] | None = None,
        provider_name: str = "unknown",
        model_name: str = "unknown",
        provider_fallback: bool = False,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self._stream_iterator = stream_iterator
        self.provider_name = provider_name
        self.model_name = model_name
        self.provider_fallback = provider_fallback
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    async def stream(self) -> AsyncIterator[str]:
        """Yields text chunks if streaming is enabled."""
        if self._stream_iterator:
            async for chunk in self._stream_iterator:
                yield chunk
        else:
            yield self.content

    def as_assistant_message(self) -> Message:
        """Converts response into a Message domain object."""
        # If there are tool calls, we return a message with empty content or containing serializations
        msg_type = "tool_call" if self.tool_calls else "text"
        return Message(
            role="assistant",
            content=self.content,
            message_type=msg_type,
            metadata_json={
                "tool_calls": [tc.to_openai_dict() for tc in self.tool_calls]
            } if self.tool_calls else None
        )


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = True,
    ) -> LLMResponse:
        """Executes a chat session with historical messages and authorized tools."""
        pass

    @abstractmethod
    async def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Generates a simple, single text completion (e.g. for summarization)."""
        pass

    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass
