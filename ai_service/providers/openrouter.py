import json
import logging
import httpx
from typing import Any, AsyncIterator
from ai_service.config.settings import settings
from ai_service.models.messages import Message, ToolCall
from ai_service.tools.base import ToolDefinition
from ai_service.providers.base import LLMProvider, LLMResponse
from ai_service.errors import ProviderUnavailableError, ProviderTimeoutError, ProviderRateLimitError

logger = logging.getLogger("ai_service.providers.openrouter")

class OpenRouterProvider(LLMProvider):
    MODEL = "z-ai/glm-4.5-air:free"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
            "X-Title": "University AI Assistant",
        }

    @property
    def supports_tool_calling(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = True
    ) -> LLMResponse:
        """
        Sends historical messages and authorized tools to OpenRouter chat completion.
        If the model decides to call a tool, the streaming chunks are accumulated internally
        to assemble the complete tool call. Otherwise, text chunks are streamed dynamically.
        """
        payload = {
            "model": self.MODEL,
            "messages": [m.to_openai_dict() for m in messages],
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = [t.to_llm_schema() for t in tools]
            payload["tool_choice"] = "auto"

        client = httpx.AsyncClient(timeout=30.0)
        
        try:
            if not stream:
                # Static chat completion
                resp = await client.post(f"{self.BASE_URL}/chat/completions", headers=self.headers, json=payload)
                self._check_status_errors(resp)
                data = resp.json()
                await client.aclose()
                return self._parse_static_response(data)

            # Streaming chat completion
            # We open the request in a stream context
            request = client.build_request("POST", f"{self.BASE_URL}/chat/completions", headers=self.headers, json=payload)
            resp = await client.send(request, stream=True)
            self._check_status_errors(resp)

            # We need to determine if the stream starts with a tool call or text.
            # We will read the first chunk to inspect choice.delta.
            return await self._process_stream(resp, client)

        except httpx.TimeoutException as e:
            await client.aclose()
            logger.error(f"OpenRouter timeout: {e}")
            raise ProviderTimeoutError(f"OpenRouter request timed out: {e}")
        except httpx.HTTPStatusError as e:
            await client.aclose()
            logger.error(f"OpenRouter HTTP error {e.response.status_code}: {e.response.text}")
            raise ProviderUnavailableError(f"OpenRouter HTTP error: {e}")
        except Exception as e:
            await client.aclose()
            logger.error(f"OpenRouter error: {e}")
            raise ProviderUnavailableError(f"Failed to call OpenRouter: {e}")

    async def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Simple completion for utility runs (like summarization)."""
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                resp = await client.post(f"{self.BASE_URL}/chat/completions", headers=self.headers, json=payload)
                self._check_status_errors(resp)
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except (ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError) as e:
                raise e
            except httpx.TimeoutException as e:
                logger.error(f"OpenRouter complete timeout: {e}")
                raise ProviderTimeoutError(f"OpenRouter complete timed out: {e}")
            except Exception as e:
                logger.error(f"OpenRouter complete error: {e}")
                raise ProviderUnavailableError(f"OpenRouter complete failed: {e}")

    def _check_status_errors(self, response: httpx.Response):
        """Raises standard Provider errors based on HTTP status codes."""
        if response.status_code == 429:
            raise ProviderRateLimitError("OpenRouter API rate limit reached")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"OpenRouter server error (status: {response.status_code})")
        response.raise_for_status()

    def _parse_static_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parses a non-streaming JSON response from OpenRouter."""
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        
        tool_calls = []
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            for tc in raw_tool_calls:
                func = tc["function"]
                try:
                    args = json.loads(func["arguments"])
                except Exception:
                    args = func["arguments"]
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=func["name"],
                    arguments=args if isinstance(args, dict) else {}
                ))
                
        return LLMResponse(content=content, tool_calls=tool_calls)

    async def _process_stream(self, response: httpx.Response, client: httpx.AsyncClient) -> LLMResponse:
        """
        Parses streaming chunks. 
        If the stream represents a tool call, we consume the whole stream,
        parse the arguments, and return them in the LLMResponse.
        If it represents text content, we return an LLMResponse holding
        an async iterator yielding the text chunks dynamically.
        """
        # We need to buffer the start of the stream to see if it contains a tool call.
        lines_iterator = response.aiter_lines()
        
        first_content_chunk = ""
        tool_calls_accumulator = {} # index -> {id, name, arguments_str}
        is_tool_call = False
        
        # Read the first few lines until we find a choice delta
        async for line in lines_iterator:
            if not line.strip() or line.strip() == "data: [DONE]":
                continue
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    
                    # Check for tool call initialization
                    if "tool_calls" in delta:
                        is_tool_call = True
                        self._accumulate_tool_calls(delta["tool_calls"], tool_calls_accumulator)
                    
                    # Check for content
                    if "content" in delta and delta["content"]:
                        first_content_chunk = delta["content"]
                        break
                        
                    if is_tool_call:
                        # Continue reading tool call chunks
                        pass
                except Exception as e:
                    logger.debug(f"Error parsing line in stream initialization: {e}")
                    
            if is_tool_call:
                # If it's a tool call, we must continue consuming lines to assemble arguments
                continue

        if is_tool_call:
            # We consume the REST of the stream to get the full tool call details
            async for line in lines_iterator:
                if not line.strip() or line.strip() == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "tool_calls" in delta:
                                self._accumulate_tool_calls(delta["tool_calls"], tool_calls_accumulator)
                    except Exception:
                        pass
            
            # Close connection resources
            await response.aclose()
            await client.aclose()
            
            # Format accumulated tool calls into ToolCall list
            final_tool_calls = []
            for index, tc_data in sorted(tool_calls_accumulator.items()):
                try:
                    args = json.loads(tc_data["arguments"])
                except Exception:
                    args = {}
                final_tool_calls.append(ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    arguments=args if isinstance(args, dict) else {}
                ))
            return LLMResponse(content="", tool_calls=final_tool_calls)

        # If it's not a tool call, it's a text stream! We yield the first content chunk
        # and return an async generator that yields subsequent chunks
        async def text_generator() -> AsyncIterator[str]:
            try:
                if first_content_chunk:
                    yield first_content_chunk
                    
                async for line in lines_iterator:
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                        except Exception:
                            pass
            finally:
                await response.aclose()
                await client.aclose()

        return LLMResponse(content="", tool_calls=[], stream_iterator=text_generator())

    def _accumulate_tool_calls(self, tool_calls_delta: list[dict], accumulator: dict):
        """Helper to accumulate streaming tool call chunks."""
        for tc in tool_calls_delta:
            idx = tc.get("index", 0)
            if idx not in accumulator:
                accumulator[idx] = {"id": "", "name": "", "arguments": ""}
                
            if "id" in tc:
                accumulator[idx]["id"] += tc["id"]
                
            func = tc.get("function", {})
            if "name" in func:
                accumulator[idx]["name"] += func["name"]
            if "arguments" in func:
                accumulator[idx]["arguments"] += func["arguments"]
