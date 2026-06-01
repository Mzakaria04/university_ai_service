import json
import logging
import httpx
from typing import Any, AsyncIterator
from ai_service.config.settings import settings
from ai_service.models.messages import Message, ToolCall
from ai_service.tools.base import ToolDefinition
from ai_service.providers.base import LLMProvider, LLMResponse
from ai_service.errors import ProviderUnavailableError, ProviderTimeoutError, ProviderRateLimitError

logger = logging.getLogger("ai_service.providers.groq")

class GroqProvider(LLMProvider):
    MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
        Sends historical messages and authorized tools to Groq API.
        If the model decides to call a tool, the streaming chunks are accumulated internally
        to assemble the complete tool call. Otherwise, text chunks are streamed dynamically.
        """
        from ai_service.observability.tracing import get_tracer
        from opentelemetry import trace
        tracer = get_tracer()

        with tracer.start_as_current_span("llm_provider_chat") as span:
            span.set_attribute("provider.name", "groq")
            span.set_attribute("provider.model", self.MODEL)
            span.set_attribute("provider.stream", stream)

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
                from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception
                
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
                    retry=retry_if_exception(lambda e: isinstance(e, (httpx.TimeoutException, httpx.NetworkError, ProviderUnavailableError)) or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)),
                    reraise=True
                ):
                    with attempt:
                        if not stream:
                            # Static completion
                            resp = await client.post(f"{self.BASE_URL}/chat/completions", headers=self.headers, json=payload)
                            self._check_status_errors(resp)
                        else:
                            # Streaming completion
                            request = client.build_request("POST", f"{self.BASE_URL}/chat/completions", headers=self.headers, json=payload)
                            resp = await client.send(request, stream=True)
                            self._check_status_errors(resp)

                if not stream:
                    data = resp.json()
                    await client.aclose()
                    span.set_status(trace.StatusCode.OK)
                    result = self._parse_static_response(data)
                    usage = data.get("usage") or {}
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
                    result.total_tokens = usage.get("total_tokens")
                else:
                    result = await self._process_stream(resp, client)
                    span.set_status(trace.StatusCode.OK)

                result.provider_name = "groq"
                result.model_name = self.MODEL
                result.provider_fallback = False
                return result

            except httpx.TimeoutException as e:
                await client.aclose()
                logger.error(f"Groq timeout: {e}")
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise ProviderTimeoutError(f"Groq request timed out: {e}")
            except httpx.HTTPStatusError as e:
                await client.aclose()
                logger.error(f"Groq HTTP error {e.response.status_code}: {e.response.text}")
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise ProviderUnavailableError(f"Groq HTTP error: {e}")
            except Exception as e:
                await client.aclose()
                if not isinstance(e, (ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError)):
                    logger.error(f"Groq error: {e}")
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                if isinstance(e, (ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError)):
                    raise e
                raise ProviderUnavailableError(f"Failed to call Groq: {e}")

    async def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Simple completion for utility runs."""
        from ai_service.observability.tracing import get_tracer
        from opentelemetry import trace
        tracer = get_tracer()

        with tracer.start_as_current_span("llm_provider_complete") as span:
            span.set_attribute("provider.name", "groq")
            span.set_attribute("provider.model", self.MODEL)

            payload = {
                "model": self.MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False
            }
            async with httpx.AsyncClient(timeout=20.0) as client:
                try:
                    from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception
                    
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(3),
                        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
                        retry=retry_if_exception(lambda e: isinstance(e, (httpx.TimeoutException, httpx.NetworkError, ProviderUnavailableError)) or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)),
                        reraise=True
                    ):
                        with attempt:
                            resp = await client.post(f"{self.BASE_URL}/chat/completions", headers=self.headers, json=payload)
                            self._check_status_errors(resp)
                            
                    data = resp.json()
                    span.set_status(trace.StatusCode.OK)
                    return data["choices"][0]["message"]["content"] or ""
                except (ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError) as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise e
                except httpx.TimeoutException as e:
                    logger.error(f"Groq complete timeout: {e}")
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise ProviderTimeoutError(f"Groq complete timed out: {e}")
                except Exception as e:
                    logger.error(f"Groq complete error: {e}")
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise ProviderUnavailableError(f"Groq complete failed: {e}")

    def _check_status_errors(self, response: httpx.Response):
        """Raises standard Provider errors based on HTTP status codes."""
        if response.status_code == 429:
            raise ProviderRateLimitError("Groq API rate limit reached")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"Groq server error (status: {response.status_code})")
        response.raise_for_status()

    def _parse_static_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parses a non-streaming JSON response from Groq."""
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
        """Parses streaming chunks."""
        lines_iterator = response.aiter_lines()
        
        first_content_chunk = ""
        tool_calls_accumulator = {}
        is_tool_call = False
        
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
                    
                    if "tool_calls" in delta:
                        is_tool_call = True
                        self._accumulate_tool_calls(delta["tool_calls"], tool_calls_accumulator)
                    
                    if "content" in delta and delta["content"]:
                        first_content_chunk = delta["content"]
                        break
                        
                    if is_tool_call:
                        pass
                except Exception as e:
                    logger.debug(f"Error parsing line in stream initialization: {e}")
                    
            if is_tool_call:
                continue

        if is_tool_call:
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
            
            await response.aclose()
            await client.aclose()
            
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
