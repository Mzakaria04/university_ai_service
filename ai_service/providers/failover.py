import logging
from typing import Any
from opentelemetry import trace

from ai_service.models.messages import Message
from ai_service.tools.base import ToolDefinition
from ai_service.providers.base import LLMProvider, LLMResponse
from ai_service.providers.openrouter import OpenRouterProvider
from ai_service.providers.groq import GroqProvider
from ai_service.providers.circuit_breaker import CircuitBreaker, CircuitState
from ai_service.errors import ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError
from ai_service.observability.metrics import ai_provider_failover_total

logger = logging.getLogger("ai_service.providers.failover")

class FailoverProviderOrchestrator(LLMProvider):
    _shared_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    def __init__(
        self,
        primary: LLMProvider | None = None,
        fallback: LLMProvider | None = None,
        circuit_breaker: CircuitBreaker | None = None
    ):
        self.primary = primary or OpenRouterProvider()
        self.fallback = fallback or GroqProvider()
        self.circuit_breaker = circuit_breaker or self._shared_circuit_breaker

    @property
    def supports_tool_calling(self) -> bool:
        # Both must support it
        return self.primary.supports_tool_calling and self.fallback.supports_tool_calling

    @property
    def supports_streaming(self) -> bool:
        return self.primary.supports_streaming and self.fallback.supports_streaming

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = True
    ) -> LLMResponse:
        """
        Routes chat completion request to the primary provider if healthy,
        otherwise falls back to the secondary provider if the primary is failing or rate-limited.
        """
        current_span = trace.get_current_span()
        
        # Check if primary is allowed to execute
        if not self.circuit_breaker.can_execute():
            logger.warning("Primary provider circuit breaker is OPEN. Fast-failing over to fallback provider.")
            ai_provider_failover_total.labels(primary_provider="openrouter", fallback_provider="groq").inc()
            if current_span:
                current_span.set_attribute("provider.fallback", True)
                current_span.set_attribute("provider.name", "groq")
                current_span.set_attribute("provider.model", getattr(self.fallback, "MODEL", "unknown"))
            
            # Execute directly with fallback
            response = await self.fallback.chat(messages=messages, tools=tools, stream=stream)
            response.provider_fallback = True
            return response

        try:
            # Try primary
            response = await self.primary.chat(messages=messages, tools=tools, stream=stream)
            self.circuit_breaker.record_success()
            return response
        except (ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError, Exception) as e:
            # Record failure in circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.warning(
                f"Primary provider failed with error: {str(e)}. Failing over to fallback provider.",
                exc_info=True
            )
            
            # Record metrics & span attributes
            ai_provider_failover_total.labels(primary_provider="openrouter", fallback_provider="groq").inc()
            if current_span:
                current_span.set_attribute("provider.fallback", True)
                current_span.set_attribute("provider.name", "groq")
                current_span.set_attribute("provider.model", getattr(self.fallback, "MODEL", "unknown"))
                current_span.record_exception(e)
            
            # Run fallback
            response = await self.fallback.chat(messages=messages, tools=tools, stream=stream)
            response.provider_fallback = True
            return response

    async def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Routes utility completion request with failover capability."""
        current_span = trace.get_current_span()
        
        if not self.circuit_breaker.can_execute():
            logger.warning("Primary provider circuit breaker is OPEN. Fast-failing over to fallback provider.")
            ai_provider_failover_total.labels(primary_provider="openrouter", fallback_provider="groq").inc()
            if current_span:
                current_span.set_attribute("provider.fallback", True)
                current_span.set_attribute("provider.name", "groq")
                current_span.set_attribute("provider.model", getattr(self.fallback, "MODEL", "unknown"))
            
            return await self.fallback.complete(prompt, max_tokens)

        try:
            result = await self.primary.complete(prompt, max_tokens)
            self.circuit_breaker.record_success()
            return result
        except (ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError, Exception) as e:
            self.circuit_breaker.record_failure()
            logger.warning(
                f"Primary provider failed in complete with error: {str(e)}. Failing over to fallback provider.",
                exc_info=True
            )
            
            ai_provider_failover_total.labels(primary_provider="openrouter", fallback_provider="groq").inc()
            if current_span:
                current_span.set_attribute("provider.fallback", True)
                current_span.set_attribute("provider.name", "groq")
                current_span.set_attribute("provider.model", getattr(self.fallback, "MODEL", "unknown"))
                current_span.record_exception(e)
                
            return await self.fallback.complete(prompt, max_tokens)
