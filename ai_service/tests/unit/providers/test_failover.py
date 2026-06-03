import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from ai_service.providers.failover import FailoverProviderOrchestrator
from ai_service.providers.circuit_breaker import CircuitBreaker, CircuitState
from ai_service.providers.base import LLMProvider, LLMResponse
from ai_service.errors import ProviderRateLimitError, ProviderUnavailableError
from ai_service.models.messages import Message
from ai_service.observability.metrics import ai_provider_failover_total

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

class MockLLMProvider(LLMProvider):
    def __init__(self, name: str, should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.calls = 0

    async def chat(self, messages, tools=None, stream=True):
        self.calls += 1
        if self.should_fail:
            raise ProviderUnavailableError(f"{self.name} failed")
        return LLMResponse(content=f"Response from {self.name}")

    async def complete(self, prompt, max_tokens=512):
        self.calls += 1
        if self.should_fail:
            raise ProviderUnavailableError(f"{self.name} failed")
        return f"Response from {self.name}"

    @property
    def supports_tool_calling(self):
        return True

    @property
    def supports_streaming(self):
        return True

async def test_failover_success_primary():
    """Verify that primary is used and fallback is untouched when primary is successful."""
    primary = MockLLMProvider("primary")
    fallback = MockLLMProvider("fallback")
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    orchestrator = FailoverProviderOrchestrator(primary, fallback, cb)

    res = await orchestrator.complete("hello")
    assert res == "Response from primary"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert cb.state == CircuitState.CLOSED
    assert cb.failures == 0

async def test_failover_to_fallback():
    """Verify that when primary fails, call falls back to secondary and metrics are incremented."""
    primary = MockLLMProvider("primary", should_fail=True)
    fallback = MockLLMProvider("fallback")
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    orchestrator = FailoverProviderOrchestrator(primary, fallback, cb)

    try:
        initial_count = ai_provider_failover_total.labels(primary_provider="openrouter", fallback_provider="groq")._value.get()
    except Exception:
        initial_count = 0

    res = await orchestrator.complete("hello")
    assert res == "Response from fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert cb.state == CircuitState.CLOSED
    assert cb.failures == 1

    final_count = ai_provider_failover_total.labels(primary_provider="openrouter", fallback_provider="groq")._value.get()
    assert final_count == initial_count + 1

async def test_circuit_breaker_opens_after_threshold():
    """Verify that 3 failures transition circuit to OPEN and fast-fail subsequent calls."""
    primary = MockLLMProvider("primary", should_fail=True)
    fallback = MockLLMProvider("fallback")
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    orchestrator = FailoverProviderOrchestrator(primary, fallback, cb)

    # 1st failure
    await orchestrator.complete("hello")
    assert cb.state == CircuitState.CLOSED
    
    # 2nd failure
    await orchestrator.complete("hello")
    assert cb.state == CircuitState.CLOSED
    
    # 3rd failure
    await orchestrator.complete("hello")
    assert cb.state == CircuitState.OPEN

    # 4th call: primary shouldn't even be called!
    primary.calls = 0
    fallback.calls = 0
    res = await orchestrator.complete("hello")
    assert res == "Response from fallback"
    assert primary.calls == 0
    assert fallback.calls == 1
    assert cb.state == CircuitState.OPEN

async def test_circuit_breaker_recovery_half_open():
    """Verify that after recovery timeout, the circuit enters HALF_OPEN and retries primary."""
    primary = MockLLMProvider("primary", should_fail=True)
    fallback = MockLLMProvider("fallback")
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1) # 1 failure threshold, 0.1s timeout
    orchestrator = FailoverProviderOrchestrator(primary, fallback, cb)

    # Trigger failure to open circuit
    await orchestrator.complete("hello")
    assert cb.state == CircuitState.OPEN

    # Sleep past recovery timeout
    time.sleep(0.15)

    # Next call should check primary (HALF_OPEN)
    # Let's make primary healthy now
    primary.should_fail = False
    primary.calls = 0
    fallback.calls = 0

    res = await orchestrator.complete("hello")
    assert res == "Response from primary"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert cb.state == CircuitState.CLOSED
