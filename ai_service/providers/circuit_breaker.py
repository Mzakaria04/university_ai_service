import time
import logging
from enum import Enum

logger = logging.getLogger("ai_service.providers.circuit_breaker")

class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0

    def can_execute(self) -> bool:
        """Determines if the circuit allows execution."""
        current_time = time.monotonic()
        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker recovery timeout reached. Transitioning to HALF_OPEN.")
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        """Records a successful operation and resets state if half-open."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker request succeeded. Transitioning to CLOSED.")
            self.state = CircuitState.CLOSED
            self.failures = 0
        elif self.state == CircuitState.CLOSED:
            self.failures = 0

    def record_failure(self):
        """Records a failed operation and opens the circuit if threshold exceeded."""
        self.failures += 1
        self.last_failure_time = time.monotonic()
        
        if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            if self.failures >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker threshold of {self.failure_threshold} failures reached. "
                    "Transitioning to OPEN."
                )
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker request failed in HALF_OPEN. Transitioning back to OPEN.")
                self.state = CircuitState.OPEN
