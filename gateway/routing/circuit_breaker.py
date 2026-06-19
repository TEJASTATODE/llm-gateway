import time
from enum import Enum
from dataclasses import dataclass, field


class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Tripped — requests blocked immediately
    HALF_OPEN = "half_open" # Testing — one request allowed through


@dataclass
class CircuitBreaker:
    """
    Tracks health of one provider.
    Each provider gets its own CircuitBreaker instance.
    
    Real world example:
    - OpenAI goes down at 3am
    - After 5 failures, circuit OPENS
    - For next 30 seconds, all OpenAI requests fail instantly (no waiting)
    - After 30 seconds, one test request goes through (HALF_OPEN)
    - If test succeeds → CLOSED again, traffic resumes
    - If test fails → back to OPEN for another 30 seconds
    """
    provider_name: str
    failure_threshold: int = 5      # How many failures before tripping
    recovery_timeout: float = 30.0  # Seconds to wait before testing recovery
    
    # Internal state — not set by caller
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _success_count: int = field(default=0, init=False)
    _total_requests: int = field(default=0, init=False)

    def is_closed(self) -> bool:
        """
        The gateway calls this before every request.
        Returns True if requests should go through.
        Returns False if provider should be skipped.
        
        This is the key method — it also handles the
        OPEN → HALF_OPEN transition automatically.
        """
        self._total_requests += 1

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if enough time has passed to try recovery
            time_since_failure = time.monotonic() - self._last_failure_time
            if time_since_failure >= self.recovery_timeout:
                # Transition to HALF_OPEN — allow one test request
                self._state = CircuitState.HALF_OPEN
                print(f"[CircuitBreaker] {self.provider_name}: OPEN → HALF_OPEN (testing recovery)")
                return True
            else:
                remaining = self.recovery_timeout - time_since_failure
                print(f"[CircuitBreaker] {self.provider_name}: OPEN — blocking request ({remaining:.1f}s remaining)")
                return False

        if self._state == CircuitState.HALF_OPEN:
            # Only allow one request through at a time
            # If this succeeds → record_success() closes the circuit
            # If this fails → record_failure() reopens it
            return True

        return False

    def record_success(self):
        """
        Called after a successful provider response.
        In HALF_OPEN state, one success closes the circuit.
        In CLOSED state, resets the failure count.
        """
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # Provider has recovered — resume normal traffic
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            print(f"[CircuitBreaker] {self.provider_name}: HALF_OPEN → CLOSED (recovered)")
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            # Prevents old failures from counting against future requests
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """
        Called after a failed provider response.
        Increments failure count. If threshold reached — trips the circuit.
        In HALF_OPEN — one failure sends back to OPEN.
        """
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Recovery test failed — back to OPEN
            self._state = CircuitState.OPEN
            print(f"[CircuitBreaker] {self.provider_name}: HALF_OPEN → OPEN (recovery failed)")

        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                # Too many failures — trip the circuit
                self._state = CircuitState.OPEN
                print(f"[CircuitBreaker] {self.provider_name}: CLOSED → OPEN ({self._failure_count} failures)")

    @property
    def state(self) -> str:
        return self._state.value

    @property 
    def failure_count(self) -> int:
        return self._failure_count

    def get_stats(self) -> dict:
        """For the dashboard — shows health of this provider"""
        return {
            "provider": self.provider_name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_requests": self._total_requests,
            "error_rate": round(
                self._failure_count / max(self._total_requests, 1), 3
            ),
        }