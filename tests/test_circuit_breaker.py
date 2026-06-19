"""
Circuit breaker test suite — validates state transitions
and recovery behaviour.
"""
import pytest
import time
from gateway.routing.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture
def cb():
    return CircuitBreaker(
        provider_name="test",
        failure_threshold=3,
        recovery_timeout=0.1,  # 100ms for fast tests
    )


class TestCircuitBreaker:

    def test_starts_closed(self, cb):
        assert cb.state == "closed"
        assert cb.is_closed()

    def test_success_keeps_closed(self, cb):
        cb.record_success()
        cb.record_success()
        assert cb.state == "closed"

    def test_failures_open_circuit(self, cb):
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert not cb.is_closed()

    def test_open_blocks_requests(self, cb):
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        # Should not allow requests through
        assert not cb.is_closed()

    def test_recovery_timeout_transitions_to_half_open(self, cb):
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next is_closed() call should transition to half_open
        result = cb.is_closed()
        assert cb.state == "half_open"
        assert result is True  # allows one test request through

    def test_half_open_success_closes_circuit(self, cb):
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.is_closed()  # transitions to half_open

        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens_circuit(self, cb):
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.is_closed()  # transitions to half_open

        cb.record_failure()
        assert cb.state == "open"

    def test_failure_count_resets_on_recovery(self, cb):
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.is_closed()
        cb.record_success()

        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_stats_accurate(self, cb):
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        stats = cb.get_stats()

        assert stats["provider"] == "test"
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 2
        assert "error_rate" in stats

    def test_below_threshold_stays_closed(self, cb):
        """Failures below threshold should not open circuit"""
        for _ in range(2):  # threshold is 3
            cb.record_failure()
        assert cb.state == "closed"