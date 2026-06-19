from gateway.routing.circuit_breaker import CircuitBreaker
from gateway.config import settings


class CircuitBreakerRegistry:
    """
    One CircuitBreaker per provider.
    Gateway asks this registry which providers are healthy
    and in what order to try them.
    
    Fallback chain logic lives here — not scattered through the codebase.
    """

    # Default order to try providers
    # In production this could be config-driven per customer
    DEFAULT_ORDER = ["gemini", "openai", "anthropic"]

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(
                provider_name=name,
                failure_threshold=settings.cb_failure_threshold,
                recovery_timeout=settings.cb_recovery_timeout,
            )
            for name in self.DEFAULT_ORDER
        }

    def get(self, provider_name: str) -> CircuitBreaker:
        """Get circuit breaker for a specific provider"""
        return self._breakers[provider_name]

    def get_healthy_providers(self, preferred: str = None) -> list[str]:
        """
        Returns providers in priority order, healthy ones first.
        
        If preferred provider is specified and healthy — it goes first.
        Then remaining providers in default order.
        
        Example:
        - preferred=openai, openai is OPEN (failing)
        - Returns: [gemini, anthropic]  ← openai skipped entirely
        
        Example:
        - preferred=openai, openai is CLOSED (healthy)  
        - Returns: [openai, gemini, anthropic]
        """
        order = []

        # Preferred provider goes first if specified
        if preferred and preferred in self._breakers:
            order.append(preferred)

        # Add remaining providers in default order
        for name in self.DEFAULT_ORDER:
            if name not in order:
                order.append(name)

        return order

    def all_stats(self) -> list[dict]:
        """Returns health stats for all providers — used by dashboard"""
        return [cb.get_stats() for cb in self._breakers.values()]