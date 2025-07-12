"""Circuit breaker pattern implementation for AI service providers.

This module provides a circuit breaker implementation to handle failures
and prevent cascading failures in distributed systems.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar('T')


class CircuitState(Enum):
    """Possible states of the circuit breaker."""
    CLOSED = auto()  # Normal operation, requests allowed
    OPEN = auto()    # Circuit is open, requests fail fast
    HALF_OPEN = auto()  # Test requests allowed to check if service has recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    state_transitions: Dict[str, int] = field(default_factory=dict)
    
    def record_success(self) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.consecutive_failures = 0
    
    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
    
    def record_state_change(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Record a state transition."""
        transition = f"{from_state.name}_TO_{to_state.name}"
        self.state_transitions[transition] = self.state_transitions.get(transition, 0) + 1


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening the circuit
    recovery_timeout: float = 30.0  # Time in seconds before attempting recovery
    success_threshold: int = 3  # Number of successful requests before closing the circuit
    
    class Config:
        frozen = True  # Make config immutable after creation


class CircuitBreaker:
    """Circuit breaker implementation for handling failures gracefully."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize the circuit breaker.
        
        Args:
            config: Configuration for the circuit breaker.
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_state_change = time.time()
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function call.
            
        Raises:
            CircuitBreakerError: If the circuit is open.
            Exception: Any exception raised by the wrapped function.
        """
        # Check if we should allow the request
        if self.state == CircuitState.OPEN:
            # Check if we should attempt recovery
            if time.time() - self.last_state_change > self.config.recovery_timeout:
                self._set_state(CircuitState.HALF_OPEN)
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle a successful request."""
        self.stats.record_success()
        
        # If we're in half-open state and have enough successes, close the circuit
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_failures == 0 and \
               self.stats.total_requests >= self.config.success_threshold:
                self._set_state(CircuitState.CLOSED)
    
    def _on_failure(self) -> None:
        """Handle a failed request."""
        self.stats.record_failure()
        
        # Update circuit state based on failure count
        if self.state == CircuitState.HALF_OPEN:
            self._set_state(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                self._set_state(CircuitState.OPEN)
    
    def is_open(self) -> bool:
        """Check if the circuit breaker is in the open state.
        
        Returns:
            True if the circuit is open (requests should fail fast), False otherwise.
        """
        # If in HALF_OPEN state and recovery time has passed, try allowing requests
        if self.state == CircuitState.HALF_OPEN:
            return False
            
        # If in OPEN state, check if recovery timeout has elapsed
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_state_change
            if elapsed >= self.config.recovery_timeout:
                # Transition to HALF_OPEN to test if service recovered
                self._set_state(CircuitState.HALF_OPEN)
                return False
            return True
            
        # CLOSED state means circuit is healthy
        return False
    
    def _set_state(self, new_state: CircuitState) -> None:
        """Update the circuit breaker state."""
        if new_state == self.state:
            return
            
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        self.stats.record_state_change(old_state, new_state)
        
        # Reset stats when transitioning to half-open
        if new_state == CircuitState.HALF_OPEN:
            self.stats = CircuitBreakerStats()


class CircuitBreakerError(Exception):
    """Exception raised when the circuit breaker is open."""
    pass
