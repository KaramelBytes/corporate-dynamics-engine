"""Provider health monitoring system for AI service providers.

This module provides monitoring capabilities for AI service providers, tracking their
health status, response times, error rates, and implementing fallback strategies.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from .data_models import ProviderType


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status of an AI provider."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProviderMetrics(BaseModel):
    """Metrics for an AI provider."""
    
    provider_type: ProviderType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    avg_latency_ms: float = 0
    error_rate: float = 0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    consecutive_failures: int = 0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    
    def record_success(self, latency_ms: float) -> None:
        """Record a successful request.
        
        Args:
            latency_ms: Request latency in milliseconds.
        """
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.successful_requests
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        self.last_success_time = datetime.now()
        self.consecutive_failures = 0
        self._update_health_status()
    
    def record_failure(self, error_message: str) -> None:
        """Record a failed request.
        
        Args:
            error_message: Error message from the failed request.
        """
        self.total_requests += 1
        self.failed_requests += 1
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        self.last_failure_time = datetime.now()
        self.last_error_message = error_message
        self.consecutive_failures += 1
        self._update_health_status()
    
    def _update_health_status(self) -> None:
        """Update the health status based on metrics."""
        if self.total_requests < 5:
            # Not enough data to determine health
            self.health_status = HealthStatus.UNKNOWN
            return
        
        if self.consecutive_failures >= 3:
            self.health_status = HealthStatus.UNHEALTHY
        elif self.error_rate > 0.2:  # More than 20% error rate
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.HEALTHY


class ProviderHealthMonitor:
    """Monitors the health of AI service providers.
    
    Tracks provider performance metrics, detects degraded service, and
    implements fallback strategies to ensure reliable AI service delivery.
    """
    
    def __init__(
        self,
        providers: List[ProviderType],
        degraded_threshold: float = 0.2,
        unhealthy_threshold: float = 0.5,
        recovery_threshold: int = 3,
        metrics_window_minutes: int = 60,
    ):
        """Initialize the provider health monitor.
        
        Args:
            providers: List of provider types to monitor.
            degraded_threshold: Error rate threshold for degraded status.
            unhealthy_threshold: Error rate threshold for unhealthy status.
            recovery_threshold: Number of consecutive successes needed for recovery.
            metrics_window_minutes: Time window for metrics collection in minutes.
        """
        self.metrics: Dict[ProviderType, ProviderMetrics] = {
            provider: ProviderMetrics(
                provider_type=provider,
                health_status=HealthStatus.HEALTHY  # Initialize as HEALTHY by default
            )
            for provider in providers
        }
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_threshold = recovery_threshold
        self.metrics_window_minutes = metrics_window_minutes
        
        # Request history for time-windowed metrics
        self._request_history: Dict[ProviderType, List[Tuple[datetime, bool, float]]] = {
            provider: [] for provider in providers
        }
        
        logger.info(f"Provider health monitor initialized for {len(providers)} providers")
    
    def record_request(
        self,
        provider: ProviderType,
        success: bool,
        latency_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a request to a provider.
        
        Args:
            provider: The provider that handled the request.
            success: Whether the request was successful.
            latency_ms: Request latency in milliseconds.
            error_message: Error message if the request failed.
        """
        if provider not in self.metrics:
            self.metrics[provider] = ProviderMetrics(provider_type=provider)
        
        # Record in metrics
        if success:
            self.metrics[provider].record_success(latency_ms)
        else:
            self.metrics[provider].record_failure(error_message or "Unknown error")
        
        # Add to history for time-windowed metrics
        self._request_history[provider].append((datetime.now(), success, latency_ms))
        
        # Clean up old history entries
        self._clean_history()
    
    def _clean_history(self) -> None:
        """Clean up request history older than the metrics window."""
        cutoff_time = datetime.now() - timedelta(minutes=self.metrics_window_minutes)
        
        for provider in self._request_history:
            self._request_history[provider] = [
                entry for entry in self._request_history[provider]
                if entry[0] >= cutoff_time
            ]
    
    def get_provider_health(self, provider: ProviderType) -> HealthStatus:
        """Get the health status of a provider.
        
        Args:
            provider: The provider to check.
            
        Returns:
            HealthStatus: The current health status of the provider.
        """
        if provider not in self.metrics:
            return HealthStatus.UNKNOWN
        
        return self.metrics[provider].health_status
    
    def get_available_providers(self) -> List[ProviderType]:
        """Get a list of available (healthy or degraded) providers.
        
        Returns:
            List[ProviderType]: List of available providers.
        """
        return [
            provider for provider, metrics in self.metrics.items()
            if metrics.health_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        ]
    
    def get_optimal_provider(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        """Get the optimal provider from a list of candidates.
        
        Args:
            providers: List of candidate providers.
            
        Returns:
            Optional[ProviderType]: The optimal provider, or None if none are available.
        """
        available = [
            p for p in providers
            if p in self.metrics and self.metrics[p].health_status != HealthStatus.UNHEALTHY
        ]
        
        if not available:
            return None
        
        # Sort by health status (healthy > degraded > unknown) and then by error rate
        sorted_providers = sorted(
            available,
            key=lambda p: (
                0 if self.metrics[p].health_status == HealthStatus.HEALTHY else
                1 if self.metrics[p].health_status == HealthStatus.DEGRADED else 2,
                self.metrics[p].error_rate
            )
        )
        
        return sorted_providers[0] if sorted_providers else None
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """Get a summary of provider metrics.
        
        Returns:
            Dict[str, Dict[str, Union[str, float, int]]]: Summary of metrics by provider.
        """
        return {
            provider.value: {
                "status": metrics.health_status.value,
                "error_rate": round(metrics.error_rate * 100, 2),
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "total_requests": metrics.total_requests,
                "success_rate": round(
                    (metrics.successful_requests / metrics.total_requests * 100)
                    if metrics.total_requests > 0 else 0,
                    2
                ),
            }
            for provider, metrics in self.metrics.items()
        }
    
    def should_retry_with_fallback(
        self, 
        provider: ProviderType, 
        error_message: str
    ) -> Tuple[bool, Optional[ProviderType]]:
        """Determine if a request should be retried with a fallback provider.
        
        Args:
            provider: The provider that failed.
            error_message: The error message from the failed request.
            
        Returns:
            Tuple[bool, Optional[ProviderType]]: Whether to retry and with which provider.
        """
        # Record the failure
        self.record_request(
            provider=provider,
            success=False,
            latency_ms=0,
            error_message=error_message
        )
        
        # Check if we should retry
        if provider not in self.metrics:
            return False, None
        
        # If the provider is unhealthy, try to find a fallback
        if self.metrics[provider].health_status == HealthStatus.UNHEALTHY:
            # Get all providers except the failed one
            candidates = [p for p in self.metrics.keys() if p != provider]
            fallback = self.get_optimal_provider(candidates)
            return True, fallback
        
        # If this is a consecutive failure, consider a fallback
        if self.metrics[provider].consecutive_failures >= 2:
            candidates = [p for p in self.metrics.keys() if p != provider]
            fallback = self.get_optimal_provider(candidates)
            return True, fallback
        
        # Otherwise, retry with the same provider
        return True, provider
