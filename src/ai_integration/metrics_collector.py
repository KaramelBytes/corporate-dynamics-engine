"""Metrics collection and request tracing for AI service calls.

This module provides comprehensive metrics collection and request tracing
capabilities for AI service calls, enabling performance monitoring, cost
tracking, and debugging of AI-driven features.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from .data_models import AIRequest, AIResponse, ContentType, ProviderType


logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    """Status of an AI request."""
    
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CACHED = "cached"


class RequestTrace(BaseModel):
    """Trace information for an AI request."""
    
    request_id: str
    provider: ProviderType
    content_type: ContentType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: RequestStatus = RequestStatus.SUCCESS
    error_message: Optional[str] = None
    token_count: int = 0
    cost: float = 0.0
    cache_hit: bool = False
    semantic_cache_hit: bool = False
    semantic_similarity: Optional[float] = None
    optimization_applied: bool = False
    token_savings: Optional[int] = None
    context_compression_ratio: Optional[float] = None
    request_hash: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    quality_score: Optional[float] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    
    def complete(
        self,
        status: RequestStatus,
        end_time: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Complete the request trace.
        
        Args:
            status: Final status of the request.
            end_time: End time of the request (defaults to now).
            error_message: Error message if the request failed.
        """
        self.end_time = end_time or datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        self.error_message = error_message


class MetricsCollector:
    """Collects and aggregates metrics for AI service calls.
    
    Provides comprehensive metrics collection, request tracing, and
    performance monitoring for AI service calls.
    """
    
    def __init__(
        self,
        metrics_dir: Optional[Union[str, Path]] = None,
        enable_file_logging: bool = False,
        retention_days: int = 30,
    ):
        """Initialize the metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics data.
            enable_file_logging: Whether to log metrics to files.
            retention_days: Number of days to retain metrics data.
        """
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        self.enable_file_logging = enable_file_logging
        self.retention_days = retention_days
        
        # In-memory storage for recent requests
        self.request_traces: Dict[str, RequestTrace] = {}
        
        # Aggregated metrics
        self.provider_metrics: Dict[ProviderType, Dict[str, Any]] = {}
        self.content_type_metrics: Dict[ContentType, Dict[str, Any]] = {}
        self.hourly_metrics: Dict[str, Dict[str, Any]] = {}
        self.daily_metrics: Dict[str, Dict[str, Any]] = {}
        
        if self.enable_file_logging and self.metrics_dir:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Metrics collector initialized")
    
    def start_request_trace(
        self,
        request: AIRequest,
        provider: ProviderType,
        request_hash: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start tracing an AI request.
        
        Args:
            request: The AI request being traced.
            provider: The provider handling the request.
            request_hash: Hash of the request for caching.
            tags: Additional tags for the request.
            
        Returns:
            str: The request ID for the trace.
        """
        request_id = str(uuid.uuid4())
        
        trace = RequestTrace(
            request_id=request_id,
            provider=provider,
            content_type=request.content_type,
            start_time=datetime.now(),
            token_count=request.estimated_tokens if request.estimated_tokens is not None else 0,
            request_hash=request_hash,
            tags=tags or {},
        )
        
        self.request_traces[request_id] = trace
        return request_id
    
    def complete_request_trace(
        self,
        request_id: str,
        status: RequestStatus,
        response: Optional[AIResponse] = None,
        error_message: Optional[str] = None,
        cost: Optional[float] = None,
        cache_hit: bool = False,
        semantic_cache_hit: bool = False,
        semantic_similarity: Optional[float] = None,
        token_savings: Optional[int] = None,
        context_compression_ratio: Optional[float] = None,
        quality_score: Optional[float] = None,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """Complete a request trace.
        
        Args:
            request_id: The ID of the request to complete.
            status: The final status of the request.
            response: The AI response if successful.
            error_message: Error message if the request failed.
            cost: The cost of the request.
            cache_hit: Whether the request was served from cache.
            semantic_cache_hit: Whether the request was served from semantic cache.
            semantic_similarity: Similarity score for semantic cache hit.
            token_savings: Number of tokens saved by optimization.
            context_compression_ratio: Ratio of compressed context size to original.
            quality_score: Quality score of the response.
            fallback_used: Whether a fallback provider was used.
            fallback_reason: Reason for using fallback provider.
        """
        if request_id not in self.request_traces:
            logger.warning(f"Request trace not found: {request_id}")
            return
        
        trace = self.request_traces[request_id]
        trace.complete(status, error_message=error_message)
        
        # Update additional metrics
        if cost is not None:
            trace.cost = cost
        
        trace.cache_hit = cache_hit
        trace.semantic_cache_hit = semantic_cache_hit
        trace.semantic_similarity = semantic_similarity
        trace.token_savings = token_savings
        trace.context_compression_ratio = context_compression_ratio
        trace.quality_score = quality_score
        trace.fallback_used = fallback_used
        trace.fallback_reason = fallback_reason
        
        # Update aggregated metrics
        self._update_aggregated_metrics(trace)
        
        # Log to file if enabled
        if self.enable_file_logging and self.metrics_dir:
            self._log_trace_to_file(trace)
    
    def _update_aggregated_metrics(self, trace: RequestTrace) -> None:
        """Update aggregated metrics with a completed trace.
        
        Args:
            trace: The completed request trace.
        """
        # Initialize provider metrics if needed
        if trace.provider not in self.provider_metrics:
            self.provider_metrics[trace.provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_duration_ms": 0.0,
                "cache_hits": 0,
                "semantic_cache_hits": 0,
                "fallbacks_used": 0,
            }
        
        # Initialize content type metrics if needed
        if trace.content_type not in self.content_type_metrics:
            self.content_type_metrics[trace.content_type] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_duration_ms": 0.0,
                "cache_hits": 0,
                "semantic_cache_hits": 0,
            }
        
        # Update provider metrics
        provider_metrics = self.provider_metrics[trace.provider]
        provider_metrics["total_requests"] += 1
        
        if trace.status == RequestStatus.SUCCESS:
            provider_metrics["successful_requests"] += 1
        elif trace.status in (RequestStatus.FAILURE, RequestStatus.TIMEOUT):
            provider_metrics["failed_requests"] += 1
        
        provider_metrics["total_cost"] += trace.cost
        provider_metrics["total_tokens"] += trace.token_count
        
        if trace.duration_ms:
            provider_metrics["total_duration_ms"] += trace.duration_ms
        
        if trace.cache_hit:
            provider_metrics["cache_hits"] += 1
        
        if trace.semantic_cache_hit:
            provider_metrics["semantic_cache_hits"] += 1
        
        if trace.fallback_used:
            provider_metrics["fallbacks_used"] += 1
        
        # Update content type metrics
        content_metrics = self.content_type_metrics[trace.content_type]
        content_metrics["total_requests"] += 1
        
        if trace.status == RequestStatus.SUCCESS:
            content_metrics["successful_requests"] += 1
        elif trace.status in (RequestStatus.FAILURE, RequestStatus.TIMEOUT):
            content_metrics["failed_requests"] += 1
        
        content_metrics["total_cost"] += trace.cost
        content_metrics["total_tokens"] += trace.token_count
        
        if trace.duration_ms:
            content_metrics["total_duration_ms"] += trace.duration_ms
        
        if trace.cache_hit:
            content_metrics["cache_hits"] += 1
        
        if trace.semantic_cache_hit:
            content_metrics["semantic_cache_hits"] += 1
        
        # Update time-based metrics
        self._update_time_metrics(trace)
    
    def _update_time_metrics(self, trace: RequestTrace) -> None:
        """Update time-based metrics.
        
        Args:
            trace: The completed request trace.
        """
        # Hourly metrics
        hour_key = trace.start_time.strftime("%Y-%m-%d-%H")
        
        if hour_key not in self.hourly_metrics:
            self.hourly_metrics[hour_key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "cache_hits": 0,
                "semantic_cache_hits": 0,
                "providers": {},
                "content_types": {},
            }
        
        hourly = self.hourly_metrics[hour_key]
        hourly["total_requests"] += 1
        
        if trace.status == RequestStatus.SUCCESS:
            hourly["successful_requests"] += 1
        elif trace.status in (RequestStatus.FAILURE, RequestStatus.TIMEOUT):
            hourly["failed_requests"] += 1
        
        hourly["total_cost"] += trace.cost
        hourly["total_tokens"] += trace.token_count
        
        if trace.cache_hit:
            hourly["cache_hits"] += 1
        
        if trace.semantic_cache_hit:
            hourly["semantic_cache_hits"] += 1
        
        # Provider breakdown
        provider = trace.provider.value
        if provider not in hourly["providers"]:
            hourly["providers"][provider] = {
                "requests": 0,
                "cost": 0.0,
            }
        
        hourly["providers"][provider]["requests"] += 1
        hourly["providers"][provider]["cost"] += trace.cost
        
        # Content type breakdown
        content_type = trace.content_type.value
        if content_type not in hourly["content_types"]:
            hourly["content_types"][content_type] = {
                "requests": 0,
                "cost": 0.0,
            }
        
        hourly["content_types"][content_type]["requests"] += 1
        hourly["content_types"][content_type]["cost"] += trace.cost
        
        # Daily metrics (similar structure)
        day_key = trace.start_time.strftime("%Y-%m-%d")
        
        if day_key not in self.daily_metrics:
            self.daily_metrics[day_key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "cache_hits": 0,
                "semantic_cache_hits": 0,
                "providers": {},
                "content_types": {},
                "hours": {},
            }
        
        daily = self.daily_metrics[day_key]
        daily["total_requests"] += 1
        
        if trace.status == RequestStatus.SUCCESS:
            daily["successful_requests"] += 1
        elif trace.status in (RequestStatus.FAILURE, RequestStatus.TIMEOUT):
            daily["failed_requests"] += 1
        
        daily["total_cost"] += trace.cost
        daily["total_tokens"] += trace.token_count
        
        if trace.cache_hit:
            daily["cache_hits"] += 1
        
        if trace.semantic_cache_hit:
            daily["semantic_cache_hits"] += 1
        
        # Provider breakdown
        if provider not in daily["providers"]:
            daily["providers"][provider] = {
                "requests": 0,
                "cost": 0.0,
            }
        
        daily["providers"][provider]["requests"] += 1
        daily["providers"][provider]["cost"] += trace.cost
        
        # Content type breakdown
        if content_type not in daily["content_types"]:
            daily["content_types"][content_type] = {
                "requests": 0,
                "cost": 0.0,
            }
        
        daily["content_types"][content_type]["requests"] += 1
        daily["content_types"][content_type]["cost"] += trace.cost
        
        # Hour breakdown
        hour = trace.start_time.hour
        if hour not in daily["hours"]:
            daily["hours"][hour] = {
                "requests": 0,
                "cost": 0.0,
            }
        
        daily["hours"][hour]["requests"] += 1
        daily["hours"][hour]["cost"] += trace.cost
    
    def _log_trace_to_file(self, trace: RequestTrace) -> None:
        """Log a trace to a file.
        
        Args:
            trace: The trace to log.
        """
        if not self.metrics_dir:
            return
        
        # Create date-based directory structure
        date_dir = self.metrics_dir / trace.start_time.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Write trace to file
        trace_file = date_dir / f"{trace.request_id}.json"
        with open(trace_file, "w") as f:
            f.write(trace.json(indent=2))
    
    def get_metrics_summary(
        self,
        days: int = 7,
        provider: Optional[ProviderType] = None,
        content_type: Optional[ContentType] = None,
    ) -> Dict[str, Any]:
        """Get a summary of metrics.
        
        Args:
            days: Number of days to include in summary.
            provider: Filter by provider.
            content_type: Filter by content type.
            
        Returns:
            Dict[str, Any]: Metrics summary.
        """
        # Basic metrics from request traces
        summary = {
            "total_requests": self.get_request_count(),
            "total_cost": self.get_total_cost(),
            "cache_hits": self.get_cache_hit_count(),
            "cost_by_provider": self.get_cost_by_provider()
        }
        
        return summary
        
    def get_total_cost(self) -> float:
        """Get the total cost of all AI requests.
        
        Returns:
            float: The total cost in USD.
        """
        total_cost = 0.0
        for trace in self.request_traces.values():
            if trace.cost is not None:
                total_cost += trace.cost
        return total_cost
        
    def get_cost_by_provider(self) -> Dict[ProviderType, float]:
        """Get costs broken down by provider.
        
        Returns:
            Dict[ProviderType, float]: Provider-specific costs in USD.
        """
        provider_costs: Dict[ProviderType, float] = {}
        for trace in self.request_traces.values():
            if trace.cost is not None:
                if trace.provider not in provider_costs:
                    provider_costs[trace.provider] = 0.0
                provider_costs[trace.provider] += trace.cost
        return provider_costs
        
    def get_request_count(self) -> int:
        """Get the total number of requests made.
        
        Returns:
            int: Total request count.
        """
        return len(self.request_traces)
        
    def get_cache_hit_count(self) -> int:
        """Get the number of cache hits.
        
        Returns:
            int: Number of cache hits.
        """
        return sum(1 for trace in self.request_traces.values() 
                 if trace.status == RequestStatus.CACHED)
        
        provider_breakdown = {}
        content_type_breakdown = {}
        daily_breakdown = {}
        
        for day, metrics in filtered_days.items():
            # Apply provider filter if specified
            if provider:
                provider_key = provider.value
                if provider_key not in metrics["providers"]:
                    continue
                
                provider_metrics = metrics["providers"][provider_key]
                day_requests = provider_metrics["requests"]
                day_cost = provider_metrics["cost"]
            else:
                day_requests = metrics["total_requests"]
                day_cost = metrics["total_cost"]
            
            # Apply content type filter if specified
            if content_type:
                content_key = content_type.value
                if content_key not in metrics["content_types"]:
                    continue
                
                content_metrics = metrics["content_types"][content_key]
                day_requests = content_metrics["requests"]
                day_cost = content_metrics["cost"]
            
            # Aggregate totals
            total_requests += day_requests
            successful_requests += metrics["successful_requests"]
            failed_requests += metrics["failed_requests"]
            total_cost += day_cost
            total_tokens += metrics["total_tokens"]
            cache_hits += metrics["cache_hits"]
            semantic_cache_hits += metrics["semantic_cache_hits"]
            
            # Daily breakdown
            daily_breakdown[day] = {
                "requests": day_requests,
                "cost": day_cost,
            }
            
            # Provider breakdown
            for p, p_metrics in metrics["providers"].items():
                if provider and p != provider.value:
                    continue
                
                if p not in provider_breakdown:
                    provider_breakdown[p] = {
                        "requests": 0,
                        "cost": 0.0,
                    }
                
                provider_breakdown[p]["requests"] += p_metrics["requests"]
                provider_breakdown[p]["cost"] += p_metrics["cost"]
            
            # Content type breakdown
            for ct, ct_metrics in metrics["content_types"].items():
                if content_type and ct != content_type.value:
                    continue
                
                if ct not in content_type_breakdown:
                    content_type_breakdown[ct] = {
                        "requests": 0,
                        "cost": 0.0,
                    }
                
                content_type_breakdown[ct]["requests"] += ct_metrics["requests"]
                content_type_breakdown[ct]["cost"] += ct_metrics["cost"]
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "cache_hit_rate": (cache_hits / total_requests * 100) if total_requests > 0 else 0,
            "semantic_cache_hit_rate": (semantic_cache_hits / total_requests * 100) if total_requests > 0 else 0,
            "cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "providers": provider_breakdown,
            "content_types": content_type_breakdown,
            "daily": daily_breakdown,
        }
    
    def get_request_trace(self, request_id: str) -> Optional[RequestTrace]:
        """Get a request trace by ID.
        
        Args:
            request_id: The ID of the request trace.
            
        Returns:
            Optional[RequestTrace]: The request trace if found.
        """
        return self.request_traces.get(request_id)
    
    def cleanup_old_traces(self) -> int:
        """Clean up old request traces.
        
        Returns:
            int: Number of traces removed.
        """
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        initial_count = len(self.request_traces)
        self.request_traces = {
            request_id: trace
            for request_id, trace in self.request_traces.items()
            if trace.start_time >= cutoff_time
        }
        
        removed = initial_count - len(self.request_traces)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old request traces")
        
        return removed
