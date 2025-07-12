"""Enterprise-grade AI service orchestrator with advanced reliability features.

This module provides a comprehensive AI service orchestration layer that integrates
semantic caching, provider health monitoring, metrics collection, and advanced
fallback strategies for reliable AI service delivery.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from pydantic import BaseModel, Field

from .caching.semantic_cache import SemanticCache
from .circuit_breaker import CircuitBreaker
from .data_models import (
    AIConfig,
    AIRequest,
    AIResponse,
    ContentType,
    OptimizationResult,
    ProviderType,
)
from .metrics_collector import MetricsCollector, RequestStatus
from .cost_optimization_manager import CostOptimizationManager
from .provider_health_monitor import ProviderHealthMonitor
from .provider_interfaces import (
    BaseAIProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    TemplateProvider,
)


logger = logging.getLogger(__name__)


class AIServiceOrchestrator:
    """Enterprise-grade AI service orchestration with advanced reliability features.
    
    This orchestrator manages multiple AI providers with intelligent routing,
    semantic caching, health monitoring, metrics collection, and fallback strategies.
    """
    
    def __init__(self, config: AIConfig):
        """Initialize the AI service orchestrator.
        
        Args:
            config: Configuration for the orchestrator and providers.
        """
        self.config = config
        self.cost_manager = CostOptimizationManager(config)
        
        # Initialize fallback provider first
        self.fallback_provider = TemplateProvider(config, self.cost_manager)
        
        # Initialize other providers
        self.providers = self._initialize_providers()
        
        # Set to track providers that have exceeded their quota in this session
        self.excluded_providers = set()
        
        # Initialize reliability components
        from .circuit_breaker import CircuitBreakerConfig
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=2
        )
        
        self.circuit_breakers = {
            provider_type: CircuitBreaker(config=circuit_breaker_config)
            for provider_type in self.providers
        }
        
        # Initialize semantic cache
        self.semantic_cache = SemanticCache(
            similarity_threshold=config.cache_similarity_threshold
            if hasattr(config, "cache_similarity_threshold") else 0.85
        )
        
        # Initialize health monitoring with ProviderType values only
        provider_types = [
            provider_type for provider_type in self.providers.keys() 
            if isinstance(provider_type, ProviderType)  # Only include ProviderType values
        ]
        
        # If no valid provider types found, use all available providers
        if not provider_types:
            provider_types = list(ProviderType)
            
        self.health_monitor = ProviderHealthMonitor(
            providers=provider_types,
            degraded_threshold=0.2,
            unhealthy_threshold=0.5,
        )
        
        # Initialize metrics collection
        self.metrics_collector = MetricsCollector(
            metrics_dir=config.metrics_dir if hasattr(config, "metrics_dir") else None,
            enable_file_logging=config.enable_metrics_logging
            if hasattr(config, "enable_metrics_logging") else False,
        )
        
        logger.info(
            f"AI Service Orchestrator initialized with {len(self.providers)} providers"
        )
    
    def select_provider(self, provider_preference: Optional[ProviderType] = None) -> BaseAIProvider:
        """Select best available provider based on quotas and health.
        
        Args:
            provider_preference: Optional preferred provider type.
            
        Returns:
            Selected AI service provider.
            
        Raises:
            ServiceUnavailableError: If no suitable provider is available.
        """
        # If specific provider requested, try to use it if available
        if provider_preference:
            if provider_preference in self.providers:
                # Check if circuit breaker allows calls to this provider
                if not self.circuit_breakers[provider_preference].is_open():
                    logger.info(f"Using explicitly requested provider: {provider_preference}")
                    # Record usage for this model if it's a model type
                    from src.ai_integration.cost.provider_models import ModelType
                    if isinstance(provider_preference, ModelType):
                        self.usage_tracker.record_request(provider_preference)
                    return self.providers[provider_preference]
                else:
                    logger.warning(
                    f"Circuit breaker open for {provider_preference}. Using fallback."
                    )
            else:
                # If template provider is requested or unavailable, use fallback
                if provider_preference == ModelType.TEMPLATE or not self.providers:
                    return self.fallback_provider
                # Otherwise warn that preferred provider is unavailable
                logger.warning(
                    f"Provider {provider_preference} unavailable. Using fallback."
                )
        
        # If no preference specified, use quota-aware selection
        # Get models with available quota, sorted by priority
        available_models = self.usage_tracker.get_available_models()
        
        logger.info(f"Available models with quota: {available_models}")
        
        # Try each available model in priority order
        for model_type in available_models:
            if model_type not in self.providers:
                logger.warning(f"Model {model_type} not in providers, skipping")
                continue
                
            # Check circuit breaker
            if model_type in self.circuit_breakers:
                if self.circuit_breakers[model_type].is_open():
                    logger.info(f"Circuit breaker open for {model_type}, skipping")
                    continue
            
            # Record usage for this model
            self.usage_tracker.record_request(model_type)
            logger.info(f"Selected provider: {model_type}")
            return self.providers[model_type]
        
        # If no models available, log quota status and return template
        self._log_quota_exhaustion()
        return self.fallback_provider
        
    def _log_quota_exhaustion(self):
        """Log current quota usage for debugging."""
        from src.ai_integration.cost.model_configs import FREE_TIER_QUOTAS
        from src.ai_integration.cost.provider_models import ModelType
        
        logger.warning("All AI models quota exhausted for today:")
        for model_type, quota_info in FREE_TIER_QUOTAS.items():
            if model_type == ModelType.TEMPLATE:
                continue
            usage = self.usage_tracker.get_usage_count(model_type)
            limit = quota_info["daily_limit"]
            logger.warning(f"  {model_type}: {usage}/{limit} requests used")
    
    def _initialize_providers(self) -> Dict[ProviderType, BaseAIProvider]:
        """Initialize all available AI providers.
        
        Returns:
            Dict[ProviderType, BaseAIProvider]: Dictionary of available providers.
        """
        providers = {}
        
        # Try to initialize each provider
        try:
            openai_provider = OpenAIProvider(self.config, self.cost_manager)
            if openai_provider.available:
                providers[ProviderType.OPENAI] = openai_provider
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        try:
            anthropic_provider = AnthropicProvider(self.config, self.cost_manager)
            if anthropic_provider.available:
                providers[ProviderType.ANTHROPIC] = anthropic_provider
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")
        
        # Initialize a single Gemini provider to be used for all Gemini models
        from src.ai_integration.cost.provider_models import ModelType
        
        gemini_models = [
            ModelType.GEMINI_20_FLASH,
            ModelType.GEMINI_15_FLASH_002, 
            ModelType.GEMINI_15_FLASH,
            ModelType.GEMINI_15_FLASH_8B
        ]
        
        try:
            # Create a single Gemini provider
            gemini_provider = GeminiProvider(self.config, self.cost_manager)
            if gemini_provider.available:
                # Map the same provider instance to all model types
                for model_type in gemini_models:
                    providers[model_type] = gemini_provider
                logger.info(f"Initialized Gemini provider for {len(gemini_models)} models")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini provider: {e}")
        
        # Always include template provider as final fallback
        providers[ModelType.TEMPLATE] = self.fallback_provider
        
        # Initialize usage tracker
        from src.ai_integration.cost.usage_tracker import MultiModelUsageTracker
        self.usage_tracker = MultiModelUsageTracker()
        
        logger.info(f"Initialized {len(providers)} providers: {list(providers.keys())}")
        return providers
    
    async def generate_content(
        self,
        request: AIRequest,
        optimization_result: Optional[OptimizationResult] = None,
    ) -> AIResponse:
        """Generate content with multi-model failover and usage tracking."""
        # Use optimized request if available
        effective_request = (
            optimization_result.optimized_request if optimization_result else request
        )
        
        start_time = time.time()
        
        # Step 1: Check semantic cache for similar requests
        cached_response = self.semantic_cache.get_similar(effective_request)
        if cached_response:
            # Record cache hit in metrics
            request_id = self.metrics_collector.start_request_trace(
                request=effective_request,
                provider=cached_response.provider,
                tags={"source": "semantic_cache"},
            )
            
            self.metrics_collector.complete_request_trace(
                request_id=request_id,
                status=RequestStatus.CACHED,
                response=cached_response,
                cache_hit=True,
                semantic_cache_hit=True,
                cost=0.0,  # No cost for cached responses
            )
            
            logger.info(f"Semantic cache hit for {effective_request.content_type}")
            return cached_response
        
        # Step 2: Select best available provider with quota awareness
        selected_provider = self.select_provider()
        
        # If all quotas exhausted, use template fallback
        from src.ai_integration.cost.provider_models import ModelType
        if selected_provider == ModelType.TEMPLATE:
            logger.warning("Using template fallback - all AI models quota exhausted")
            template_response = await self.fallback_provider.generate_content(request)
            
            # Start trace for template fallback
            request_id = self.metrics_collector.start_request_trace(
                request=effective_request,
                provider=ModelType.TEMPLATE,
                tags={"quota_exhausted": "true"},
            )
            
            # Complete metrics trace
            self.metrics_collector.complete_request_trace(
                request_id=request_id,
                status=RequestStatus.SUCCESS,
                response=template_response,
                cost=0.0,  # No cost for template fallback
                fallback_used=True,
                fallback_reason="quota_exhausted",
            )
            
            return template_response
        
        # Try the selected AI provider
        try:
            logger.info(f"Attempting generation with {selected_provider}")
            
            # Set the provider_type in the request so the provider knows which model to use
            # This is especially important for Gemini models
            effective_request.provider_type = selected_provider
            
            # Ensure model field is always set with the exact model value
            if hasattr(selected_provider, 'value'):
                effective_request.model = selected_provider.value
                logger.info(f"Set model in request to {effective_request.model} from {selected_provider}")
            else:
                effective_request.model = str(selected_provider)
                logger.info(f"Set model in request to {effective_request.model} (string conversion)")
            
            # Start request trace for metrics
            request_id = self.metrics_collector.start_request_trace(
                request=effective_request,
                provider=selected_provider,
                tags={"optimized": "true" if optimization_result else "false"},
            )
            
            # Check circuit breaker
            if selected_provider in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[selected_provider]
                if circuit_breaker.is_open():
                    logger.warning(f"Circuit breaker open for {selected_provider}")
                    
                    # Update request trace with circuit breaker info
                    self.metrics_collector.complete_request_trace(
                        request_id=request_id,
                        status=RequestStatus.FAILURE,
                        error_message="Circuit breaker open",
                        fallback_used=True,
                        fallback_reason="circuit_breaker_open",
                    )
                    
                    # Try next available provider (recursive call)
                    logger.info("Trying next available provider...")
                    return await self.generate_content(request)
            
            # Make the request
            provider = self.providers[selected_provider]
            response = await provider.generate(request)
            
            # Record successful request in health monitor
            request_duration_ms = (time.time() - start_time) * 1000
            self.health_monitor.record_request(
                provider=selected_provider,
                success=True,
                latency_ms=request_duration_ms,
            )
            
            # Log successful usage
            from src.ai_integration.cost.model_configs import FREE_TIER_QUOTAS
            if selected_provider in FREE_TIER_QUOTAS:
                usage_count = self.usage_tracker.get_usage_count(selected_provider)
                quota_limit = FREE_TIER_QUOTAS[selected_provider]["daily_limit"]
                logger.info(f"{selected_provider} request successful. Usage: {usage_count}/{quota_limit}")
            
            # Complete metrics trace
            self.metrics_collector.complete_request_trace(
                request_id=request_id,
                status=RequestStatus.SUCCESS,
                response=response,
                cost=response.cost,
                token_savings=(
                    optimization_result.token_savings
                    if optimization_result
                    else None
                ),
                context_compression_ratio=(
                    optimization_result.compression_ratio
                    if optimization_result
                    else None
                )
            )
            
            # Cache the response
            self.semantic_cache.store(effective_request, response)
            
            return response
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"{selected_provider} failed: {error_str}")
            
            # Record circuit breaker failure
            if selected_provider in self.circuit_breakers:
                self.circuit_breakers[selected_provider].record_failure()
                
            # Check specifically for quota exceeded errors
            if "quota" in error_str.lower() or "rate limit" in error_str.lower() or "429" in error_str:
                logger.warning(f"Quota exceeded for {selected_provider}, marking as unavailable")
                
                # Ensure this provider is excluded from future attempts this session
                if selected_provider not in self.excluded_providers:
                    self.excluded_providers.add(selected_provider)
                
                # Create a modified request that excludes the current provider
                modified_request = copy.deepcopy(request)
                if not hasattr(modified_request, 'excluded_providers'):
                    modified_request.excluded_providers = set()
                modified_request.excluded_providers.add(selected_provider)
                
                # Try with a specifically selected fallback provider
                fallback_provider = self._select_fallback_provider(selected_provider)
                logger.info(f"Selected fallback provider: {fallback_provider}")
                
                # Try next available provider (recursive call with modified request)
                return await self.generate_content(modified_request)
            else:
                # For other types of errors, try next available provider
                logger.info("Trying next available provider...")
                return await self.generate_content(request)
            
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current usage summary for all models."""
        from src.ai_integration.cost.model_configs import FREE_TIER_QUOTAS
        from src.ai_integration.cost.provider_models import ModelType
        
        summary = {}
        for model_type, quota_info in FREE_TIER_QUOTAS.items():
            if model_type == ModelType.TEMPLATE:
                continue
            
            usage = self.usage_tracker.get_usage_count(model_type)
            limit = quota_info["daily_limit"]
            available = self.usage_tracker.is_quota_available(model_type)
            
            summary[model_type] = {
                "usage": usage,
                "limit": limit, 
                "available": available,
                "percentage": (usage / limit * 100) if limit > 0 else 0
            }
        
        return summary
    
    def _select_provider(self, request: AIRequest) -> ProviderType:
        """Select the optimal provider for a request.
        
        This method considers:
        - Explicit provider preference in the request
        - Provider health status
        - Cost constraints
        - Content type suitability
        - Quota status (excluded providers)
        
        Args:
            request: The AI request.
            
        Returns:
            ProviderType: The selected provider.
        """
        logger.info(f"Selecting provider for request type: {request.content_type}")
        logger.debug(f"Available providers: {list(self.providers.keys())}")
        
        # Check for providers excluded due to quota limits
        excluded = set()
        
        # Add providers that have exceeded quota in this session
        excluded.update(self.excluded_providers)
        
        # Add providers explicitly excluded in this request (from previous failures)
        if hasattr(request, 'excluded_providers'):
            excluded.update(request.excluded_providers)
            
        if excluded:
            logger.info(f"Excluding providers due to quota limitations: {excluded}")
        
        
        # Check for explicit preference
        if request.provider_preference and request.provider_preference in self.providers:
            preferred = request.provider_preference
            logger.info(f"Explicit provider preference found: {preferred}")
            
            # Check if preferred provider is healthy
            health_status = self.health_monitor.get_provider_health(preferred)
            logger.info(f"Health status for {preferred}: {health_status}")
            
            if health_status != "unhealthy":
                logger.info(f"Using preferred provider: {preferred}")
                return preferred
            else:
                logger.warning(
                    f"Preferred provider {preferred} is unhealthy, selecting alternative"
                )
        else:
            logger.info(f"No explicit provider preference in request or preference not available")
        
        # Get available (healthy or degraded) providers
        all_healthy_providers = self.health_monitor.get_available_providers()
        
        # Filter out excluded providers
        available_providers = [p for p in all_healthy_providers if p not in excluded]
        logger.info(f"Available healthy providers: {all_healthy_providers}")
        logger.info(f"Available providers after quota filtering: {available_providers}")
        
        # If no providers are available, use template fallback
        if not available_providers:
            logger.warning("No healthy providers available, using template fallback")
            return ProviderType.TEMPLATE
        
        # Check circuit breaker status for Gemini specifically
        if ProviderType.GEMINI in self.circuit_breakers:
            gemini_circuit_open = self.circuit_breakers[ProviderType.GEMINI].is_open()
            logger.info(f"Gemini circuit breaker status: {'OPEN' if gemini_circuit_open else 'CLOSED'}")
        else:
            logger.info("No circuit breaker configured for Gemini")
            
        # If cost is constrained, prioritize by cost
        if request.max_cost:
            logger.info(f"Cost constraint present: {request.max_cost}")
            # Sort by estimated cost
            sorted_providers = sorted(
                available_providers,
                key=lambda p: self.providers[p].estimate_cost(request)
                if p in self.providers
                else float("inf"),
            )
            selected = sorted_providers[0] if sorted_providers else ProviderType.TEMPLATE
            logger.info(f"Selected lowest cost provider: {selected}")
            return selected
        
        # Otherwise, use preferred order based on capabilities
        preferred_order = [
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.GEMINI,
            ProviderType.TEMPLATE,
        ]
        logger.info(f"Using preferred provider order: {preferred_order}")
        
        for provider in preferred_order:
            if provider in available_providers:
                logger.info(f"Selected provider from preferred order: {provider}")
                return provider
            else:
                logger.debug(f"Provider {provider} not in available providers, skipping")
        
        # Fallback to template if no preferred providers are available
        logger.warning("No providers from preferred order available, using template fallback")
        return ProviderType.TEMPLATE
    
    def _select_fallback_provider(self, failed_provider: ProviderType) -> ProviderType:
        """Select a fallback provider when the primary provider fails.
        
        Args:
            failed_provider: The provider that failed.
            
        Returns:
            ProviderType: The selected fallback provider.
        """
        # Get available providers excluding the failed one
        available_providers = [
            p for p in self.health_monitor.get_available_providers()
            if p != failed_provider
        ]
        
        # If no providers are available, use template fallback
        if not available_providers:
            logger.warning("No healthy fallback providers available, using template fallback")
            return ProviderType.TEMPLATE
        
        # Use preferred order for fallback
        preferred_order = [
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.GEMINI,
            ProviderType.TEMPLATE,
        ]
        
        for provider in preferred_order:
            if provider in available_providers:
                return provider
        
        # Fallback to template if no preferred providers are available
        return ProviderType.TEMPLATE
    
    def get_provider_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of provider health status.
        
        Returns:
            Dict[str, Dict[str, Any]]: Summary of provider health metrics.
        """
        return self.health_monitor.get_metrics_summary()
    
    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of AI usage metrics.
        
        Args:
            days: Number of days to include in summary.
            
        Returns:
            Dict[str, Any]: Metrics summary.
        """
        # FUTURE: Implement comprehensive metrics aggregation from MetricsCollector
        return {}  # Placeholder
        
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about the semantic cache performance.
        
        Returns:
            Dict[str, Any]: Cache statistics including hit/miss rates and entry counts.
        """
        current_time = time.time()
        cached_entries = len(self.semantic_cache.cache) if hasattr(self.semantic_cache, "cache") else 0
        total_requests = self.metrics_collector.get_request_count()
        cache_hits = self.metrics_collector.get_cache_hit_count()
        
        # Calculate cache hit rate if there have been requests
        hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0
        miss_rate = 1.0 - hit_rate if total_requests > 0 else 0.0
        
        # Calculate average age of cache entries
        entry_ages = []
        if hasattr(self.semantic_cache, "cache"):
            for entry in self.semantic_cache.cache.values():
                entry_ages.append(current_time - entry.timestamp)
                
        avg_age = sum(entry_ages) / len(entry_ages) if entry_ages else 0.0
        
        return {
            "cached_entries": cached_entries,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "avg_entry_age_seconds": avg_age,
            "similarity_threshold": self.semantic_cache.similarity_threshold
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary information about AI service costs.
        
        Returns:
            Dict[str, Any]: Cost metrics including total cost, cost per provider.
        """
        # Gather cost data from metrics collector
        total_cost = self.metrics_collector.get_total_cost()
        cost_by_provider = self.metrics_collector.get_cost_by_provider()
        
        return {
            "total_cost": total_cost,
            "cost_by_provider": cost_by_provider,
            "average_cost_per_request": total_cost / self.metrics_collector.get_request_count() 
                if self.metrics_collector.get_request_count() > 0 else 0.0
        }

    def get_available_providers(self) -> List[ProviderType]:
        """Get a list of available provider types in priority order.
        
        Returns:
            List[ProviderType]: List of available provider types, excluding template provider.
        """
        # Start with the available provider types (keys from self.providers)
        available_providers = list(self.providers.keys())
        
        # Filter out template provider (we don't want to include it in the cascade)
        if ProviderType.TEMPLATE in available_providers:
            available_providers.remove(ProviderType.TEMPLATE)
            
        # Filter out providers with open circuit breakers (only check providers that have circuit breakers)
        available_providers = [
            provider_type for provider_type in available_providers 
            if provider_type not in self.circuit_breakers or not self.circuit_breakers[provider_type].is_open()
        ]
            
        # Import provider preferences from cost manager
        from src.ai_integration.cost_manager import COST_THRESHOLDS
        
        # Get the preferred provider order
        model_preferences = COST_THRESHOLDS.get("provider_preferences", [])
        
        # Build priority list
        provider_priority = []
        for model_name in model_preferences:
            # Skip template fallback
            if model_name == "template_fallback":
                continue
                
            # Map model names to provider types
            if model_name.startswith("gemini"):
                provider_type = ProviderType.GEMINI
            elif model_name.startswith("claude"):
                provider_type = ProviderType.ANTHROPIC
            elif model_name.startswith("gpt"):
                provider_type = ProviderType.OPENAI
            else:
                continue
                
            # Only add if this provider is actually available
            if provider_type in available_providers and provider_type not in provider_priority:
                provider_priority.append(provider_type)
        
        # Add any remaining available providers not in the priority list
        for provider_type in available_providers:
            if provider_type not in provider_priority:
                provider_priority.append(provider_type)
                
        logger.info(f"Available providers in priority order: {provider_priority}")
        return provider_priority
    
    def cleanup_resources(self) -> None:
        """Clean up resources and perform maintenance tasks.
        
        This includes cleaning old cache entries and metrics data.
        """
        # Clean cache
        self.semantic_cache.cleanup()
        
        # Clean metrics
        self.metrics_collector.cleanup_old_traces()
        
        logger.info("Cleaned up AI service orchestrator resources")
