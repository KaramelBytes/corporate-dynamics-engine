"""Cost Optimization Manager for Corporate Dynamics Simulator.

This module provides budget tracking, cost estimation, and optimization features
for the AI integration layer. It enables efficient use of AI resources while
maintaining quality and staying within budget constraints.

Note: This is a minimal implementation with stub methods for complex features.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .data_models import AIConfig, AIRequest, AIResponse, CostOptimizationResult, ProviderType
from .usage_logger import UsageLogger

logger = logging.getLogger(__name__)


class BudgetUsage(BaseModel):
    """Records budget usage over time."""
    
    timestamp: float
    provider: str
    request_type: str
    tokens: int
    cost: float
    request_id: str


class CostOptimizationManager:
    """Manages AI service costs, budgets, and optimization strategies.
    
    This minimal implementation includes basic budget tracking and cost estimation
    with stub methods for more complex features.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize the cost optimization manager.
        
        Args:
            config: Configuration for cost management.
        """
        self.config = config or AIConfig()
        self.usage_log: List[BudgetUsage] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        
        # Initialize provider calculators
        self.provider_calculators = {
            ProviderType.OPENAI: self._openai_calculator(),
            ProviderType.ANTHROPIC: self._anthropic_calculator(),
            ProviderType.GEMINI: self._gemini_calculator(),
            ProviderType.TEMPLATE: self._template_calculator(),
        }
        
        # Initialize context optimizer and cache
        self.context_optimizer = self._create_context_optimizer()
        self.cache = self._create_cache()
        self.usage_logger = UsageLogger()
        
        self._load_usage_history()
    
    def _load_usage_history(self) -> None:
        """Load historical usage data from storage.
        
        In a full implementation, this would load from a database or file.
        """
        # FUTURE: Add persistent storage for enterprise cost tracking and historical analysis
        # For now, we'll just initialize with empty data
        self.usage_log = []
        self.total_cost = 0.0
        self.total_tokens = 0
    
    def record_usage(self, response: AIResponse) -> None:
        """Record API usage for budget tracking.
        
        Args:
            response: The AI response with usage information.
        """
        usage = BudgetUsage(
            timestamp=time.time(),
            provider=response.provider,
            request_type=response.content_type,
            tokens=response.tokens_used,
            cost=response.cost,
            request_id=response.request_id,
        )
        
        self.usage_log.append(usage)
        self.total_cost += response.cost
        self.total_tokens += response.tokens_used
        
        logger.info(
            f"Recorded API usage: {response.tokens_used} tokens, "
            f"${response.cost:.6f}, provider={response.provider}, "
            f"total_cost=${self.total_cost:.4f}"
        )
        
        # Check if we're approaching budget limits
        self._check_budget_limits()
    
    def _check_budget_limits(self) -> None:
        """Check if current usage is approaching budget limits with detailed alerts."""
        monthly_budget = self.config.monthly_budget
        
        if monthly_budget <= 0:
            return  # No budget set, nothing to check
            
        usage_percent = (self.total_cost / monthly_budget) * 100
        
        if usage_percent >= 90:
            logger.warning(
                f"CRITICAL BUDGET ALERT: Used {usage_percent:.1f}% of monthly budget! "
                f"${self.total_cost:.2f} of ${monthly_budget:.2f}"
            )
        elif usage_percent >= 80:
            logger.warning(
                f"Budget warning: {usage_percent:.1f}% of monthly budget used. "
                f"${self.total_cost:.2f} of ${monthly_budget:.2f}"
            )
        elif usage_percent >= 50:
            logger.info(
                f"Budget update: {usage_percent:.1f}% of monthly budget used. "
                f"${self.total_cost:.2f} of ${monthly_budget:.2f}"
            )

    def record_gemini_usage(
        self, model_name: str, prompt_text: str, response_text: str
    ) -> None:
        """
        Records Gemini API usage and logs it.

        This method estimates token counts from text and calculates cost based on
        specific Gemini pricing, then logs the request to the usage logger.

        Args:
            model_name: The name of the Gemini model used.
            prompt_text: The text of the prompt sent to the API.
            response_text: The text of the response received from the API.
        """
        # Estimate tokens: ~4 characters per token
        prompt_tokens = len(prompt_text) // 4
        response_tokens = len(response_text) // 4

        # Calculate cost based on specific Gemini pricing.
        # This cost is for detailed logging here. The UsageLogger may use a
        # different, more general cost estimation.
        cost = (prompt_tokens * 0.075 / 1_000_000) + (
            response_tokens * 0.30 / 1_000_000
        )

        # Log the request using the central usage logger
        self.usage_logger.log_request(
            model_name=model_name,
            prompt_length=prompt_tokens,
            response_length=response_tokens,
            timestamp=datetime.now(),
        )

        logger.info(
            "Logged Gemini usage for %s. Prompt: %d tokens, Response: %d tokens. Estimated cost: $%.8f",
            model_name,
            prompt_tokens,
            response_tokens,
            cost,
        )
    
    # Provider calculator methods
    def _openai_calculator(self):
        """Create a calculator for OpenAI pricing."""
        class OpenAICalculator:
            def get_default_model(self) -> str:
                """Get the default model for this provider."""
                # Import model type for free-tier model reference
                from src.ai_integration.cost.provider_models import ModelType
                return ModelType.GPT4O_MINI.value  # Return the free tier model
            
            def supports_model(self, model: str) -> bool:
                """Check if the model is supported by this provider."""
                # All common OpenAI models including free-tier options
                supported_models = [
                    # Free tier models
                    "gpt-4o-mini",
                    # Legacy and paid models 
                    "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", 
                    "gpt-3.5-turbo-16k"
                ]
                # Check if the model starts with any of the supported prefixes
                return any(model.startswith(prefix) for prefix in supported_models)
            
            def estimate_output_tokens(self, model: str, input_text: str, max_tokens: int) -> int:
                """Estimate the number of output tokens for a given input and model.
                
                Args:
                    model: The model to use for estimation
                    input_text: The input text
                    max_tokens: The maximum number of tokens to generate
                    
                Returns:
                    Estimated number of output tokens
                """
                # Simple heuristic: for most conversational use cases, output is ~50% of input
                # but capped by max_tokens and with a minimum reasonable response
                input_token_count = len(input_text) // 4  # Rough estimate: ~4 chars per token
                estimated_output = min(max(input_token_count // 2, 100), max_tokens)
                return estimated_output
                
            def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, is_streaming: bool = False) -> float:
                # Import model configs for pricing
                from src.ai_integration.cost.model_configs import MODEL_CONFIGS
                from src.ai_integration.cost.provider_models import ModelType
                
                # GPT-4o mini pricing (free tier model)
                if model == ModelType.GPT4O_MINI.value:
                    input_cost = 0.0000015 * input_tokens  # $0.0015 per 1K tokens
                    output_cost = 0.000002 * output_tokens   # $0.002 per 1K tokens
                # GPT-4 pricing
                elif model.startswith("gpt-4"):
                    input_cost = 0.00003 * input_tokens  # $0.03 per 1K tokens
                    output_cost = 0.00006 * output_tokens  # $0.06 per 1K tokens
                else:  # gpt-3.5-turbo and others
                    input_cost = 0.0000015 * input_tokens  # $0.0015 per 1K tokens
                    output_cost = 0.000002 * output_tokens  # $0.002 per 1K tokens
                return input_cost + output_cost
        return OpenAICalculator()
    
    def _anthropic_calculator(self):
        """Create a calculator for Anthropic pricing."""
        class AnthropicCalculator:
            def get_default_model(self) -> str:
                """Get the default model for this provider."""
                # Import model type for free-tier model reference
                from src.ai_integration.cost.provider_models import ModelType
                return ModelType.CLAUDE_35_HAIKU.value  # Return the free tier model
            
            def supports_model(self, model: str) -> bool:
                """Check if the model is supported by this provider."""
                # All common Anthropic Claude models including free-tier options
                supported_models = [
                    # Free tier models
                    "claude-3.5-haiku",
                    # Legacy and paid models
                    "claude-3-", "claude-3.5-", "claude-2", "claude-instant"
                ]
                # Check if the model starts with any of the supported prefixes
                return any(model.startswith(prefix) for prefix in supported_models)
            
            def estimate_output_tokens(self, model: str, input_text: str, max_tokens: int) -> int:
                """Estimate the number of output tokens for a given input and model.
                
                Args:
                    model: The model to use for estimation
                    input_text: The input text
                    max_tokens: The maximum number of tokens to generate
                    
                Returns:
                    Estimated number of output tokens
                """
                # Claude models can produce longer outputs than some other models
                # but still follows similar estimation patterns
                input_token_count = len(input_text) // 4  # Rough estimate: ~4 chars per token
                # Claude tends to be more verbose, so estimate 60% of input length
                estimated_output = min(max(int(input_token_count * 0.6), 150), max_tokens)
                return estimated_output
                
            def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, is_streaming: bool = False) -> float:
                # Import model configs for pricing
                from src.ai_integration.cost.model_configs import MODEL_CONFIGS
                from src.ai_integration.cost.provider_models import ModelType
                
                # Claude 3.5 Haiku pricing (free tier model)
                if model == ModelType.CLAUDE_35_HAIKU.value:
                    input_cost = 0.000003 * input_tokens   # $0.003 per 1K tokens
                    output_cost = 0.000015 * output_tokens  # $0.015 per 1K tokens
                # Claude 3 models
                elif model.startswith("claude-3") or model.startswith("claude-instant"):
                    input_cost = 0.000008 * input_tokens  # $0.008 per 1K tokens
                    output_cost = 0.000024 * output_tokens  # $0.024 per 1K tokens
                else:  # Other Claude models
                    input_cost = 0.000011 * input_tokens  # $0.011 per 1K tokens
                    output_cost = 0.000032 * output_tokens  # $0.032 per 1K tokens
                return input_cost + output_cost
        return AnthropicCalculator()
    
    def _gemini_calculator(self):
        """Create a calculator for Google Gemini pricing."""
        class GeminiCalculator:
            def get_default_model(self) -> str:
                """Get the default model for this provider."""
                # Import model type for free-tier model reference
                from src.ai_integration.cost.provider_models import ModelType
                return ModelType.GEMINI_15_FLASH_8B.value  # Return the free tier model
            
            def supports_model(self, model: str) -> bool:
                """Check if the model is supported by this provider."""
                # All common Google Gemini models including free-tier options
                supported_models = [
                    # Free tier models
                    "gemini-1.5-flash", "gemini-1.5-flash-8b",
                    # Legacy and paid models
                    "gemini-pro", "gemini-ultra", "gemini-nano", "gemini-1.5-pro"
                ]
                # Check if the model starts with any of the supported prefixes
                return any(model.startswith(prefix) for prefix in supported_models)
            
            def estimate_output_tokens(self, model: str, input_text: str, max_tokens: int) -> int:
                """Estimate the number of output tokens for a given input and model.
                
                Args:
                    model: The model to use for estimation
                    input_text: The input text
                    max_tokens: The maximum number of tokens to generate
                    
                Returns:
                    Estimated number of output tokens
                """
                input_token_count = len(input_text) // 4  # Rough estimate: ~4 chars per token
                
                # Different estimates based on model capability
                if model.startswith("gemini-1.5-flash"):
                    # Flash models are optimized for efficiency
                    estimated_output = min(max(int(input_token_count * 0.4), 80), max_tokens)
                elif model.startswith("gemini-ultra"):
                    # Ultra is more capable, estimate higher token usage
                    estimated_output = min(max(int(input_token_count * 0.55), 120), max_tokens)
                else:
                    # Other Gemini models, standard estimation
                    estimated_output = min(max(int(input_token_count * 0.45), 100), max_tokens)
                    
                return estimated_output
                
            def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, is_streaming: bool = False) -> float:
                # Import model configs for pricing
                from src.ai_integration.cost.provider_models import ModelType
                
                # Gemini 1.5 Flash pricing (free tier model)
                if model == ModelType.GEMINI_15_FLASH_8B.value:
                    input_cost = 0.0000005 * input_tokens  # $0.0005 per 1K tokens
                    output_cost = 0.0000015 * output_tokens # $0.0015 per 1K tokens
                # Ultra models
                elif model.startswith("gemini-ultra"):
                    input_cost = 0.00001 * input_tokens  # $0.01 per 1K tokens
                    output_cost = 0.00002 * output_tokens  # $0.02 per 1K tokens
                else:  # gemini-pro and others
                    input_cost = 0.000001 * input_tokens  # $0.001 per 1K tokens
                    output_cost = 0.000002 * output_tokens  # $0.002 per 1K tokens
                
                return input_cost + output_cost
                
        return GeminiCalculator()
    
    def _template_calculator(self):
        """Create a calculator for template provider (minimal cost)."""
        class TemplateCalculator:
            def get_default_model(self) -> str:
                """Get the default model for this provider."""
                return "template-default"
            
            def supports_model(self, model: str) -> bool:
                """Check if the model is supported by this provider."""
                # Template provider supports any model name as it doesn't use real models
                return True
            
            def estimate_output_tokens(self, model: str, input_text: str, max_tokens: int) -> int:
                """Estimate the number of output tokens for a given input and model.
                
                Args:
                    model: The model to use for estimation
                    input_text: The input text
                    max_tokens: The maximum number of tokens to generate
                    
                Returns:
                    Estimated number of output tokens
                """
                # For template provider, return a conservative token estimate
                # as it usually returns deterministic templates
                return min(400, max_tokens)  # Fixed output size for templates
                
            def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, is_streaming: bool = False) -> float:
                # Template provider is essentially free (minimal cost for tracking)
                return 0.0000001 * (input_tokens + output_tokens)
        return TemplateCalculator()
    
    def estimate_request_cost(self, request: AIRequest) -> float:
        """Estimate the cost of an AI request using provider-specific models.
        
        Args:
            request: The AI request to estimate cost for.
            
        Returns:
            float: Estimated cost in dollars.
        """
        provider = request.provider_preference or ProviderType.OPENAI
        calculator = self.provider_calculators.get(provider)
        
        if not calculator:
            # Fallback to default calculator if provider not found
            calculator = self.provider_calculators[ProviderType.OPENAI]
            
        # Get token counts using provider-specific tokenization when available
        input_tokens = self._count_tokens(request.prompt_template, provider)
        output_tokens = min(
            request.max_tokens or 500,  # Default to 500 tokens if not specified
            self._estimate_output_tokens(request.prompt_template, provider, request.max_tokens or 500)
        )
        
        # Get cost estimate from provider calculator
        # Use provider-specific default model if none specified
        default_model = "gpt-3.5-turbo"  # Default for OpenAI
        
        if provider == ProviderType.GEMINI:
            default_model = calculator.get_default_model()
            logger.debug(f"Using default Gemini model: {default_model}")
        elif provider == ProviderType.ANTHROPIC:
            default_model = "claude-3.5-haiku"
        
        model = request.model or default_model
        logger.debug(f"Using model for cost estimation: {model} (provider: {provider})")
        
        estimated_cost = calculator.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            is_streaming=request.stream
        )
        
        return estimated_cost
    
    def _count_tokens(self, text: str, provider: str = ProviderType.OPENAI) -> int:
        """Count the number of tokens in a text string using provider-specific tokenizers.
        
        Args:
            text: The text to count tokens for.
            provider: The provider to use for tokenization.
            
        Returns:
            int: Token count.
        """
        # Use provider's token counter if available
        calculator = self.provider_calculators.get(provider)
        if calculator and hasattr(calculator, 'count_tokens'):
            return calculator.count_tokens(text)
            
        # Fallback to simple approximation
        return len(text) // 4  # Rough approximation: 1 token ≈ 4 characters
        
    def _estimate_output_tokens(self, prompt: str, provider: str, max_tokens: int) -> int:
        """Estimate the number of output tokens for a given prompt.
        
        Args:
            prompt: The input prompt.
            provider: The provider being used.
            max_tokens: Maximum tokens requested.
            
        Returns:
            int: Estimated number of output tokens.
        """
        calculator = self.provider_calculators.get(provider)
        if calculator and hasattr(calculator, 'estimate_output_tokens'):
            return calculator.estimate_output_tokens(
                model=None,  # Will use default model for the provider
                input_text=prompt,
                max_tokens=max_tokens
            )
        
        # Default estimation: 30% of input tokens, up to max_tokens
        input_tokens = self._count_tokens(prompt, provider)
        return min(int(input_tokens * 0.3), max_tokens)
    
    def optimize_request(
        self, request: AIRequest, max_cost: Optional[float] = None
    ) -> CostOptimizationResult:
        """Optimize a request to reduce cost while maintaining quality.
        
        This method applies several optimization strategies:
        1. Semantic caching to reuse similar responses
        2. Context optimization to reduce token usage
        3. Provider/model selection based on cost and performance
        
        Args:
            request: The AI request to optimize.
            max_cost: Maximum allowed cost, defaults to None.
            
        Returns:
            CostOptimizationResult: The optimization result.
        """
        result = CostOptimizationResult()
        result.baseline_cost = self.estimate_request_cost(request)
        result.optimized_request = request
        
        # Check if we have a cached response first
        cached_response = self.manage_intelligent_caching(request)
        if cached_response:
            result.optimized_request = cached_response
            result.optimization_strategy = "cached_response"
            result.final_estimated_cost = 0.0  # Cached responses have no cost
            result.cache_hit = True
            return result
        
        # If no cache hit, apply optimization strategies
        optimized = request.copy(deep=True)
        optimization_strategies = []
        
        # 1. Optimize context if it's too large
        if self._should_optimize_context(request):
            optimization_result = self.context_optimizer.optimize(
                context=request.context or {},
                max_tokens=request.max_tokens * 2  # Allow some room for prompt template
            )
            if optimization_result.is_optimized:
                optimized.context = optimization_result.optimized_context
                optimization_strategies.append("context_optimization")
        
        # 2. Select optimal provider/model based on cost and performance
        if len(self.provider_calculators) > 1:
            best_provider, best_model = self._select_optimal_provider(optimized)
            if best_provider != optimized.provider_preference or \
               best_model != (optimized.model or "gpt-3.5-turbo"):
                optimized.provider_preference = best_provider
                optimized.model = best_model
                optimization_strategies.append(f"provider_selection:{best_provider}:{best_model}")
        
        # 3. Adjust generation parameters if needed
        if self._should_adjust_generation_params(optimized, max_cost):
            optimized = self._adjust_generation_params(optimized, max_cost)
            optimization_strategies.append("generation_parameter_optimization")
        
        # Calculate final cost and set result
        result.optimized_request = optimized
        result.final_estimated_cost = self.estimate_request_cost(optimized)
        result.optimization_strategy = ",".join(optimization_strategies) or "none"
        
        return result
        
    def _should_optimize_context(self, request: AIRequest) -> bool:
        """Determine if context should be optimized based on size.
        
        Args:
            request: The AI request to check.
            
        Returns:
            bool: True if context should be optimized.
        """
        # Simple implementation: Check if context exists and is large
        if not request.context:
            return False
            
        # Estimate token count of context (if serialized to JSON)
        context_json = json.dumps(request.context)
        context_tokens = self._count_tokens(context_json)
        
        # If context is more than 1000 tokens, optimize
        return context_tokens > 1000
    
    def _select_optimal_provider(self, request: AIRequest) -> Tuple[str, str]:
        """Select the optimal provider and model based on cost and performance.
        
        Args:
            request: The AI request to optimize.
            
        Returns:
            Tuple[str, str]: Optimal provider and model.
        """
        # For now, just use the existing provider or default to OpenAI
        provider = request.provider_preference or ProviderType.OPENAI
        model = request.model or "gpt-3.5-turbo"
        
        # In a real implementation, we would compare costs across providers
        # and select the most cost-effective option based on the request type
        
        return provider, model
    
    def _should_adjust_generation_params(self, request: AIRequest, max_cost: Optional[float] = None) -> bool:
        """Determine if generation parameters should be adjusted to reduce cost.
        
        Args:
            request: The AI request to check.
            max_cost: Maximum allowed cost.
            
        Returns:
            bool: True if generation parameters should be adjusted.
        """
        if max_cost is None:
            return False
            
        estimated_cost = self.estimate_request_cost(request)
        return estimated_cost > max_cost
    
    def _adjust_generation_params(self, request: AIRequest, max_cost: float) -> AIRequest:
        """Adjust generation parameters to meet cost constraints.
        
        Args:
            request: The AI request to optimize.
            max_cost: Maximum allowed cost.
            
        Returns:
            AIRequest: Optimized request with adjusted parameters.
        """
        optimized = request.copy(deep=True)
        
        # Reduce max_tokens if needed
        while self.estimate_request_cost(optimized) > max_cost and optimized.max_tokens > 100:
            optimized.max_tokens = int(optimized.max_tokens * 0.8)  # Reduce by 20%
            
        # If still too expensive, adjust other parameters
        if self.estimate_request_cost(optimized) > max_cost:
            # Simplify options like temperature, top_p, etc.
            # These would be in the request.parameters dictionary
            pass
            
        return optimized
        
        # If we still exceed max cost after optimizations, return the original
        if max_cost is not None and result.final_estimated_cost > max_cost:
            logger.warning(
                f"Could not optimize request below max cost of ${max_cost:.4f}. "
                f"Estimated cost: ${result.final_estimated_cost:.4f}"
            )
            result.optimized_request = request
            result.final_estimated_cost = result.baseline_cost
            result.optimization_strategy = "none"
        
        return result
    
    def _should_optimize_context(self, request: AIRequest) -> bool:
        """Determine if context optimization should be applied to this request.
        
        Args:
            request: The request to check.
            
        Returns:
            bool: True if context optimization should be applied.
        """
        if not request.context:
            return False
            
        # Check if context is large enough to benefit from optimization
        context_size = len(json.dumps(request.context))
        return context_size > 2048  # 2KB threshold
    
    def _select_optimal_provider(
        self, 
        request: AIRequest
    ) -> Tuple[str, str]:
        """Select the optimal provider and model for the given request.
        
        Args:
            request: The request to optimize.
            
        Returns:
            Tuple of (provider_name, model_name)
        """
        if not self.provider_calculators:
            return (request.provider_preference or "openai", 
                   request.model or "gpt-3.5-turbo")
        
        # Get token counts for the request
        input_tokens = self._count_tokens(
            request.prompt_template, 
            request.provider_preference or "openai"
        )
        
        # Evaluate each provider's cost for this request
        provider_costs = {}
        
        for provider_name, calculator in self.provider_calculators.items():
            # Get default model for this provider if none specified
            model = request.model or calculator.get_default_model()
            
            # Skip if model not supported by provider
            if not calculator.supports_model(model):
                continue
                
            # Estimate output tokens
            output_tokens = min(
                request.max_tokens,
                calculator.estimate_output_tokens(
                    model=model,
                    input_text=request.prompt_template,
                    max_tokens=request.max_tokens
                )
            )
            
            # Calculate cost
            cost = calculator.calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_streaming=request.stream
            )
            
            provider_costs[(provider_name, model)] = cost
        
        if not provider_costs:
            # Fallback to original provider/model if no valid options
            return (request.provider_preference or "openai", 
                   request.model or "gpt-3.5-turbo")
        
        # Return provider/model with lowest cost
        return min(provider_costs.items(), key=lambda x: x[1])[0]
    
    def _should_adjust_generation_params(
        self, 
        request: AIRequest, 
        max_cost: Optional[float] = None
    ) -> bool:
        """Determine if generation parameters should be adjusted.
        
        Args:
            request: The request to check.
            max_cost: Maximum allowed cost, if any.
            
        Returns:
            bool: True if generation parameters should be adjusted.
        """
        if max_cost is None:
            return False
            
        current_cost = self.estimate_request_cost(request)
        return current_cost > max_cost
    
    def _adjust_generation_params(
        self, 
        request: AIRequest, 
        max_cost: float
    ) -> AIRequest:
        """Adjust generation parameters to meet cost constraints.
        
        Args:
            request: The request to adjust.
            max_cost: Maximum allowed cost.
            
        Returns:
            AIRequest: The adjusted request.
        """
        adjusted = request.copy(deep=True)
        
        # 1. Reduce max tokens (most effective way to reduce cost)
        current_cost = self.estimate_request_cost(adjusted)
        if current_cost <= 0:
            return adjusted  # Avoid division by zero
            
        # Calculate target max tokens based on cost ratio
        cost_ratio = max_cost / current_cost
        adjusted.max_tokens = max(
            10,  # Minimum reasonable tokens
            int(adjusted.max_tokens * cost_ratio * 0.9)  # 90% of target to be safe
        )
        
        # 2. If still over budget, adjust temperature
        if self.estimate_request_cost(adjusted) > max_cost:
            adjusted.temperature = max(0.1, adjusted.temperature - 0.2)
        
        # 3. If still over budget, reduce top_p
        if self.estimate_request_cost(adjusted) > max_cost:
            adjusted.top_p = max(0.5, adjusted.top_p - 0.1)
        
        return adjusted
    
    def get_usage_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get a summary of recent usage statistics.
        
        Args:
            days: Number of days to include in the summary.
            
        Returns:
            Dict[str, Any]: Usage summary statistics.
        """
        # Get current time and calculate cutoff
        now = time.time()
        cutoff = now - (days * 24 * 3600)
        
        # Filter recent usage
        recent_usage = [u for u in self.usage_log if u.timestamp >= cutoff]
        
        if not recent_usage:
            return {
                "period_days": days,
                "total_cost": 0.0,
                "total_tokens": 0,
                "requests_count": 0,
                "by_provider": {},
                "by_request_type": {},
            }
        
        # Calculate statistics
        total_cost = sum(u.cost for u in recent_usage)
        total_tokens = sum(u.tokens for u in recent_usage)
        
        # Group by provider
        by_provider = {}
        for usage in recent_usage:
            provider = usage.provider
            if provider not in by_provider:
                by_provider[provider] = {"cost": 0.0, "tokens": 0, "count": 0}
            
            by_provider[provider]["cost"] += usage.cost
            by_provider[provider]["tokens"] += usage.tokens
            by_provider[provider]["count"] += 1
        
        # Group by request type
        by_type = {}
        for usage in recent_usage:
            req_type = usage.request_type
            if req_type not in by_type:
                by_type[req_type] = {"cost": 0.0, "tokens": 0, "count": 0}
            
            by_type[req_type]["cost"] += usage.cost
            by_type[req_type]["tokens"] += usage.tokens
            by_type[req_type]["count"] += 1
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "requests_count": len(recent_usage),
            "by_provider": by_provider,
            "by_request_type": by_type,
        }
    
    # Stub methods for future implementation
    
    def apply_context_compression(self, request: AIRequest) -> AIRequest:
        """Compress context to reduce token usage and cost.
        
        Args:
            request: The AI request with context to compress.
            
        Returns:
            AIRequest: Request with compressed context.
        """
        # FUTURE: Implement intelligent context compression for enhanced cost optimization
        logger.info("Context compression not yet implemented")
        return request
    
    def apply_prompt_simplification(self, request: AIRequest) -> AIRequest:
        """Simplify prompt to reduce token usage and cost.
        
        Args:
            request: The AI request with prompt to simplify.
            
        Returns:
            AIRequest: Request with simplified prompt.
        """
        # FUTURE: Add advanced prompt optimization algorithms for token efficiency
        logger.info("Prompt simplification not yet implemented")
        return request
    
    def manage_intelligent_caching(self, request: AIRequest) -> Optional[AIResponse]:
        """Check for semantically similar cached responses.
        
        Args:
            request: The AI request to check for in cache.
            
        Returns:
            Optional[AIResponse]: Cached response if available.
        """
        # FUTURE: Implement intelligent semantic caching for enhanced performance
        if not self.config.enable_caching or not self.cache:
            return None
            
        # Simple implementation - exact match on prompt
        cache_key = request.prompt
        if cache_key in self.cache:
            logger.info("Cache hit for request")
            cached_data = self.cache[cache_key]
            
            return AIResponse(
                request_id=request.request_id,
                content=cached_data.get("content", ""),
                model=request.model,
                provider=request.provider_preference,
                tokens=cached_data.get("tokens", 0),
                cost=0.0,  # No cost for cached responses
                status="success",
                metadata={"source": "cache"}
            )
            
        logger.info("Cache miss for request")
        return None
        
    def _create_context_optimizer(self):
        """Create a context optimizer instance.
        
        Returns:
            An object with an optimize method for context optimization.
        """
        # Simple implementation that provides the expected interface
        class SimpleContextOptimizer:
            def optimize(self, context: Dict[str, Any], max_tokens: int = 1000):
                class OptimizationResult:
                    def __init__(self):
                        self.is_optimized = False
                        self.optimized_context = context
                return OptimizationResult()
                
        return SimpleContextOptimizer()
        
    def _create_cache(self) -> Dict[str, Any]:
        """Create a cache for storing AI responses.
        
        Returns:
            Dict[str, Any]: Simple dictionary cache.
        """
        # In a full implementation, this would be a proper semantic cache
        # For now, just return a simple dictionary
        return {}
