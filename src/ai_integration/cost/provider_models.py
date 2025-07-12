"""Provider-specific cost models for different AI services.

This module contains cost calculation logic for various AI providers, including
pricing information and token counting utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Union, ClassVar

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Enumeration of current free-tier and cost-effective AI models."""
    # OpenAI - Free tier models (as of 2024)
    GPT4O_MINI = "gpt-4o-mini"  # Primary free tier model
    GPT35_TURBO = "gpt-3.5-turbo"  # Legacy free tier model
    
    # Anthropic - Free tier models
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"  # Latest free tier model
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"  # Legacy free tier model
    
    # Google - Free tier models (most generous free tiers)
    GEMINI_20_FLASH = "gemini-2.0-flash"  # 1,500/day (newest)
    GEMINI_15_FLASH = "gemini-1.5-flash"  # 1,500/day (proven)
    GEMINI_15_FLASH_002 = "gemini-1.5-flash-002"  # 1,500/day (stable)
    GEMINI_15_FLASH_8B = "gemini-1.5-flash-8b"  # 50/day
    GEMINI_PRO = "gemini-pro"  # Legacy free tier model
    
    # Groq - Very fast free inference
    LLAMA_3_8B_GROQ = "llama3-8b-8192"  # Fast free inference
    MIXTRAL_8X7B_GROQ = "mixtral-8x7b-32768"  # Larger context, free
    
    # Fallback/other
    TEMPLATE = "template"
    
    @property
    def provider(self) -> str:
        """Get the provider name for this model."""
        if self.value.startswith("gpt"):
            return "openai"
        elif self.value.startswith("claude"):
            return "anthropic"
        elif self.value.startswith("gemini"):
            return "google"
        return "other"


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""
    model: ModelType
    input_cost_per_1k: float  # Cost per 1K input tokens in USD
    output_cost_per_1k: float  # Cost per 1K output tokens in USD
    context_window: int  # Max tokens in context window
    
    # Optional fields for fine-tuned models
    training_cost_per_1k: Optional[float] = None
    base_model: Optional[ModelType] = None


class CostCalculator(BaseModel):
    """Base class for provider-specific cost calculators."""
    
    class Config:
        arbitrary_types_allowed = True
    
    def calculate_cost(
        self,
        model: Union[str, ModelType],
        input_tokens: int,
        output_tokens: int,
        is_streaming: bool = False
    ) -> float:
        """Calculate the cost for a given number of input and output tokens.
        
        Args:
            model: The model ID or ModelType enum
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            is_streaming: Whether this is a streaming request (may affect pricing)
            
        Returns:
            float: The cost in USD
        """
        model_type = self._get_model_type(model)
        pricing = self._get_pricing(model_type)
        
        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        
        # Apply any streaming surcharge if needed
        if is_streaming:
            output_cost *= self._get_streaming_surcharge(model_type)
            
        return input_cost + output_cost
    
    def estimate_output_tokens(
        self,
        model: Union[str, ModelType],
        input_text: str,
        max_tokens: int
    ) -> int:
        """Estimate the number of output tokens for a given input.
        
        Args:
            model: The model ID or ModelType enum
            input_text: The input text
            max_tokens: Maximum tokens requested in the output
            
        Returns:
            int: Estimated number of output tokens
        """
        model_type = self._get_model_type(model)
        return self._estimate_output_tokens_impl(model_type, input_text, max_tokens)
    
    def _get_model_type(self, model: Union[str, ModelType]) -> ModelType:
        """Convert model string to ModelType enum."""
        if isinstance(model, ModelType):
            return model
            
        model_lower = model.lower()
        for mt in ModelType:
            if mt.value.lower() in model_lower:
                return mt
                
        # Default to a common model if unknown
        return ModelType.GPT35_TURBO
    
    def _get_pricing(self, model_type: ModelType) -> ModelPricing:
        """Get pricing information for a model type."""
        raise NotImplementedError("Subclasses must implement _get_pricing")
    
    def _estimate_output_tokens_impl(
        self,
        model_type: ModelType,
        input_text: str,
        max_tokens: int
    ) -> int:
        """Implementation of output token estimation."""
        raise NotImplementedError("Subclasses must implement _estimate_output_tokens_impl")
    
    def _get_streaming_surcharge(self, model_type: ModelType) -> float:
        """Get any additional surcharge for streaming requests."""
        # Default: no surcharge for streaming
        return 1.0


class OpenAICostCalculator(CostCalculator):
    """Cost calculator for OpenAI models."""
    
    # Pricing as of 2024 (in USD per 1K tokens)
    PRICING: ClassVar[Dict[ModelType, ModelPricing]] = {
        ModelType.GPT4O_MINI: ModelPricing(
            model=ModelType.GPT4O_MINI,
            input_cost_per_1k=0.0005,  # $0.0005 per 1K input tokens
            output_cost_per_1k=0.0015,  # $0.0015 per 1K output tokens
            context_window=128000
        ),
        ModelType.GPT35_TURBO: ModelPricing(
            model=ModelType.GPT35_TURBO,
            input_cost_per_1k=0.0005,  # $0.0005 per 1K input tokens
            output_cost_per_1k=0.0015,  # $0.0015 per 1K output tokens
            context_window=16385
        ),
    }
    
    def _get_pricing(self, model_type: ModelType) -> ModelPricing:
        """Get pricing for an OpenAI model."""
        if model_type not in self.PRICING:
            # Default to GPT-3.5 Turbo pricing for unknown models
            return self.PRICING[ModelType.GPT35_TURBO]
        return self.PRICING[model_type]
    
    def _estimate_output_tokens_impl(
        self,
        model_type: ModelType,
        input_text: str,
        max_tokens: int
    ) -> int:
        """Estimate output tokens for OpenAI models."""
        # Simple estimation: output is typically 20-50% of input length for most tasks
        # but won't exceed max_tokens
        input_len = len(input_text.split())  # Rough word count
        estimated = min(int(input_len * 0.3), max_tokens)
        return max(estimated, 1)  # At least 1 token


class AnthropicCostCalculator(CostCalculator):
    """Cost calculator for Anthropic models."""
    
    # Pricing as of 2024 for free-tier models
    PRICING: ClassVar[Dict[ModelType, ModelPricing]] = {
        ModelType.CLAUDE_35_HAIKU: ModelPricing(
            model=ModelType.CLAUDE_35_HAIKU,
            input_cost_per_1k=0.0001,  # $0.0001 per 1K input tokens (free tier)
            output_cost_per_1k=0.0005,  # $0.0005 per 1K output tokens (free tier)
            context_window=200000
        ),
        ModelType.CLAUDE_3_HAIKU: ModelPricing(
            model=ModelType.CLAUDE_3_HAIKU,
            input_cost_per_1k=0.00025,  # $0.00025 per 1K input tokens
            output_cost_per_1k=0.00125,  # $0.00125 per 1K output tokens
            context_window=200000
        ),
    }
    
    def _get_pricing(self, model_type: ModelType) -> ModelPricing:
        """Get pricing for an Anthropic model."""
        if model_type not in self.PRICING:
            # Default to Haiku pricing for unknown models
            return self.PRICING[ModelType.CLAUDE_3_HAIKU]
        return self.PRICING[model_type]
    
    def _estimate_output_tokens_impl(
        self,
        model_type: ModelType,
        input_text: str,
        max_tokens: int
    ) -> int:
        """Estimate output tokens for Anthropic models."""
        # Anthropic models tend to be more verbose, so estimate higher output ratio
        input_len = len(input_text.split())  # Rough word count
        estimated = min(int(input_len * 0.4), max_tokens)
        return max(estimated, 1)  # At least 1 token


class GoogleCostCalculator(CostCalculator):
    """Cost calculator for Google models."""
    
    # Pricing as of 2024 (in USD per 1K tokens)
    PRICING: ClassVar[Dict[ModelType, ModelPricing]] = {
        ModelType.GEMINI_PRO: ModelPricing(
            model=ModelType.GEMINI_PRO,
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.0005,
            context_window=30720
        ),
    }
    
    def _get_pricing(self, model_type: ModelType) -> ModelPricing:
        """Get pricing for a Google model."""
        if model_type not in self.PRICING:
            # Default to Gemini Pro pricing for unknown models
            return self.PRICING[ModelType.GEMINI_PRO]
        return self.PRICING[model_type]
    
    def _estimate_output_tokens_impl(
        self,
        model_type: ModelType,
        input_text: str,
        max_tokens: int
    ) -> int:
        """Estimate output tokens for Google models."""
        # Google models tend to be concise, so estimate lower output ratio
        input_len = len(input_text.split())  # Rough word count
        estimated = min(int(input_len * 0.25), max_tokens)
        return max(estimated, 1)  # At least 1 token
