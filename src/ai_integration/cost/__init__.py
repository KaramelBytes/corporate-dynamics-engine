"""Cost calculation and optimization for AI providers.

This package provides tools for calculating and optimizing costs across different
AI providers, including token counting, cost estimation, and budget management.
"""

# Import cost models and calculators
from .provider_models import (
    ModelType,
    ModelPricing,
    CostCalculator,
    OpenAICostCalculator,
    AnthropicCostCalculator,
    GoogleCostCalculator,
)

# Import usage tracking
from .usage_tracker import MultiModelUsageTracker

# Default calculator instances
openai_calculator = OpenAICostCalculator()
anthropic_calculator = AnthropicCostCalculator()
google_calculator = GoogleCostCalculator()

# Provider to calculator mapping
PROVIDER_CALCULATORS = {
    "openai": openai_calculator,
    "anthropic": anthropic_calculator,
    "google": google_calculator,
}

def get_calculator(provider: str) -> CostCalculator:
    """Get the cost calculator for a specific provider.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic', 'google')
        
    Returns:
        CostCalculator: The appropriate calculator instance
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    if provider not in PROVIDER_CALCULATORS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_CALCULATORS[provider]

__all__ = [
    'ModelType',
    'ModelPricing',
    'CostCalculator',
    'OpenAICostCalculator',
    'AnthropicCostCalculator',
    'GoogleCostCalculator',
    'get_calculator',
    'MultiModelUsageTracker',
]
