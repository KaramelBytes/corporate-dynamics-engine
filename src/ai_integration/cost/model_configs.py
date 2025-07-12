"""Model-specific configurations for AI providers and models.

This module contains configuration settings for various AI models, including
API endpoints, authentication methods, and model-specific parameters.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

from src.ai_integration.cost.provider_models import ModelType

# Free tier quota limits with priority for model selection
FREE_TIER_QUOTAS = {
    ModelType.GEMINI_20_FLASH: {"daily_limit": 1500, "priority": 1},
    ModelType.GEMINI_15_FLASH_002: {"daily_limit": 1500, "priority": 2}, 
    ModelType.GEMINI_15_FLASH: {"daily_limit": 1500, "priority": 3},
    ModelType.GEMINI_15_FLASH_8B: {"daily_limit": 50, "priority": 4},
    ModelType.TEMPLATE: {"daily_limit": -1, "priority": 999}  # Unlimited
}

# Model-specific configurations including context windows and capabilities
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI Models
    ModelType.GPT4O_MINI: {
        "max_context_window": 128000,
        "max_completion_tokens": 4096,
        "supports_tools": True,
        "supports_vision": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 50000,
            "requests_per_minute": 4,
        }
    },
    ModelType.GPT35_TURBO: {
        "max_context_window": 16000,
        "max_completion_tokens": 4096,
        "supports_tools": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 20000,
            "requests_per_minute": 3,
        }
    },
    
    # Anthropic Models
    ModelType.CLAUDE_35_HAIKU: {
        "max_context_window": 200000,
        "max_completion_tokens": 4096,
        "supports_tools": True,
        "supports_vision": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 100000,
            "requests_per_minute": 5,
        }
    },
    ModelType.CLAUDE_3_HAIKU: {
        "max_context_window": 100000,
        "max_completion_tokens": 4096,
        "supports_tools": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 20000,
            "requests_per_minute": 3,
        }
    },
    
    # Google Models
    ModelType.GEMINI_20_FLASH: {
        "max_context_window": 1000000,
        "max_completion_tokens": 8192,
        "supports_tools": True,
        "supports_vision": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 1000000,
            "requests_per_minute": 12,
        }
    },
    ModelType.GEMINI_15_FLASH_002: {
        "max_context_window": 1000000,
        "max_completion_tokens": 8192,
        "supports_tools": True,
        "supports_vision": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 750000,
            "requests_per_minute": 10,
        }
    },
    ModelType.GEMINI_15_FLASH: {
        "max_context_window": 1000000,
        "max_completion_tokens": 8192,
        "supports_tools": True,
        "supports_vision": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 750000,
            "requests_per_minute": 10,
        }
    },
    ModelType.GEMINI_15_FLASH_8B: {
        "max_context_window": 1000000,
        "max_completion_tokens": 8192,
        "supports_tools": True,
        "supports_vision": True,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 500000,
            "requests_per_minute": 10,
        }
    },
    ModelType.GEMINI_PRO: {
        "max_context_window": 32000,
        "max_completion_tokens": 2048,
        "supports_tools": False,
        "supports_vision": False,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 60000,
            "requests_per_minute": 6,
        }
    },
    
    # Groq Models
    ModelType.LLAMA_3_8B_GROQ: {
        "max_context_window": 8192,
        "max_completion_tokens": 4096,
        "supports_tools": False,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 1000000,  # Very generous
            "requests_per_minute": 20,
        }
    },
    ModelType.MIXTRAL_8X7B_GROQ: {
        "max_context_window": 32768,
        "max_completion_tokens": 8192,
        "supports_tools": False,
        "supports_streaming": True,
        "free_tier_limits": {
            "tokens_per_day": 800000,
            "requests_per_minute": 15,
        }
    },
}

# Provider-specific configurations including API endpoints and authentication
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "auth_method": "api_key",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer",
        "models": [ModelType.GPT4O_MINI, ModelType.GPT35_TURBO],
        "streaming_supported": True,
        "default_model": ModelType.GPT4O_MINI,
        "fallback_model": ModelType.GPT35_TURBO,
    },
    "anthropic": {
        "api_base": "https://api.anthropic.com/v1",
        "auth_method": "api_key",
        "auth_header": "x-api-key",
        "models": [ModelType.CLAUDE_35_HAIKU, ModelType.CLAUDE_3_HAIKU],
        "streaming_supported": True,
        "default_model": ModelType.CLAUDE_35_HAIKU,
        "fallback_model": ModelType.CLAUDE_3_HAIKU,
    },
    "google": {
        "auth_method": "api_key",
        "models": [
            ModelType.GEMINI_15_FLASH_8B, 
            ModelType.GEMINI_15_FLASH, 
            ModelType.GEMINI_PRO
        ],
        "streaming_supported": True,
        "default_model": ModelType.GEMINI_15_FLASH_8B,
        "fallback_model": ModelType.GEMINI_PRO,
    },
    "groq": {
        "api_base": "https://api.groq.com/openai/v1",
        "auth_method": "api_key",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer",
        "models": [ModelType.LLAMA_3_8B_GROQ, ModelType.MIXTRAL_8X7B_GROQ],
        "streaming_supported": True,
        "default_model": ModelType.LLAMA_3_8B_GROQ,
    },
}

def get_model_config(model_type: ModelType) -> Dict[str, Any]:
    """Get configuration for a specific model.
    
    Args:
        model_type: The model type to get configuration for
        
    Returns:
        Dict containing model configuration
        
    Raises:
        ValueError: If model_type is not found in MODEL_CONFIGS
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"No configuration found for model {model_type}")
    return MODEL_CONFIGS[model_type]

def get_provider_for_model(model_type: ModelType) -> Optional[str]:
    """Get the provider name for a specific model.
    
    Args:
        model_type: The model to find the provider for
        
    Returns:
        Provider name or None if not found
    """
    for provider, config in PROVIDER_CONFIGS.items():
        if model_type in config.get("models", []):
            return provider
    return None

def get_preferred_models(task_requirements: Dict[str, Any] = None) -> List[ModelType]:
    """Get a list of models in order of preference based on requirements and quotas.
    
    Args:
        task_requirements: Optional dict with required capabilities
        
    Returns:
        List of ModelType ordered by preference based on quotas and capabilities
    """
    # Default requirements if none specified
    if not task_requirements:
        task_requirements = {
            "supports_tools": False,
            "min_context_window": 8000,
            "supports_vision": False,
        }
    
    # Filter models based on requirements
    eligible_models = []
    for model, config in MODEL_CONFIGS.items():
        # Skip models that don't meet minimum requirements
        if (task_requirements.get("supports_tools", False) and 
                not config.get("supports_tools", False)):
            continue
            
        if (task_requirements.get("supports_vision", False) and 
                not config.get("supports_vision", False)):
            continue
            
        if (config.get("max_context_window", 0) < 
                task_requirements.get("min_context_window", 0)):
            continue
        
        eligible_models.append(model)
    
    # Sort eligible models by FREE_TIER_QUOTAS priority if available
    quota_prioritized = []
    non_prioritized = []
    
    for model in eligible_models:
        if model in FREE_TIER_QUOTAS:
            # Use tuple of (priority, daily_limit, model) for sorting
            quota_prioritized.append(
                (FREE_TIER_QUOTAS[model]["priority"], 
                 FREE_TIER_QUOTAS[model]["daily_limit"],
                 model)
            )
        else:
            # For models not in FREE_TIER_QUOTAS, use their capabilities for scoring
            score = (
                MODEL_CONFIGS[model].get("free_tier_limits", {}).get("tokens_per_day", 0) / 10000 +
                MODEL_CONFIGS[model].get("max_context_window", 0) / 10000
            )
            non_prioritized.append((999, 0, model, score))  # Lower priority than quota-defined models
    
    # Sort by priority (ascending), then by daily_limit (descending)
    quota_prioritized.sort(key=lambda x: (x[0], -x[1]))
    
    # Sort non-prioritized models by score
    non_prioritized.sort(key=lambda x: x[3], reverse=True)
    
    # Combine the lists (prioritized models first)
    result = [model for _, _, model in quota_prioritized]
    result.extend([model for _, _, model, _ in non_prioritized])
    
    # Return the prioritized model types
    return result
