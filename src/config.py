"""Configuration management for the Corporate Dynamics Simulator.

This module handles loading configuration from files and environment variables,
providing a unified interface for accessing configuration settings across the application.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel
from dotenv import load_dotenv


class SimulatorConfig(BaseModel):
    """Configuration for the Corporate Dynamics Simulator."""
    
    # AI service configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Cost and performance settings
    monthly_budget: float = 10.0  # Default $10 monthly budget
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    cache_similarity_threshold: float = 0.85
    
    # Game settings
    default_scenario: str = "ai_hype_cycle"
    max_turns: int = 10
    
    # Paths
    scenarios_dir: str = "src/scenarios"
    metrics_dir: Optional[str] = None
    
    # Feature flags
    enable_metrics_logging: bool = False
    enable_detailed_relationship_tracking: bool = True
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True


def load_config(config_file: Optional[str] = None) -> SimulatorConfig:
    """Load configuration from file and/or environment variables.
    
    This function ensures .env files are loaded BEFORE reading environment variables,
    making it work seamlessly for users who just add their API key to .env.
    
    Args:
        config_file: Optional path to config file, defaults to config.json in root directory
        
    Returns:
        SimulatorConfig: Configuration object
    """
    # 🔥 CRITICAL FIX: Load .env file first to ensure environment variables are available
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)  # Don't override existing env vars
        print(f"✅ Loaded .env file: {env_path}")
    
    config_data = {}
    
    # Override with environment variables if present
    env_vars = {
        "OPENAI_API_KEY": "openai_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key", 
        "GEMINI_API_KEY": "gemini_api_key",
        "GOOGLE_API_KEY": "gemini_api_key",  # Support both naming conventions
        "MONTHLY_BUDGET": "monthly_budget",
        "ENABLE_CACHING": "enable_caching",
        "DEFAULT_SCENARIO": "default_scenario",
        "ENABLE_METRICS_LOGGING": "enable_metrics_logging"
    }
    
    for env_var, config_key in env_vars.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Convert to appropriate type
            if config_key in ["enable_caching", "enable_metrics_logging"]:
                value = value.lower() in ["true", "1", "yes"]
            elif config_key == "monthly_budget":
                value = float(value)
            
            # For gemini_api_key, prefer GEMINI_API_KEY over GOOGLE_API_KEY
            if config_key == "gemini_api_key" and env_var == "GOOGLE_API_KEY":
                if "GEMINI_API_KEY" in os.environ:
                    continue  # Skip GOOGLE_API_KEY if GEMINI_API_KEY exists
                    
            config_data[config_key] = value
    
    # Debug output for verification
    api_keys_found = []
    if config_data.get("openai_api_key"):
        api_keys_found.append("OpenAI")
    if config_data.get("anthropic_api_key"):
        api_keys_found.append("Anthropic")
    if config_data.get("gemini_api_key"):
        api_keys_found.append("Gemini")
        
    if api_keys_found:
        print(f"✅ API keys loaded: {', '.join(api_keys_found)}")
    else:
        print("⚠️  No API keys found - will use template fallback")
    
    return SimulatorConfig(**config_data)