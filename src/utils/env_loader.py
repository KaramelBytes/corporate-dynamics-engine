"""
Environment variable loading utility for Corporate Dynamics Simulator.
Provides secure loading of API keys and other configuration from environment variables.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_environment_variables(env_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load environment variables from .env file and OS environment.
    
    Prioritizes variables in the following order:
    1. OS environment variables (highest priority)
    2. .env file variables
    
    Args:
        env_path: Optional path to .env file. If None, looks in current directory.
        
    Returns:
        Dict of loaded environment variables
    """
    # Default to .env in project root if not specified
    if env_path is None:
        env_path = Path.cwd() / ".env"
        
    # Load from .env file if it exists
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logger.info("No .env file found, using system environment variables only")
    
    # Collect relevant environment variables
    api_keys = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
    }
    
    # Log which keys were found (without showing actual keys)
    for key, value in api_keys.items():
        if value:
            logger.info(f"Found {key.upper()} in environment variables")
        else:
            logger.info(f"{key.upper()} not found in environment variables")
            
    return api_keys
