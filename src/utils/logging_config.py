"""Professional logging configuration for Corporate Dynamics Simulator."""
import logging
import os
from typing import Dict, Any

class LoggingMode:
    SILENT = "silent"
    PRODUCTION = "production" 
    DEMO = "demo"
    DEVELOPMENT = "development"

def setup_logging(mode: str = LoggingMode.DEMO) -> None:
    """Setup logging configuration for the application."""
    if mode == LoggingMode.SILENT:
        logging.basicConfig(level=logging.CRITICAL, format="%(message)s")
    elif mode == LoggingMode.PRODUCTION:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    elif mode == LoggingMode.DEMO:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        # Suppress verbose loggers
        for logger_name in [
            'src.utils.env_loader',
            'src.ai_integration.provider_interfaces',
            'src.ai_integration.service_orchestrator',
            'src.ai_integration.provider_health_monitor',
            'src.ai_integration.metrics_collector',
            'src.ai_integration.corporate_game_ai_integration',
            'src.campaign.campaign_manager'  # Add campaign manager to suppressed loggers
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    else:  # DEVELOPMENT
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
    # Always set markdown_it logger to WARNING level to reduce noise
    # regardless of the overall logging mode
    logging.getLogger('markdown_it').setLevel(logging.WARNING)

def get_logging_mode_from_env() -> str:
    """Get logging mode from environment, loading .env.logging if present."""
    # Try to load .env.logging file first
    from pathlib import Path
    
    env_logging_path = Path(".env.logging")
    if env_logging_path.exists():
        try:
            with open(env_logging_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"Warning: Could not load .env.logging: {e}")
    
    return os.getenv("CSIM_LOG_MODE", LoggingMode.DEMO).lower()
