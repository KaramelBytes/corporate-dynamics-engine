"""Usage tracking for AI model quotas.

This module provides tracking of daily usage for free-tier models to manage quotas
and enable intelligent model selection based on available quota.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Dict, List, Tuple

from src.ai_integration.cost.provider_models import ModelType
from src.ai_integration.cost.model_configs import FREE_TIER_QUOTAS
from src.ai_integration.usage_logger import UsageLogger
import logging

logger = logging.getLogger(__name__)


class MultiModelUsageTracker:
    """Tracks daily usage for each free-tier model to manage quotas.
    
    This class maintains a record of how many times each model has been used
    in the current day, automatically resetting counters at midnight. It provides
    methods to check quota availability and get lists of available models.
    """
    
    def __init__(self):
        """Initialize the usage tracker with empty counters."""
        self.daily_usage: Dict[ModelType, int] = {}  # {model_name: count}
        self.last_reset: date = date.today()
        
    def record_request(self, model_type: ModelType) -> None:
        """Record that a request was made to this model.
        
        Args:
            model_type: The model type that was used
        """
        self._check_daily_reset()
        
        # Update our internal counter
        if model_type not in self.daily_usage:
            self.daily_usage[model_type] = 0
        self.daily_usage[model_type] += 1
        
        # Also update the singleton UsageLogger to ensure metrics consistency
        # Get the string value of the model type to use as model_name
        model_name = model_type.value if hasattr(model_type, 'value') else str(model_type)
        
        try:
            # Create a basic usage record in the singleton UsageLogger
            # We don't have actual tokens here, but we'll ensure the request is counted
            usage_logger = UsageLogger()
            usage_logger.log_request(
                model_name=model_name,
                prompt_length=1,  # Placeholder
                response_length=1,  # Placeholder
                timestamp=datetime.now()
            )
            logger.debug(f"MultiModelUsageTracker synchronized with UsageLogger for {model_name}")
        except Exception as e:
            logger.error(f"Failed to sync with UsageLogger: {e}")
            # Continue execution even if this fails
        
    def get_usage_count(self, model_type: ModelType) -> int:
        """Get current daily usage for a model.
        
        Args:
            model_type: The model to check usage for
            
        Returns:
            Current number of requests made to this model today
        """
        self._check_daily_reset()
        return self.daily_usage.get(model_type, 0)
        
    def is_quota_available(self, model_type: ModelType) -> bool:
        """Check if model has remaining quota today.
        
        Args:
            model_type: The model to check quota for
            
        Returns:
            True if the model has remaining quota, False otherwise
        """
        if model_type == ModelType.TEMPLATE:
            return True  # Template is unlimited
            
        quota_info = FREE_TIER_QUOTAS.get(model_type)
        if not quota_info:
            return False
            
        daily_limit = quota_info["daily_limit"]
        current_usage = self.get_usage_count(model_type)
        
        # -1 indicates unlimited quota
        if daily_limit == -1:
            return True
            
        return current_usage < daily_limit
        
    def get_available_models(self) -> List[ModelType]:
        """Get list of models with available quota, sorted by priority.
        
        Returns:
            List of ModelType with available quota, ordered by priority
            (lower priority number = higher priority)
        """
        available: List[Tuple[ModelType, int]] = []
        for model_type, quota_info in FREE_TIER_QUOTAS.items():
            if self.is_quota_available(model_type):
                available.append((model_type, quota_info["priority"]))
        
        # Sort by priority (lower number = higher priority)
        available.sort(key=lambda x: x[1])
        return [model for model, _ in available]
        
    def _check_daily_reset(self) -> None:
        """Reset usage counters if it's a new day."""
        today = date.today()
        if today != self.last_reset:
            self.daily_usage = {}
            self.last_reset = today
