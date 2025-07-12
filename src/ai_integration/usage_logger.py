"""
A logger for tracking AI API usage with persistent storage.
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class UsageLogger:
    """A logger for tracking AI API usage with persistent storage.
    
    Implements the singleton pattern to ensure data persistence across application.
    Usage data is automatically persisted to disk as a JSON file and loaded on initialization.
    """
    
    # Singleton instance
    _instance = None
    
    # Default path for storing usage data
    USAGE_DATA_PATH = Path(os.path.expanduser("~")) / ".corporate_simulator" / "usage_data.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UsageLogger, cls).__new__(cls)
            # Initialize attributes
            cls._instance.usage_records = []
            cls._instance.daily_counts = {}
            
            # Load persisted data if available
            cls._instance.load_from_disk()
        return cls._instance

    def __init__(self) -> None:
        """Initializes the UsageLogger.
        
        Note: Actual initialization is handled in __new__ to ensure
        the singleton pattern works correctly.
        """
        # All initialization is done in __new__
        pass

    def load_from_disk(self) -> None:
        """
        Load usage data from disk if available.
        """
        try:
            # Create directory if it doesn't exist
            self.USAGE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.USAGE_DATA_PATH.exists():
                logger.info(f"No usage data file found at {self.USAGE_DATA_PATH}")
                return
                
            with open(self.USAGE_DATA_PATH, 'r') as f:
                data = json.load(f)
                
            # Convert stored records back to proper format
            # Timestamps need special handling as they're stored as strings
            self.usage_records = []
            for record in data.get('usage_records', []):
                if 'timestamp' in record and isinstance(record['timestamp'], str):
                    record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                self.usage_records.append(record)
                
            # Daily counts are stored with tuple keys as strings like "2023-07-06|gemini-1.5-flash"
            # We need to convert them back to tuple keys
            self.daily_counts = {}
            for key_str, value in data.get('daily_counts', {}).items():
                date_str, model_name = key_str.split('|')
                self.daily_counts[(date_str, model_name)] = value
                
            logger.info(f"Loaded {len(self.usage_records)} usage records from {self.USAGE_DATA_PATH}")
                
        except Exception as e:
            logger.error(f"Error loading usage data: {e}", exc_info=True)
            # Initialize with empty data if loading fails
            self.usage_records = []
            self.daily_counts = {}
    
    def save_to_disk(self) -> None:
        """
        Save usage data to disk.
        """
        try:
            # Create directory if it doesn't exist
            self.USAGE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert records for JSON serialization
            serializable_records = []
            for record in self.usage_records:
                # Create a copy to avoid modifying the original
                record_copy = dict(record)
                # Convert datetime to string if present
                if 'timestamp' in record_copy and isinstance(record_copy['timestamp'], datetime):
                    record_copy['timestamp'] = record_copy['timestamp'].isoformat()
                serializable_records.append(record_copy)
                
            # Convert tuple keys in daily_counts to strings
            serializable_daily_counts = {}
            for (date_str, model_name), count in self.daily_counts.items():
                key_str = f"{date_str}|{model_name}"
                serializable_daily_counts[key_str] = count
                
            # Prepare data for serialization
            data = {
                'usage_records': serializable_records,
                'daily_counts': serializable_daily_counts
            }
            
            # Write to disk
            with open(self.USAGE_DATA_PATH, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self.usage_records)} usage records to {self.USAGE_DATA_PATH}")
                
        except Exception as e:
            logger.error(f"Error saving usage data: {e}", exc_info=True)
    
    def log_request(
        self,
        model_name: str,
        prompt_length: int,
        response_length: int,
        timestamp: datetime,
    ) -> None:
        """
        Logs a single API request and updates daily counts.

        Args:
            model_name: The name of the model used.
            prompt_length: The length of the prompt (e.g., token count).
            response_length: The length of the response (e.g., token count).
            timestamp: The timestamp of the request.
        """
        # Simple cost estimation placeholder.
        # In a real system, this would use a proper pricing model.
        estimated_cost = (prompt_length * 0.00001) + (response_length * 0.00003)

        record = {
            "model_name": model_name,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "timestamp": timestamp,
            "estimated_cost": estimated_cost,
        }
        self.usage_records.append(record)

        date_str = timestamp.strftime('%Y-%m-%d')
        key = (date_str, model_name)
        self.daily_counts[key] = self.daily_counts.get(key, 0) + 1
        
        # Save to disk after each update
        self.save_to_disk()

    def get_daily_count(self, model_name: str, date_str: str) -> int:
        """
        Gets the number of requests for a given model on a specific date.

        Args:
            model_name: The name of the model.
            date_str: The date in 'YYYY-MM-DD' format.

        Returns:
            The number of requests for the given model and date.
        """
        return self.daily_counts.get((date_str, model_name), 0)

    def get_total_stats(self) -> Dict[str, float | int]:
        """
        Calculates total usage statistics across all logged records.

        Returns:
            A dictionary with total requests, total tokens, and estimated cost.
        """
        total_requests = len(self.usage_records)
        if not total_requests:
            return {"total_requests": 0, "total_tokens": 0, "estimated_cost": 0.0}

        total_tokens = sum(
            r["prompt_length"] + r["response_length"] for r in self.usage_records
        )
        total_cost = sum(r["estimated_cost"] for r in self.usage_records)

        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
        }

    def get_quota_status(self, model_name: str, daily_limit: int = 1500) -> str:
        """
        Determines the quota status for a model based on its daily usage.

        Args:
            model_name: The name of the model.
            daily_limit: The daily request limit for the model.

        Returns:
            A status string: 'OK', 'WARNING', or 'CRITICAL'.
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        daily_count = self.get_daily_count(model_name, date_str)

        if daily_limit <= 0:
            return "OK"

        usage_percentage = (daily_count / daily_limit) * 100

        if usage_percentage > 90:
            return "CRITICAL"
        elif usage_percentage >= 70:
            return "WARNING"
        else:
            return "OK"
