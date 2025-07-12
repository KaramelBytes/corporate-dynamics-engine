"""Cost management utilities focused on free-tier protection for AI calls."""
from __future__ import annotations

import datetime as _dt
from typing import Dict, Any

COST_THRESHOLDS: Dict[str, Any] = {
    "free_tier_limits": {
        "max_requests_per_hour": 60,  # Increased for free-tier models
        "max_tokens_per_request": 8000,  # Increased for modern context sizes
        "max_monthly_requests": 1000,  # Increased for modern free tier limits
        "emergency_brake_requests": 100,  # Increased for safety
    },
    "provider_preferences": [
        # Free tier models by priority
        "gemini-1.5-flash-8b",  # Most generous free tier
        "gemini-1.5-flash",  # High capability/limit free model
        "claude-3-5-haiku-20241022",  # Latest Anthropic free model
        "gpt-4o-mini",  # OpenAI free tier
        "claude-3-haiku-20240307",  # Legacy free model
        "gpt-3.5-turbo",  # Legacy free model
        "gemini-pro",  # Legacy free model
        "template_fallback",  # Last resort fallback
    ],
    "cost_optimization": {
        "cache_similar_requests": True,
        "compress_prompts": True,
        "fallback_on_quota_exceeded": True,
    },
    # New: Model-specific rate limits based on free tier allowances
    "model_rate_limits": {
        "gemini-1.5-flash-8b": {"requests_per_minute": 10, "tokens_per_day": 500000},
        "gemini-1.5-flash": {"requests_per_minute": 8, "tokens_per_day": 300000},
        "claude-3-5-haiku-20241022": {"requests_per_minute": 5, "tokens_per_day": 100000},
        "gpt-4o-mini": {"requests_per_minute": 4, "tokens_per_day": 50000},
        "llama3-8b-8192": {"requests_per_minute": 20, "tokens_per_day": 1000000},  # Very generous
    },
}


class CostManager:  # noqa: D101, pylint: disable=too-few-public-methods
    def __init__(self, thresholds: Dict[str, Any] | None = None):
        self.rules = thresholds or COST_THRESHOLDS

        # Request counters
        self._session_requests = 0
        self._hourly_requests: Dict[int, int] = {}
        self._monthly_requests = 0

        # Cache: prompt_hash -> response
        self._response_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def can_make_request(self, tokens: int) -> bool:  # noqa: D401
        """Return True if a new request is allowed under free-tier rules."""
        limits = self.rules["free_tier_limits"]

        # Emergency brake (per session)
        if self._session_requests >= limits["emergency_brake_requests"]:
            return False

        # Token limit per request
        if tokens > limits["max_tokens_per_request"]:
            return False

        now = _dt.datetime.utcnow()
        month_key = now.year * 100 + now.month
        hour_key = now.replace(minute=0, second=0, microsecond=0).timestamp()

        # Update counters initialisation
        self._hourly_requests.setdefault(hour_key, 0)

        # Check hourly and monthly limits
        if self._hourly_requests[hour_key] >= limits["max_requests_per_hour"]:
            return False
        if self._monthly_requests >= limits["max_monthly_requests"]:
            return False

        return True

    def register_request(self, tokens: int) -> None:
        """Record a request usage. Must be called only after can_make_request=True."""
        now = _dt.datetime.utcnow()
        hour_key = now.replace(minute=0, second=0, microsecond=0).timestamp()

        self._session_requests += 1
        self._hourly_requests[hour_key] = self._hourly_requests.get(hour_key, 0) + 1
        self._monthly_requests += 1

    # ------------------------------------------------------------------
    # Provider selection / fallback
    # ------------------------------------------------------------------
    def select_provider(self) -> str:
        """Return preferred provider respecting cost rules and current quotas."""
        return self.rules["provider_preferences"][0]

    def fallback_provider(self) -> str:
        """Return fallback provider when quotas exceeded or failures occur."""
        prefs = self.rules["provider_preferences"]
        return prefs[-1] if len(prefs) > 1 else "template_fallback"

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------
    def get_cached_response(self, prompt_hash: str) -> Any | None:
        return self._response_cache.get(prompt_hash)

    def cache_response(self, prompt_hash: str, response: Any) -> None:
        if self.rules["cost_optimization"]["cache_similar_requests"]:
            self._response_cache[prompt_hash] = response
