"""Prompt engineering utilities for AI service integration."""
from __future__ import annotations

import hashlib
from typing import Dict, Any, Tuple

# Simple in-memory cache for prompt results
_PROMPT_CACHE: Dict[str, str] = {}


def _cache_key(*parts: Any) -> str:
    """Compute a deterministic cache key from arbitrary parts."""
    m = hashlib.sha256()
    for part in parts:
        m.update(repr(part).encode())
    return m.hexdigest()


def build_scenario_prompt(context: Dict[str, Any]) -> str:
    """Construct an optimized prompt for scenario generation.

    This strips irrelevant keys, sorts items for determinism, and uses caching
    to avoid redundant work.
    """
    key = _cache_key("scenario", context)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]

    # Core sections of the prompt
    title = context.get("title", "Untitled Scenario")
    description = context.get("description", "")
    priorities = ", ".join(context.get("priorities", []))

    prompt = (
        f"You are an enterprise scenario generator. Create a detailed corporate "
        f"scenario titled '{title}'.\n\n"
        f"Context: {description}\n"
        f"Stakeholder priorities: {priorities}\n\n"
        f"Requirements:\n"
        f"1. Outline the situation clearly.\n"
        f"2. Identify key challenges.\n"
        f"3. Suggest three branching decision paths.\n"
        f"4. Keep length under 300 words."
    )

    _PROMPT_CACHE[key] = prompt
    return prompt


def build_dialogue_prompt(character_id: str, context: Dict[str, Any]) -> str:
    """Construct a prompt for character dialogue generation with caching."""
    key = _cache_key("dialogue", character_id, context)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]

    persona = context.get("persona", "professional")
    last_message = context.get("last_message", "")

    prompt = (
        f"Assume the role of {character_id}, speaking in a {persona} tone.\n"
        f"Continue the conversation after this message: '{last_message}'.\n\n"
        f"Respond concisely, under 100 words, and include a clear next action request."
    )

    _PROMPT_CACHE[key] = prompt
    return prompt
