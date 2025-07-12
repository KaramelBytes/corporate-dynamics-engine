"""Caching components for AI responses and semantic similarity.

This package provides caching mechanisms for AI responses, including semantic
caching that can find and return similar cached responses based on the semantic
similarity of requests.
"""

from .semantic_cache import SemanticCache, CachedResponse

__all__ = [
    'SemanticCache',
    'CachedResponse',
]
