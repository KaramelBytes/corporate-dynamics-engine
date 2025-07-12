"""Semantic caching system for AI responses with similarity-based lookup.

This module provides a caching layer that can find and return similar cached responses
based on semantic similarity of requests, reducing the need for redundant API calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..data_models import AIRequest, AIResponse


class CachedResponse(BaseModel):
    """Represents a cached AI response with metadata for similarity matching."""
    
    request_hash: str
    request_embedding: Optional[List[float]] = None
    request_key_elements: Dict[str, Any] = Field(default_factory=dict)
    response: AIResponse
    timestamp: float = Field(default_factory=time.time)
    ttl_seconds: int = 3600  # Default 1-hour cache lifetime
    
    @property
    def is_expired(self) -> bool:
        """Check if the cached response has expired."""
        return (time.time() - self.timestamp) > self.ttl_seconds


class SemanticCache:
    """Implements semantic caching for AI responses with similarity-based lookup.
    
    This cache stores AI responses and can retrieve them based on semantic similarity
    of the request, not just exact matches. This helps reduce API calls for similar requests.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize the semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider requests
                                similar enough to use a cached response.
        """
        self.cache: Dict[str, CachedResponse] = {}
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = 384  # Default embedding dimension
        
        # Simple in-memory index for similarity search
        self.embedding_index: List[Tuple[str, List[float]]] = []
    
    def _generate_key(self, request: AIRequest) -> str:
        """Generate a deterministic cache key from request data.
        
        Args:
            request: The AI request to generate a key for.
            
        Returns:
            str: A unique hash key for the request.
        """
        # Create a stable string representation of the request
        request_data = {
            "content_type": request.content_type,
            "prompt_template": request.prompt_template,
            "max_tokens": request.max_tokens,
            "provider_preference": request.provider_preference,
            "response_format": request.response_format,
            "temperature": request.temperature,
            # Include important context keys that affect the response
            "context_keys": sorted(list(request.context.keys())) if request.context else []
        }
        
        # Convert to JSON string and hash it
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a semantic embedding for the given text.
        
        In a production environment, this would use a pre-trained model like
        Sentence-BERT or OpenAI's text-embedding-ada-002.
        
        Args:
            text: The text to generate an embedding for.
            
        Returns:
            List[float]: A vector embedding of the text.
        """
        # FUTURE: Replace with production-grade embedding model for semantic similarity
        # This is a simple placeholder that returns random embeddings
        return list(np.random.rand(self.embedding_dim).astype(np.float32))
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector.
            b: Second vector.
            
        Returns:
            float: Cosine similarity between the vectors (0-1).
        """
        if not a or not b or len(a) != len(b):
            return 0.0
            
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        return float(np.dot(a, b) / (a_norm * b_norm))
    
    def get_similar(self, request: AIRequest) -> Optional[AIResponse]:
        """Find a similar cached response if it exists.
        
        Args:
            request: The AI request to find a similar cached response for.
            
        Returns:
            Optional[AIResponse]: A cached response if a similar one is found, else None.
        """
        # First try exact match
        cache_key = self._generate_key(request)
        if cache_key in self.cache and not self.cache[cache_key].is_expired:
            return self.cache[cache_key].response
        
        # Then try semantic similarity if we have embeddings
        if self.embedding_index:
            # Generate embedding for the request
            request_embedding = self._generate_embedding(
                f"{request.prompt_template} {json.dumps(request.context) if request.context else ''}"
            )
            
            # Find most similar cached request
            best_similarity = 0.0
            best_match = None
            
            for cached_key, cached_embedding in self.embedding_index:
                if cached_key in self.cache and not self.cache[cached_key].is_expired:
                    similarity = self._cosine_similarity(request_embedding, cached_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_key
            
            if best_similarity >= self.similarity_threshold and best_match in self.cache:
                logger.info(f"Semantic cache hit with similarity {best_similarity:.2f}")
                return self.cache[best_match].response
        
        return None
    
    def store(self, request: AIRequest, response: AIResponse) -> None:
        """Store a response in the cache.
        
        Args:
            request: The AI request that generated the response.
            response: The AI response to cache.
        """
        cache_key = self._generate_key(request)
        
        # Generate embedding for semantic search
        request_embedding = self._generate_embedding(
            f"{request.prompt_template} {json.dumps(request.context) if request.context else ''}"
        )
        
        # Create and store the cached response
        cached = CachedResponse(
            request_hash=cache_key,
            request_embedding=request_embedding,
            request_key_elements={
                "content_type": request.content_type,
                "provider_preference": request.provider_preference,
                "response_format": request.response_format,
            },
            response=response,
            ttl_seconds=3600  # 1 hour TTL
        )
        
        self.cache[cache_key] = cached
        self.embedding_index.append((cache_key, request_embedding))
        
        logger.debug(f"Cached response for request: {cache_key}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            int: Number of entries removed.
        """
        initial_count = len(self.cache)
        
        # Remove expired entries
        expired_keys = [k for k, v in self.cache.items() if v.is_expired]
        for key in expired_keys:
            del self.cache[key]
        
        # Update embedding index
        self.embedding_index = [
            (k, emb) for k, emb in self.embedding_index 
            if k in self.cache and not self.cache[k].is_expired
        ]
        
        removed = initial_count - len(self.cache)
        if removed > 0:
            logger.info(f"Cleaned up {removed} expired cache entries")
            
        return removed


# Module-level logger
logger = logging.getLogger(__name__)
