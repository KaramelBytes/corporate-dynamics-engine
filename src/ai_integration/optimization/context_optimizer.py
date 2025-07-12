"""Context optimization for reducing token usage while preserving meaning.

This module provides functionality to optimize and compress context data to reduce
API costs while maintaining the most relevant information for AI processing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from ..data_models import AIRequest, AIResponse


@dataclass
class ContextOptimizationResult:
    """Result of context optimization operation."""
    optimized_context: Dict[str, Any]
    original_token_count: int
    optimized_token_count: int
    compression_ratio: float
    
    @property
    def tokens_saved(self) -> int:
        """Calculate number of tokens saved by optimization."""
        return self.original_token_count - self.optimized_token_count


class ContextOptimizer:
    """Optimizes context data to reduce token usage while preserving meaning.
    
    This class provides methods to compress, simplify, and optimize context data
    before sending it to AI models, helping to reduce API costs.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        aggressive_compression: bool = False
    ):
        """Initialize the context optimizer.
        
        Args:
            max_context_tokens: Maximum number of tokens to allow in optimized context.
            aggressive_compression: Whether to use more aggressive compression techniques
                                  that might slightly reduce quality.
        """
        self.max_context_tokens = max_context_tokens
        self.aggressive_compression = aggressive_compression
        
        # Common words/phrases that can often be removed or shortened
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', "don't",
            'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
            'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
            'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
            "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
            "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
            'wouldn', "wouldn't"
        }
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        This is an approximation - actual tokenization depends on the model.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            int: Estimated number of tokens.
        """
        # Rough approximation: 1 token ~= 4 chars in English
        # This is a simplification - actual tokenization is more complex
        if not text:
            return 0
        return len(text) // 4 + 1
    
    def optimize_context(
        self,
        context: Dict[str, Any],
        important_keys: Optional[Set[str]] = None
    ) -> ContextOptimizationResult:
        """Optimize a context dictionary to reduce token count.
        
        Args:
            context: The context dictionary to optimize.
            important_keys: Set of keys that should be preserved exactly.
            
        Returns:
            ContextOptimizationResult: The optimization result with metrics.
        """
        if important_keys is None:
            important_keys = set()
            
        # Make a deep copy to avoid modifying the original
        optimized = self._deep_copy_dict(context)
        original_token_count = self.estimate_token_count(json.dumps(optimized))
        
        # Apply optimization passes
        optimized = self._remove_empty_values(optimized)
        optimized = self._simplify_structure(optimized, important_keys)
        
        if self.aggressive_compression:
            optimized = self._apply_aggressive_compression(optimized, important_keys)
        
        # If still too large, apply more aggressive truncation
        optimized_str = json.dumps(optimized)
        current_tokens = self.estimate_token_count(optimized_str)
        
        if current_tokens > self.max_context_tokens:
            optimized = self._truncate_to_token_limit(
                optimized,
                self.max_context_tokens,
                important_keys
            )
        
        optimized_token_count = self.estimate_token_count(json.dumps(optimized))
        compression_ratio = (
            (original_token_count - optimized_token_count) / original_token_count * 100
            if original_token_count > 0 else 0
        )
        
        return ContextOptimizationResult(
            optimized_context=optimized,
            original_token_count=original_token_count,
            optimized_token_count=optimized_token_count,
            compression_ratio=compression_ratio
        )
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        return json.loads(json.dumps(d))
    
    def _remove_empty_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove empty values from a dictionary."""
        if not isinstance(d, dict):
            return d
            
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested = self._remove_empty_values(v)
                if nested:  # Only add non-empty dicts
                    result[k] = nested
            elif isinstance(v, (list, tuple, set)):
                # Handle lists/tuples/sets
                cleaned = [self._remove_empty_values(i) for i in v if i not in (None, '', [], {}, ())]
                if cleaned:  # Only add non-empty sequences
                    result[k] = cleaned
            elif v not in (None, '', [], {}, ()):
                # Add non-empty, non-container values
                result[k] = v
                
        return result
    
    def _simplify_structure(
        self,
        data: Any,
        important_keys: Set[str],
        current_path: str = ''
    ) -> Any:
        """Simplify the structure of the data to reduce tokens."""
        if isinstance(data, dict):
            simplified = {}
            for k, v in data.items():
                full_path = f"{current_path}.{k}" if current_path else k
                if full_path in important_keys:
                    # Preserve important keys exactly
                    simplified[k] = v
                else:
                    simplified[k] = self._simplify_structure(v, important_keys, full_path)
            return simplified
            
        elif isinstance(data, (list, tuple, set)):
            # Only simplify lists/tuples that aren't marked as important
            if current_path in important_keys:
                return data
            
            # For lists of strings, try to join them if they're short
            if all(isinstance(i, str) for i in data):
                total_length = sum(len(str(i)) for i in data)
                if total_length < 100:  # Only join if the total is small
                    return ", ".join(str(i) for i in data)
            
            # Otherwise, process each item
            return [self._simplify_structure(i, important_keys, current_path) for i in data]
            
        elif isinstance(data, str) and current_path not in important_keys:
            # Simplify string values that aren't important
            return self._simplify_string(data)
            
        return data
    
    def _simplify_string(self, s: str) -> str:
        """Simplify a string to reduce token count."""
        if not s or not isinstance(s, str):
            return s
            
        # Remove extra whitespace
        s = ' '.join(s.split())
        
        # Remove common boilerplate
        s = re.sub(r'\s*[\r\n]+\s*', ' ', s)  # Newlines to spaces
        s = re.sub(r'\s+', ' ', s).strip()  # Multiple spaces to one
        
        # Remove common phrases that don't add much meaning
        if self.aggressive_compression:
            s = re.sub(r'\b(in|on|at|by|for|with|to|from|of|the|a|an|and|or|but|is|are|was|were|be|been|being)\b', ' ', s, flags=re.IGNORECASE)
            s = ' '.join(s.split())  # Clean up extra spaces
        
        return s
    
    def _apply_aggressive_compression(
        self,
        data: Any,
        important_keys: Set[str],
        current_path: str = ''
    ) -> Any:
        """Apply more aggressive compression techniques."""
        if isinstance(data, dict):
            compressed = {}
            for k, v in data.items():
                full_path = f"{current_path}.{k}" if current_path else k
                if full_path in important_keys:
                    compressed[k] = v
                else:
                    compressed[k] = self._apply_aggressive_compression(
                        v, important_keys, full_path
                    )
            return compressed
            
        elif isinstance(data, (list, tuple, set)):
            if current_path in important_keys:
                return data
                
            # For non-important lists, limit the number of items
            max_items = 3 if self.aggressive_compression else 5
            if len(data) > max_items:
                data = list(data)[:max_items]
                data.append(f"... and {len(data) - max_items} more items")
                
            return [
                self._apply_aggressive_compression(i, important_keys, current_path)
                for i in data
            ]
            
        elif isinstance(data, str) and current_path not in important_keys:
            # More aggressive string compression
            if len(data) > 200:
                return data[:150] + "..." + data[-50:] if len(data) > 200 else data
            return data
            
        return data
    
    def _truncate_to_token_limit(
        self,
        data: Dict[str, Any],
        max_tokens: int,
        important_keys: Set[str]
    ) -> Dict[str, Any]:
        """Recursively truncate data to stay within token limit."""
        # This is a simplified implementation - in practice, you'd want to be
        # more sophisticated about what to truncate
        result = {}
        remaining_tokens = max_tokens
        
        # Process important keys first
        for key in list(data.keys()):
            if key in important_keys:
                result[key] = data[key]
                remaining_tokens -= self.estimate_token_count(json.dumps({key: data[key]}))
                
        # Then process non-important keys if we have tokens left
        for key, value in data.items():
            if key in result:
                continue
                
            value_str = json.dumps({key: value})
            value_tokens = self.estimate_token_count(value_str)
            
            if remaining_tokens - value_tokens > 0:
                result[key] = value
                remaining_tokens -= value_tokens
            else:
                # Truncate this value if possible
                if isinstance(value, str):
                    # For strings, truncate to fit remaining tokens
                    max_chars = remaining_tokens * 4  # Rough estimate
                    result[key] = value[:max_chars] + "..."
                    remaining_tokens = 0
                # For other types, we'd need more sophisticated handling
                
                if remaining_tokens <= 0:
                    break
        
        return result
