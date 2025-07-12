"""Optimization components for AI integration.

This package provides tools for optimizing AI requests, including context
compression, token optimization, and other techniques to improve efficiency
and reduce costs.
"""

from .context_optimizer import (
    ContextOptimizer,
    ContextOptimizationResult,
)

__all__ = [
    'ContextOptimizer',
    'ContextOptimizationResult',
]
