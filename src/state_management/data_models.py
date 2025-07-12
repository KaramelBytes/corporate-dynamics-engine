"""
Core data models for corporate simulation.
Simple data structures that are easy to implement and test.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Action:
    """Represents a player action in the corporate simulation"""
    id: str
    description: str
    directly_affects: List[str]
    resource_cost: Dict[str, float]
    difficulty: float
    action_type: str
    establishes_pattern: bool = False
    # Optional explicit stakeholder trust deltas provided by scenario authors.
    # Key = stakeholder_id, Value = trust delta (−1.0 … +1.0). When present the
    # relationship engine will *override* heuristic calculations and apply these
    # values verbatim.
    relationship_deltas: Optional[Dict[str, float]] = None


@dataclass
class RelationshipDelta:
    """Represents a change in stakeholder relationship"""
    stakeholder_id: str
    delta_type: str  # 'direct', 'indirect', 'alliance'
    magnitude: float
    source_action: str
    reasoning: str = ""


@dataclass
class ConsequenceSet:
    """Container for all consequences of an action"""

    relationship_changes: Dict[str, RelationshipDelta]
    resource_changes: Dict[str, float]
    scenario_changes: Dict[str, float]
    behavior_changes: Dict[str, Any]
    meta_changes: Dict[str, Any]

    def __post_init__(self) -> None:  # noqa: D401
        # Initialize empty dicts if None or falsy were provided
        if not self.relationship_changes:
            self.relationship_changes = {}
        if not self.resource_changes:
            self.resource_changes = {}
        if not self.scenario_changes:
            self.scenario_changes = {}
        if not self.behavior_changes:
            self.behavior_changes = {}
        if not self.meta_changes:
            self.meta_changes = {}


@dataclass
class ValidationResult:
    """Result of state validation operations"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
