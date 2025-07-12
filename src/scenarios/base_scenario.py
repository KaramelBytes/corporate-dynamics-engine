"""Base classes and data models for scenario expansion infrastructure.

This module defines the building blocks required for Phase 1 of the scenario
expansion effort.

Key components
--------------
1. ``ScenarioType`` – Enum capturing high-level scenario categories.
2. ``ScenarioMetadata`` – Pydantic model containing scenario descriptors used
   by filtering, difficulty adjustment, and the composition engine.
3. ``UnlockCondition`` – Pydantic model expressing a *single* requirement that
   must be satisfied in ``GameState`` for a scenario to become available. The
   implementation purposefully stays generic; enforcement is handled by the
   scenario factory.
4. ``BaseScenario`` – Abstract base class that **all new scenarios must
   inherit** from. It provides a well-defined interface while delegating the
   concrete business logic to subclasses.

The design follows the workspace rules: full type hints, Google-style
docstrings, Pydantic for data modelling, and ``pathlib`` for any file system
interaction (not currently needed here).
"""
from __future__ import annotations

import abc
import enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from src.state_management.game_state import GameState
    from src.campaign.campaign_manager import CampaignState

from src.state_management.data_models import Action

__all__ = [
    "ScenarioType",
    "ScenarioMetadata",
    "UnlockCondition",
    "BaseScenario",
]


class ScenarioType(str, enum.Enum):
    """Enumeration of supported scenario categories."""

    CRISIS = "crisis"
    PROJECT = "project"
    POLITICS = "politics"
    INNOVATION = "innovation"
    TRANSFORMATION = "transformation"
    VENDOR_MANAGEMENT = "vendor_management"
    RESOURCE_ALLOCATION = "resource_allocation"


class ScenarioMetadata(BaseModel):
    """Metadata attached to every scenario.

    Attributes:
        scenario_id: **Globally unique** identifier.
        title: Human-readable title.
        description: Short marketing description shown to the player.
        scenario_type: Category of the scenario (``ScenarioType``).
        difficulty: Normalised difficulty from 0 (easiest) to 1 (hardest).
        themes: List of thematic tags (e.g. *budget*, *leadership*).
        stakeholders: Stakeholders directly involved in the scenario.
    """

    scenario_id: str = Field(..., min_length=3)
    title: str
    description: str
    scenario_type: ScenarioType
    difficulty: float = Field(..., ge=0.0, le=1.0)
    themes: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)

    @field_validator("themes", "stakeholders", mode="before")
    @classmethod
    def _non_empty(cls, values: List[str]) -> List[str]:  # noqa: D401
        """Ensure no empty strings are provided."""
        if isinstance(values, list):
            for value in values:
                if not value:
                    raise ValueError("Empty string is not allowed.")
        return values


class CampaignPhase(str, enum.Enum):
    """Represents different phases of a campaign."""
    TUTORIAL = "tutorial"
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    EXPERT = "expert"


class UnlockCondition(BaseModel):
    """Base class for scenario unlock conditions."""
    
    def is_met(self, game_state: 'GameState', campaign_state: Optional['CampaignState'] = None) -> bool:
        """Check if this condition is met."""
        raise NotImplementedError("Subclasses must implement is_met")


class RelationshipUnlock(UnlockCondition):
    """Unlock based on stakeholder relationship.
    
    Example::
        RelationshipUnlock(
            stakeholder_id="ceo",
            min_relationship=0.2,
        )
    """
    stakeholder_id: str
    min_relationship: float = Field(..., ge=-1.0, le=1.0)
    
    def is_met(self, game_state: 'GameState', campaign_state: Optional['CampaignState'] = None) -> bool:
        relationship = game_state.player_reputation.get(self.stakeholder_id, 0.0)
        return relationship >= self.min_relationship


class CampaignUnlock(UnlockCondition):
    """Advanced unlock conditions for campaign mode.
    
    Example::
        CampaignUnlock(
            required_completed_scenarios=["scenario_001", "scenario_002"],
            min_campaign_performance=0.7,
            required_campaign_phase=CampaignPhase.EARLY,
            max_failures_allowed=2
        )
    """
    required_completed_scenarios: List[str] = Field(default_factory=list)
    min_campaign_performance: float = 0.0
    required_campaign_phase: Optional[CampaignPhase] = None
    max_failures_allowed: int = 999
    
    def is_met(self, game_state: 'GameState', campaign_state: Optional['CampaignState'] = None) -> bool:
        if not campaign_state:
            return False
            
        # Check completed scenarios
        completed_ids = {s.scenario_id for s in campaign_state.completed_scenarios}
        if not all(sid in completed_ids for sid in self.required_completed_scenarios):
            return False
            
        # Check campaign performance
        if campaign_state.total_performance_score < self.min_campaign_performance:
            return False
            
        # Check campaign phase - only enforce if we're in an earlier phase than required
        if self.required_campaign_phase:
            current_phase_value = list(CampaignPhase).index(campaign_state.current_phase)
            required_phase_value = list(CampaignPhase).index(self.required_campaign_phase)
            if current_phase_value < required_phase_value:
                return False
            
        # Check failure count (scenarios with low performance)
        failures = sum(1 for s in campaign_state.completed_scenarios 
                      if s.performance_score < 0.35)  # Below 35% is a failure
        if failures > self.max_failures_allowed:
            return False
            
        return True

# Rebuild the model to handle forward references
CampaignUnlock.model_rebuild()


class BaseScenario(abc.ABC):
    """Abstract base class that all scenarios must extend."""

    #: Metadata describing the scenario – populated by subclasses.
    metadata: ScenarioMetadata
    
    #: List of unlock conditions that must all be met for the scenario to be available.
    #: Can include both basic and campaign-specific conditions.
    unlock_conditions: List[UnlockCondition]
    
    #: Campaign phase this scenario belongs to (for progression tracking)
    campaign_phase: CampaignPhase = CampaignPhase.TUTORIAL
    
    #: Whether this is a tutorial scenario (always available in tutorial mode)
    is_tutorial: bool = False
    
    def __init__(self, **kwargs):
        """Initialize the scenario with default unlock conditions if not provided."""
        super().__init__(**kwargs)
        if not hasattr(self, 'unlock_conditions') or self.unlock_conditions is None:
            self.unlock_conditions = []

    # ------------------------------------------------------------------
    # Mandatory interface – **must** be implemented by subclasses.
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_available_actions(self) -> List[Action]:
        """Return a list of actions currently available to the player."""

    # ------------------------------------------------------------------
    # Optional hooks – default implementation provided.
    # ------------------------------------------------------------------
    def get_scenario_context(self) -> Dict[str, Any]:
        """Return context useful for AI prompts, logging, or analytics."""
        return self.metadata.model_dump()

    # ------------------------------------------------------------------
    # Quality-of-life wrappers for common metadata access.
    # ------------------------------------------------------------------
    @property
    def scenario_id(self) -> str:  # noqa: D401
        """Unique identifier shortcut."""
        return self.metadata.scenario_id

    @property
    def title(self) -> str:  # noqa: D401
        """Human-readable title shortcut."""
        return self.metadata.title

    @property
    def description(self) -> str:  # noqa: D401
        """Scenario description shortcut."""
        return self.metadata.description
