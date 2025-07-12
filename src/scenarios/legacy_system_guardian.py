"""Legacy System Guardian – Scenario #6

Implements *The Legacy System Guardian* specification from
`docs/scenario_06_legacy_system_guardian.md`.
"""
from __future__ import annotations

from typing import List

from src.state_management.data_models import Action

from .base_scenario import (
    BaseScenario,
    CampaignPhase,
    ScenarioMetadata,
    ScenarioType,
    RelationshipUnlock,
    CampaignUnlock
)
from .scenario_factory import register_scenario


@register_scenario
class LegacySystemGuardian(BaseScenario):
    """Scenario addressing modernization vs. institutional knowledge tension."""
    # Late-phase scenario
    campaign_phase = CampaignPhase.LATE
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="legacy_system_guardian",
        title="Legacy System Guardian",
        description=(
            "A board-mandated cloud migration threatens the legacy system your senior "
            "developer has maintained for 15 years. Balance modernization with the value "
            "of institutional knowledge and system stability."
        ),
        scenario_type=ScenarioType.TRANSFORMATION,
        difficulty=0.8,
        themes=["modernization", "institutional_knowledge", "cloud_migration"],
        stakeholders=[
            "ceo",
            "it_director",
            "sales_director",
            "it_team",
            "senior_developer",
            "cfo",
        ],
    )

    def __init__(self) -> None:
        # First initialize the parent class
        super().__init__()
        
        # Initialize unlock conditions with dynamic thresholds
        self.unlock_conditions = [
            # Option 1: Complete 6+ scenarios with good performance
            CampaignUnlock(
                required_completed_scenarios=[],  # Any 6 scenarios
                min_campaign_performance=0.0,  # Will be set dynamically
                required_campaign_phase=CampaignPhase.LATE,
                max_failures_allowed=2
            ),
            # Option 2: High senior developer relationship (slightly reduced from 0.75 to 0.7 for better progression)
            RelationshipUnlock(
                stakeholder_id="senior_developer",
                min_relationship=0.7
            ),
            # Option 3: High IT Director relationship and completed technical debt scenario
            CampaignUnlock(
                required_completed_scenarios=["technical_debt_reckoning"],
                min_campaign_performance=0.0,  # Will be set dynamically
                required_campaign_phase=CampaignPhase.LATE,
                required_relationships={"it_director": 0.65}  # Reduced from 0.7
            )
        ]
        self._actions = self._build_actions()
        
    def evaluate_unlock_conditions(self, game_state: 'GameState') -> bool:
        """Evaluate unlock conditions with dynamic thresholds."""
        # Get the current campaign state
        campaign_state = getattr(game_state, 'campaign_state', None)
        if not campaign_state:
            return False
            
        # Calculate dynamic threshold based on player progress
        completed_count = len(campaign_state.completed_scenarios)
        base_threshold = 0.5  # Base threshold of 50% for legacy scenarios
        
        # More gradual progression for legacy system scenarios
        progress_adjustment = min(0.2, completed_count * 0.01)  # Up to 20% adjustment (very gradual)
        dynamic_threshold = max(0.4, base_threshold - progress_adjustment)  # 40-50% range
        
        # Update all campaign unlock conditions with dynamic threshold
        for condition in self.unlock_conditions:
            if isinstance(condition, CampaignUnlock):
                condition.min_campaign_performance = dynamic_threshold
        
        # Check all unlock conditions
        return any(condition.is_met(game_state, campaign_state) 
                 for condition in self.unlock_conditions)

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_actions(self) -> List[Action]:
        return [
            # Choice 1 – Gradual Migration Advocate
            Action(
                id="gradual_migration_advocate",
                description=(
                    "We'll do this in phases - start with non-critical systems to learn "
                    "and build expertise, while [Senior Developer] documents institutional "
                    "knowledge. Gradual migration reduces risk while respecting experience."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "cfo",
                ],
                resource_cost={"time": 0.7, "budget": 0.5},
                difficulty=0.7,
                action_type="migration",
                relationship_deltas={
                    "ceo": 0.25,
                    "it_director": 0.42,
                    "sales_director": 0.13,
                    "it_team": 0.33,
                    "senior_developer": 0.30,
                    "cfo": 0.10,
                },
            ),
            # Choice 2 – Knowledge Transfer Prioritizer
            Action(
                id="knowledge_transfer_prioritizer",
                description=(
                    "Before any migration, we're doing comprehensive knowledge transfer. "
                    "[Senior Developer] will lead documentation and training sessions. "
                    "We modernize only after capturing institutional knowledge."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "cfo",
                ],
                resource_cost={"time": 0.8},
                difficulty=0.6,
                action_type="knowledge_management",
                relationship_deltas={
                    "ceo": 0.14,
                    "it_director": 0.35,
                    "sales_director": -0.03,
                    "it_team": 0.33,
                    "senior_developer": 0.55,
                    "cfo": -0.20,
                },
            ),
            # Choice 3 – Business-First Modernizer
            Action(
                id="business_first_modernizer",
                description=(
                    "The board mandate is clear, and our competitive position is suffering. "
                    "We're hiring cloud migration consultants and moving aggressively. "
                    "[Senior Developer] can adapt or we'll need to make personnel changes."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "cfo",
                ],
                resource_cost={"budget": 0.8, "risk": 0.7, "team_morale": 0.6},
                difficulty=0.9,
                action_type="transformation",
                relationship_deltas={
                    "ceo": 0.27,
                    "it_director": -0.03,
                    "sales_director": 0.38,
                    "it_team": -0.08,
                    "senior_developer": -0.65,
                    "cfo": 0.33,
                },
            ),
            # Choice 4 – Hybrid Solution Designer
            Action(
                id="hybrid_solution_designer",
                description=(
                    "We're designing a hybrid approach - keep the stable core system, "
                    "build modern APIs and cloud interfaces around it. We get modernization "
                    "benefits while preserving institutional knowledge."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "cfo",
                ],
                resource_cost={"time": 0.6, "budget": 0.6},
                difficulty=0.8,
                action_type="architecture",
                relationship_deltas={
                    "ceo": 0.30,
                    "it_director": 0.40,
                    "sales_director": 0.20,
                    "it_team": 0.27,
                    "senior_developer": 0.23,
                    "cfo": -0.06,
                },
            ),
            # Choice 5 – Innovation Opportunity Framer
            Action(
                id="innovation_opportunity_framer",
                description=(
                    "This is a chance to build something better. [Senior Developer], "
                    "help us design the next generation system incorporating everything "
                    "you've learned. Lead the architecture, don't just resist change."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "cfo",
                ],
                resource_cost={"time": 0.5, "budget": 0.4},
                difficulty=0.75,
                action_type="innovation",
                relationship_deltas={
                    "ceo": 0.23,
                    "it_director": 0.35,
                    "sales_director": 0.11,
                    "it_team": 0.22,
                    "senior_developer": 0.25,
                    "cfo": 0.05,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # BaseScenario implementation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._actions = self._build_actions()

    def get_available_actions(self) -> List[Action]:
        return self._actions

    def __repr__(self) -> str:  # noqa: D401
        return f"<LegacySystemGuardian id={self.scenario_id} choices={len(self._actions)}>"
