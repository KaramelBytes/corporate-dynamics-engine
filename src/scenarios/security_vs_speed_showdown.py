"""Security vs. Speed Showdown – Scenario #3

Implements *The Security vs. Speed Showdown* specification from
`docs/scenario_03_security_vs_speed.md`.
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
class SecurityVsSpeedShowdown(BaseScenario):
    """Scenario highlighting tension between business deadlines and security rigor."""
    # Mid-phase scenario
    campaign_phase = CampaignPhase.MID
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="security_vs_speed_showdown",
        title="Security vs. Speed Showdown",
        description=(
            "A critical client demands immediate feature deployment while the "
            "Security Director insists on full compliance testing. Balance "
            "revenue pressure with risk management."
        ),
        scenario_type=ScenarioType.PROJECT,
        difficulty=0.65,
        themes=["security", "client_pressure", "risk_management"],
        stakeholders=[
            "ceo",
            "it_director",
            "sales_director",
            "it_team",
            "security_director",
        ],
    )

    def __init__(self) -> None:
        # First initialize the parent class
        super().__init__()
        
        # Initialize unlock conditions with dynamic thresholds
        self.unlock_conditions = [
            # Option 1: Complete 2+ scenarios with reasonable performance
            CampaignUnlock(
                required_completed_scenarios=[],  # Any 2 scenarios
                min_campaign_performance=0.0,  # Will be set dynamically in evaluate_unlock_conditions
                required_campaign_phase=CampaignPhase.MID,
                max_failures_allowed=1
            ),
            # Option 2: High IT Director relationship (slightly reduced from 0.7 to 0.65 for better progression)
            RelationshipUnlock(
                stakeholder_id="it_director",
                min_relationship=0.65
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
        base_threshold = 0.4  # Base threshold of 40%
        progress_adjustment = min(0.2, completed_count * 0.02)  # Up to 20% adjustment (slower progression)
        dynamic_threshold = max(0.35, base_threshold - progress_adjustment)  # 35-40% range
        
        # Update the first unlock condition with dynamic threshold
        if self.unlock_conditions and isinstance(self.unlock_conditions[0], CampaignUnlock):
            self.unlock_conditions[0].min_campaign_performance = dynamic_threshold
        
        # Check all unlock conditions
        return any(condition.is_met(game_state, campaign_state) 
                 for condition in self.unlock_conditions)

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_actions(self) -> List[Action]:
        return [
            # Choice 1 – Risk Assessment Mediator
            Action(
                id="risk_assessment_mediator",
                description=(
                    "Facilitate a rapid risk assessment to find a minimal viable "
                    "deployment satisfying both client and security."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "security_director",
                ],
                resource_cost={"time": 0.8},
                difficulty=0.45,
                action_type="analysis",
                relationship_deltas={
                    "ceo": 0.20,
                    "it_director": 0.33,
                    "sales_director": 0.07,
                    "it_team": 0.27,
                    "security_director": 0.15,
                },
            ),
            # Choice 2 – Security-First Guardian
            Action(
                id="security_first_guardian",
                description=(
                    "Refuse to compromise security protocols, insisting on full "
                    "testing regardless of client deadline."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "security_director",
                ],
                resource_cost={"time": 0.2},
                difficulty=0.5,
                action_type="policy_enforcement",
                relationship_deltas={
                    "ceo": 0.07,
                    "it_director": 0.42,
                    "sales_director": -0.30,
                    "it_team": 0.45,
                    "security_director": 0.55,
                },
            ),
            # Choice 3 – Business-Priority Pragmatist
            Action(
                id="business_priority_pragmatist",
                description=(
                    "Prioritise the client deadline with minimal security review, "
                    "accepting higher risk but protecting revenue."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "security_director",
                ],
                resource_cost={"time": 0.1, "risk": 0.5},
                difficulty=0.55,
                action_type="execution",
                relationship_deltas={
                    "ceo": 0.17,
                    "it_director": -0.25,
                    "sales_director": 0.45,
                    "it_team": -0.35,
                    "security_director": -0.45,
                },
            ),
            # Choice 4 – Compromise Engineer
            Action(
                id="compromise_engineer",
                description=(
                    "Deploy a limited version with enhanced monitoring while full "
                    "security review proceeds in parallel."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "security_director",
                ],
                resource_cost={"time": 0.6},
                difficulty=0.5,
                action_type="negotiation",
                relationship_deltas={
                    "ceo": 0.33,
                    "it_director": 0.38,
                    "sales_director": 0.20,
                    "it_team": 0.33,
                    "security_director": 0.10,
                },
            ),
            # Choice 5 – Innovation Accelerator
            Action(
                id="innovation_accelerator",
                description=(
                    "Adopt emergency deployment protocols with real-time scanning "
                    "and staged rollout as a pilot for faster secure delivery."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "security_director",
                ],
                resource_cost={"time": 0.4},
                difficulty=0.6,
                action_type="innovation",
                relationship_deltas={
                    "ceo": 0.25,
                    "it_director": 0.13,
                    "sales_director": 0.38,
                    "it_team": 0.07,
                    "security_director": -0.06,
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
        return f"<SecurityVsSpeedShowdown id={self.scenario_id} choices={len(self._actions)}>"
