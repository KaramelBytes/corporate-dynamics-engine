"""Burnout Crisis – Scenario #5

Implements *The Burnout Crisis* specification from
`docs/scenario_05_burnout_crisis.md`.
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
class BurnoutCrisis(BaseScenario):
    """Scenario addressing individual welfare vs. business pressure."""
    # Mid-phase scenario
    campaign_phase = CampaignPhase.MID
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="burnout_crisis",
        title="Burnout Crisis",
        description=(
            "Your star senior developer is showing severe burnout symptoms right before "
            "a critical client deadline. Navigate the tension between business needs, "
            "team welfare, and sustainable work practices."
        ),
        scenario_type=ScenarioType.CRISIS,
        difficulty=0.75,
        themes=["burnout", "welfare", "sustainable_work"],
        stakeholders=[
            "ceo",
            "it_director",
            "sales_director",
            "it_team",
            "senior_developer",
            "hr_director",
        ],
    )

    def __init__(self) -> None:
        # First initialize the parent class
        super().__init__()
        
        # Calculate dynamic thresholds based on campaign progress
        # For the first unlock condition, we want a moderate performance requirement
        # For the second, we want to catch struggling players
        self.unlock_conditions = [
            # Option 1: Complete 4+ scenarios with reasonable performance
            CampaignUnlock(
                required_completed_scenarios=[],  # Any 4 scenarios
                min_campaign_performance=0.0,  # Will be set dynamically in evaluate_unlock_conditions
                required_campaign_phase=CampaignPhase.MID,
                max_failures_allowed=1
            ),
            # Option 2: High team stress indicators (multiple low scores in previous scenarios)
            CampaignUnlock(
                required_completed_scenarios=[],  # Any 3 scenarios
                min_campaign_performance=0.0,
                max_campaign_performance=0.4,  # Lower threshold to help struggling players
                required_campaign_phase=CampaignPhase.MID,
                max_failures_allowed=3  # Multiple failures indicate high stress
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
        progress_adjustment = min(0.2, completed_count * 0.03)  # Up to 20% adjustment
        dynamic_threshold = max(0.3, base_threshold - progress_adjustment)  # 30-40% range
        
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
            # Choice 1 – Intervention Advocate
            Action(
                id="intervention_advocate",
                description=(
                    "[Senior Developer] needs immediate support. I'm recommending they take "
                    "a week off to recover while we distribute their critical tasks. Missing "
                    "the deadline is better than losing our key developer permanently."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "hr_director",
                ],
                resource_cost={"time": 0.4, "budget": 0.2},
                difficulty=0.6,
                action_type="welfare",
                relationship_deltas={
                    "ceo": -0.07,
                    "it_director": 0.45,
                    "sales_director": 0.00,
                    "it_team": 0.47,
                    "senior_developer": 0.25,
                    "hr_director": 0.58,
                },
            ),
            # Choice 2 – Support System Builder
            Action(
                id="support_system_builder",
                description=(
                    "We're implementing immediate support - pair programming, reduced hours, "
                    "and knowledge transfer sessions. [Senior Developer] stays involved but "
                    "with backup for everything critical."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "hr_director",
                ],
                resource_cost={"time": 0.6},
                difficulty=0.5,
                action_type="support",
                relationship_deltas={
                    "ceo": 0.20,
                    "it_director": 0.38,
                    "sales_director": 0.22,
                    "it_team": 0.33,
                    "senior_developer": 0.13,
                    "hr_director": 0.33,
                },
            ),
            # Choice 3 – Deadline Negotiator
            Action(
                id="deadline_negotiator",
                description=(
                    "I'm calling the client to negotiate a 2-week extension. We'll explain that "
                    "ensuring quality delivery requires proper development practices, which "
                    "benefits everyone in the long run."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "hr_director",
                ],
                resource_cost={"time": 0.3, "client_trust": 0.4},
                difficulty=0.7,
                action_type="negotiation",
                relationship_deltas={
                    "ceo": -0.13,
                    "it_director": 0.33,
                    "sales_director": 0.05,
                    "it_team": 0.33,
                    "senior_developer": 0.14,
                    "hr_director": 0.42,
                },
            ),
            # Choice 4 – Performance Maintainer
            Action(
                id="performance_maintainer",
                description=(
                    "[Senior Developer] is a professional who can handle this pressure. "
                    "Let's provide coffee, catered meals, and whatever support they need "
                    "to power through the next three weeks."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "hr_director",
                ],
                resource_cost={"budget": 0.3, "risk": 0.6},
                difficulty=0.4,
                action_type="execution",
                relationship_deltas={
                    "ceo": 0.33,
                    "it_director": -0.25,
                    "sales_director": 0.27,
                    "it_team": -0.35,
                    "senior_developer": -0.30,
                    "hr_director": -0.45,
                },
            ),
            # Choice 5 – Team Mobilization Leader
            Action(
                id="team_mobilization_leader",
                description=(
                    "This is a team effort now. We're doing intensive knowledge transfer "
                    "immediately - [Senior Developer] documents everything while the team "
                    "takes over implementation. We protect both the person and the delivery."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                    "hr_director",
                ],
                resource_cost={"time": 0.7, "team_capacity": 0.5},
                difficulty=0.8,
                action_type="leadership",
                relationship_deltas={
                    "ceo": 0.30,
                    "it_director": 0.47,
                    "sales_director": 0.23,
                    "it_team": 0.28,
                    "senior_developer": 0.08,
                    "hr_director": 0.38,
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
        return f"<BurnoutCrisis id={self.scenario_id} choices={len(self._actions)}>"
