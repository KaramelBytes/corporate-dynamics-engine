"""Technical Debt Reckoning – Scenario #4

Implements *The Technical Debt Reckoning* specification from
`docs/scenario_04_technical_debt_reckoning.md`.
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
class TechnicalDebtReckoning(BaseScenario):
    """Scenario highlighting technical debt consequences and responsibility assignment."""
    # Mid-phase scenario
    campaign_phase = CampaignPhase.MID
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="technical_debt_reckoning",
        title="Technical Debt Reckoning",
        description=(
            "A production outage reveals years of neglected technical debt. Navigate "
            "blame, responsibility, and resource allocation while managing stakeholder "
            "relationships and client confidence."
        ),
        scenario_type=ScenarioType.CRISIS,
        difficulty=0.7,
        themes=["technical_debt", "accountability", "crisis_management"],
        stakeholders=[
            "ceo",
            "it_director",
            "sales_director",
            "it_team",
            "senior_developer",
        ],
    )

    unlock_conditions: List[UnlockCondition] = [
        # Option 1: Complete 3+ scenarios with average performance
        CampaignUnlock(
            required_completed_scenarios=[],  # Any 3 scenarios
            min_campaign_performance=0.4,
            required_campaign_phase=CampaignPhase.MID,
            max_failures_allowed=2
        ),
        # Option 2: Previous poor performance (low score in other scenarios)
        CampaignUnlock(
            required_completed_scenarios=[],  # Any 2 scenarios
            min_campaign_performance=0.0,
            max_campaign_performance=0.4,  # Only available if performance is poor
            required_campaign_phase=CampaignPhase.MID
        )
    ]

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_actions(self) -> List[Action]:
        return [
            # Choice 1 – Systemic Analysis Leader
            Action(
                id="systemic_analysis_leader",
                description=(
                    "This isn't about blame - it's about systemic failure. We need to audit all our "
                    "technical debt, prioritize fixes by risk, and establish proper engineering time allocation."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                ],
                resource_cost={"time": 0.6},
                difficulty=0.5,
                action_type="analysis",
                relationship_deltas={
                    "ceo": 0.17,
                    "it_director": 0.38,
                    "sales_director": 0.18,
                    "it_team": 0.33,
                    "senior_developer": 0.27,
                },
            ),
            # Choice 2 – Historical Context Clarifier
            Action(
                id="historical_context_clarifier",
                description=(
                    "Let's get the facts straight. [Senior Developer], remind everyone what you "
                    "documented about these fixes in 2020. We need to understand how business "
                    "decisions led to permanent 'temporary' solutions."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                ],
                resource_cost={"time": 0.3, "political_capital": 0.4},
                difficulty=0.6,
                action_type="accountability",
                relationship_deltas={
                    "ceo": -0.08,
                    "it_director": 0.33,
                    "sales_director": 0.08,
                    "it_team": 0.42,
                    "senior_developer": 0.55,
                },
            ),
            # Choice 3 – Resource Allocation Advocate
            Action(
                id="resource_allocation_advocate",
                description=(
                    "The engineering team flagged these risks repeatedly. The real issue is business "
                    "priorities preventing technical debt paydown. We need dedicated infrastructure "
                    "time or this will happen again."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                ],
                resource_cost={"time": 0.4, "political_capital": 0.5},
                difficulty=0.7,
                action_type="advocacy",
                relationship_deltas={
                    "ceo": -0.10,
                    "it_director": 0.47,
                    "sales_director": 0.13,
                    "it_team": 0.53,
                    "senior_developer": 0.45,
                },
            ),
            # Choice 4 – Crisis-to-Investment Opportunity
            Action(
                id="crisis_to_investment_opportunity",
                description=(
                    "This gives us the business case we've needed. I'm proposing a 3-month "
                    "technical debt elimination project with dedicated resources. The cost of "
                    "prevention is less than another outage."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                ],
                resource_cost={"time": 0.5, "budget": 0.4},
                difficulty=0.6,
                action_type="strategic_planning",
                relationship_deltas={
                    "ceo": 0.23,
                    "it_director": 0.42,
                    "sales_director": 0.27,
                    "it_team": 0.38,
                    "senior_developer": 0.33,
                },
            ),
            # Choice 5 – Client-First Stabilizer
            Action(
                id="client_first_stabilizer",
                description=(
                    "Right now, client confidence is everything. We implement minimal fixes to "
                    "stabilize, handle contract renegotiations, then schedule proper technical "
                    "debt resolution when we're not in crisis mode."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "senior_developer",
                ],
                resource_cost={"time": 0.3, "risk": 0.4},
                difficulty=0.5,
                action_type="crisis_management",
                relationship_deltas={
                    "ceo": 0.27,
                    "it_director": 0.13,
                    "sales_director": 0.38,
                    "it_team": -0.03,
                    "senior_developer": -0.08,
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
        return f"<TechnicalDebtReckoning id={self.scenario_id} choices={len(self._actions)}>"
