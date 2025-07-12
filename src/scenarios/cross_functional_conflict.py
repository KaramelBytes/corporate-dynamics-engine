"""Cross-Functional Conflict – Scenario #8

Implements *The Cross-Functional Conflict* specification from
`docs/scenario_08_cross_functional_conflict.md`.
"""
from __future__ import annotations

from typing import List

from src.state_management.data_models import Action

from .base_scenario import (
    BaseScenario,
    ScenarioMetadata,
    ScenarioType,
    UnlockCondition,
)
from .scenario_factory import register_scenario


@register_scenario
class CrossFunctionalConflict(BaseScenario):
    """Scenario addressing resource allocation and departmental priorities."""

    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="cross_functional_conflict_001",
        title="Cross-Functional Conflict",
        description=(
            "Departments clash over limited development resources. Engineering "
            "demands technical debt reduction, Sales needs client customizations, "
            "and Marketing insists on modern features. Balance competing priorities."
        ),
        scenario_type=ScenarioType.RESOURCE_ALLOCATION,
        difficulty=0.75,
        themes=["resource_allocation", "departmental_alignment", "strategic_planning"],
        stakeholders=[
            "ceo",
            "it_director",
            "sales_director",
            "it_team",
            "marketing_team",
            "cfo",
            "admin_team",
        ],
    )

    unlock_conditions: List[UnlockCondition] = []

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_actions(self) -> List[Action]:
        return [
            # Choice 1 – Technical Debt Prioritizer
            Action(
                id="technical_debt_prioritizer",
                description=(
                    "System stability comes first. We're dedicating the next 6 months "
                    "to technical debt reduction. New features and customizations wait "
                    "until we have a solid foundation."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "marketing_team",
                    "cfo",
                    "admin_team",
                ],
                resource_cost={"time": 0.6, "opportunity_cost": 0.8},
                difficulty=0.7,
                action_type="technical_investment",
                relationship_deltas={
                    "ceo": 0.17,
                    "it_director": 0.55,
                    "sales_director": -0.18,
                    "it_team": 0.55,
                    "marketing_team": -0.15,
                    "cfo": 0.17,
                    "admin_team": 0.38,
                },
            ),
            # Choice 2 – Revenue Maximizer
            Action(
                id="revenue_maximizer",
                description=(
                    "$800K in immediate revenue justifies everything. We're building the "
                    "client customizations first, then addressing other priorities. "
                    "Cash flow enables everything else."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "marketing_team",
                    "cfo",
                    "admin_team",
                ],
                resource_cost={"time": 0.8, "technical_debt": 0.5, "team_capacity": 0.9},
                difficulty=0.8,
                action_type="revenue_focus",
                relationship_deltas={
                    "ceo": 0.27,
                    "it_director": -0.26,
                    "sales_director": 0.55,
                    "it_team": -0.37,
                    "marketing_team": -0.07,
                    "cfo": 0.38,
                    "admin_team": 0.03,
                },
            ),
            # Choice 3 – Competitive Modernizer
            Action(
                id="competitive_modernizer",
                description=(
                    "Market position is critical for long-term success. We're prioritizing "
                    "AI features and modern interfaces. Technical debt and customizations "
                    "can be addressed after we're competitive."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "marketing_team",
                    "cfo",
                    "admin_team",
                ],
                resource_cost={"time": 0.7, "budget": 0.6, "technical_debt": 0.3},
                difficulty=0.75,
                action_type="innovation",
                relationship_deltas={
                    "ceo": 0.27,
                    "it_director": -0.20,
                    "sales_director": -0.13,
                    "it_team": -0.16,
                    "marketing_team": 0.58,
                    "cfo": 0.12,
                    "admin_team": -0.08,
                },
            ),
            # Choice 4 – Hybrid Compromiser
            Action(
                id="hybrid_compromiser",
                description=(
                    "We're splitting resources: 50% technical debt, 30% sales "
                    "customizations, 20% marketing features. Everyone gets something, "
                    "though not everything they want."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "marketing_team",
                    "cfo",
                    "admin_team",
                ],
                resource_cost={"time": 0.9, "budget": 0.5, "team_capacity": 0.8},
                difficulty=0.85,
                action_type="balance",
                relationship_deltas={
                    "ceo": 0.38,
                    "it_director": 0.20,
                    "sales_director": 0.13,
                    "it_team": 0.15,
                    "marketing_team": 0.07,
                    "cfo": 0.25,
                    "admin_team": 0.27,
                },
            ),
            # Choice 5 – Sequential Strategist
            Action(
                id="sequential_strategist",
                description=(
                    "We're doing this strategically: technical debt first (3 months), "
                    "then sales customizations (3 months). Marketing features follow "
                    "once we have stable revenue and systems."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "marketing_team",
                    "cfo",
                    "admin_team",
                ],
                resource_cost={"time": 0.7, "budget": 0.4, "team_capacity": 0.7},
                difficulty=0.8,
                action_type="strategic_planning",
                relationship_deltas={
                    "ceo": 0.33,
                    "it_director": 0.42,
                    "sales_director": 0.10,
                    "it_team": 0.45,
                    "marketing_team": 0.00,
                    "cfo": 0.27,
                    "admin_team": 0.33,
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
        return f"<CrossFunctionalConflict id={self.scenario_id} choices={len(self._actions)}>"
