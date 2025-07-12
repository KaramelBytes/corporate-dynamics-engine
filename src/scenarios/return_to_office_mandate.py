"""Return-to-Office Mandate – Scenario #2

Implements the *Return-to-Office* specification from
`docs/scenario_02_return_to_office.md`.

All relationship deltas are specified verbatim and therefore leverage the
`relationship_deltas` override.
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
class ReturnToOfficeMandate(BaseScenario):
    """Scenario covering executive-driven full return to office decision."""
    # Early phase scenario
    campaign_phase = CampaignPhase.EARLY
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="return_to_office_mandate",
        title="The Return-to-Office Mandate",
        description=(
            "Navigate the executive push for a full return to on-site work while "
            "balancing data-driven productivity insights and workforce morale."
        ),
        scenario_type=ScenarioType.POLITICS,
        difficulty=0.6,
        themes=["workplace_policy", "culture", "talent_retention"],
        stakeholders=[
            "ceo",
            "it_director",
            "facilities_manager",
            "admin_team",
            "sales_director",
            "it_team",
        ],
    )

    unlock_conditions: List[UnlockCondition] = [
        # Available after completing any tutorial scenario
        CampaignUnlock(
            required_completed_scenarios=["ai_hype_cycle_crisis", "coffee_machine_crisis"],
            min_campaign_performance=0.4,
            required_campaign_phase=CampaignPhase.EARLY
        )
    ]

    # ------------------------------------------------------------------
    # Internal builder helpers
    # ------------------------------------------------------------------

    def _build_actions(self) -> List[Action]:
        return [
            # Choice 1: The Data Advocate
            Action(
                id="data_advocate",
                description=(
                    "Present productivity and retention data opposing a forced RTO, "
                    "arguing that flexibility is a competitive advantage."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                    "it_team",
                ],
                resource_cost={"time": 0.7},
                difficulty=0.45,
                action_type="analysis",
                relationship_deltas={
                    "ceo": 0.03,
                    "it_director": 0.33,
                    "facilities_manager": -0.06,
                    "admin_team": 0.33,
                    "sales_director": 0.15,
                    "it_team": 0.45,
                },
            ),
            # Choice 2: The Compromise Architect
            Action(
                id="compromise_architect",
                description=(
                    "Propose structured collaboration days combining office presence "
                    "and remote focus time to address culture and productivity."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                    "it_team",
                ],
                resource_cost={"time": 0.8},
                difficulty=0.4,
                action_type="negotiation",
                relationship_deltas={
                    "ceo": 0.07,
                    "it_director": 0.27,
                    "facilities_manager": 0.13,
                    "admin_team": 0.18,
                    "sales_director": 0.22,
                    "it_team": 0.14,
                },
            ),
            # Choice 3: The Loyal Lieutenant
            Action(
                id="loyal_lieutenant",
                description=(
                    "Fully support the CEO's mandate, focus on smooth transition "
                    "planning and retention contingency strategies."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                    "it_team",
                ],
                resource_cost={"time": 0.5},
                difficulty=0.5,
                action_type="execution",
                relationship_deltas={
                    "ceo": 0.33,
                    "it_director": -0.15,
                    "facilities_manager": 0.27,
                    "admin_team": -0.35,
                    "sales_director": 0.18,
                    "it_team": -0.43,
                },
            ),
            # Choice 4: The Talent Retention Warrior
            Action(
                id="talent_retention_warrior",
                description=(
                    "Advocate exceptions for critical talent to avoid resignations "
                    "and protect delivery capabilities."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                    "it_team",
                ],
                resource_cost={"time": 0.6},
                difficulty=0.55,
                action_type="negotiation",
                relationship_deltas={
                    "ceo": -0.03,
                    "it_director": 0.38,
                    "facilities_manager": -0.20,
                    "admin_team": 0.27,
                    "sales_director": 0.18,
                    "it_team": 0.55,
                },
            ),
            # Choice 5: The Productivity Protector
            Action(
                id="productivity_protector",
                description=(
                    "Highlight data showing deep-work productivity losses in open "
                    "offices, arguing that innovation requires focus not presence."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                    "it_team",
                ],
                resource_cost={"time": 0.6},
                difficulty=0.5,
                action_type="analysis",
                relationship_deltas={
                    "ceo": -0.10,
                    "it_director": 0.40,
                    "facilities_manager": -0.25,
                    "admin_team": 0.38,
                    "sales_director": 0.11,
                    "it_team": 0.65,
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
        return f"<ReturnToOfficeMandate id={self.scenario_id} choices={len(self._actions)}>"
