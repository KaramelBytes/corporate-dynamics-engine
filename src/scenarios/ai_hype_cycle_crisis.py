"""AI Hype Cycle Crisis – Scenario #1

Implements the *AI Hype Cycle* specification located in
`docs/scenario_01_ai_hype_cycle.md`.

All stakeholder relationship deltas are provided *verbatim* from the markdown
specification.  The engine will bypass heuristics and apply these values
exactly thanks to the `Action.relationship_deltas` override implemented in the
`StakeholderRelationshipMatrix`.
"""
from __future__ import annotations

from typing import Any, Dict, List

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

# ---------------------------------------------------------------------------
# Metadata & registration
# ---------------------------------------------------------------------------


@register_scenario
class AIHypeCycleCrisis(BaseScenario):
    """Scenario representing executive AI-mania confronting implementation reality."""

    # Tutorial scenario - always available
    is_tutorial = True
    campaign_phase = CampaignPhase.TUTORIAL
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="ai_hype_cycle_crisis",
        title="AI Hype Cycle Crisis",
        description=(
            "The CEO demands an all-encompassing AI strategy after reading a viral "
            "TechCrunch article. Balance innovation hype with implementation "
            "reality while managing stakeholder expectations."
        ),
        scenario_type=ScenarioType.INNOVATION,
        difficulty=0.4,
        themes=["tutorial", "ai", "strategy", "executive_pressure"],
        stakeholders=[
            "ceo",
            "it_director",
            "facilities_manager",
            "admin_team",
            "sales_director",
        ],
    )

    # Unlocked by default – can be refined later with conditions.
    unlock_conditions: List[UnlockCondition] = []

    # ------------------------------------------------------------------
    # Choice / Action definitions
    # ------------------------------------------------------------------
    def _build_actions(self) -> List[Action]:  # noqa: D401 – internal helper
        """Return the five player choices mapped to `Action` objects."""
        return [
            # Choice 1 – Collaborative Investigator
            Action(
                id="collaborative_investigator",
                description=(
                    "Partner with the IT Director to assess current systems and "
                    "identify realistic AI value-adds before presenting jointly."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                ],
                resource_cost={"time": 0.6},
                difficulty=0.3,
                action_type="analysis",
                relationship_deltas={
                    "ceo": 0.15,
                    "it_director": 0.35,
                    "facilities_manager": 0.15,
                    "admin_team": 0.15,
                    "sales_director": 0.05,
                },
            ),
            # Choice 2 – Hype Cycle Historian
            Action(
                id="hype_cycle_historian",
                description=(
                    "Invoke lessons from past hype cycles, insist on a structured "
                    "problem-first assessment before any AI commitments."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                ],
                resource_cost={"time": 0.8},
                difficulty=0.4,
                action_type="analysis",
                relationship_deltas={
                    "ceo": -0.10,
                    "it_director": 0.45,
                    "facilities_manager": 0.25,
                    "admin_team": 0.25,
                    "sales_director": -0.15,
                },
            ),
            # Choice 3 – Strategic Consultant
            Action(
                id="strategic_consultant",
                description=(
                    "Bring in external AI consultants with IT leading the effort "
                    "to identify opportunities."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                ],
                resource_cost={"budget": 10000, "time": 0.4},
                difficulty=0.5,
                action_type="consultancy",
                relationship_deltas={
                    "ceo": 0.18,
                    "it_director": 0.17,
                    "facilities_manager": -0.15,
                    "admin_team": -0.06,
                    "sales_director": 0.27,
                },
            ),
            # Choice 4 – Problem-First Pragmatist
            Action(
                id="problem_first_pragmatist",
                description=(
                    "Identify concrete business pain-points first, then assess if "
                    "AI, automation or process improvement is the right tool."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                ],
                resource_cost={"time": 0.7},
                difficulty=0.35,
                action_type="analysis",
                relationship_deltas={
                    "ceo": 0.14,
                    "it_director": 0.38,
                    "facilities_manager": 0.23,
                    "admin_team": 0.22,
                    "sales_director": 0.13,
                },
            ),
            # Choice 5 – Aggressive Accelerator
            Action(
                id="aggressive_accelerator",
                description=(
                    "Form a rapid AI task-force and promise a comprehensive "
                    "implementation roadmap to the board."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "facilities_manager",
                    "admin_team",
                    "sales_director",
                ],
                resource_cost={"time": 0.3, "risk": 0.4},
                difficulty=0.6,
                action_type="execution",
                relationship_deltas={
                    "ceo": 0.35,
                    "it_director": -0.30,
                    "facilities_manager": -0.23,
                    "admin_team": -0.03,
                    "sales_director": 0.33,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Required BaseScenario implementation
    # ------------------------------------------------------------------

    def __init__(self) -> None:  # noqa: D401
        # Pre-compute actions once; they are immutable.
        self._actions = self._build_actions()

    def get_available_actions(self) -> List[Action]:  # noqa: D401
        """Return the five predefined scenario choices."""
        return self._actions

    # Context helper remains inherited from BaseScenario.

    # ------------------------------------------------------------------
    # Convenience for debugging / manual tests
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return f"<AIHypeCycleCrisis id={self.scenario_id} choices={len(self._actions)}>"
