"""
Coffee Machine Crisis - POC scenario demonstrating corporate dynamics.
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


@register_scenario
class CoffeeMachineCrisis(BaseScenario):
    """
    The Great Coffee Machine Crisis of Q3 - POC scenario
    Demonstrates stakeholder dynamics, vendor relations, and crisis management.
    """
    # Tutorial scenario - always available
    is_tutorial = True
    campaign_phase = CampaignPhase.TUTORIAL

    # ------------------------------------------------------------------
    # Static metadata – evaluated once at import time.
    # ------------------------------------------------------------------
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="coffee_machine_crisis",
        title="The Great Coffee Machine Crisis of Q3",
        description=(
            "The fancy new smart coffee machine is broken. The vendor says it's a "
            "'firmware issue.' You're in back-to-back meetings. The executive team "
            "is arriving for the board presentation in 2 hours. Office productivity "
            "has dropped 40%. How do you navigate this crisis?"
        ),
        scenario_type=ScenarioType.CRISIS,
        difficulty=0.3,
        themes=["tutorial", "productivity", "vendor_management"],
        stakeholders=["facilities_manager", "ceo", "admin_team", "it_director"],
        estimated_duration_minutes=10
    )

    # No unlock conditions for tutorial scenarios
    unlock_conditions: List[UnlockCondition] = []

    # ------------------------------------------------------------------
    # Concrete implementation of abstract API.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_available_actions(self) -> List[Action]:
        """Return static list of example actions for the Coffee Machine Crisis."""
        return [
            Action(
                id="vendor_escalation",
                description="Escalate the issue to vendor support and demand urgent fix.",
                directly_affects=["facilities_manager"],
                resource_cost={"time": 0.5},
                difficulty=0.3,
                action_type="communication",
                relationship_deltas={
                    "facilities_manager": 0.2,  # Appreciates escalation effort
                    "ceo": 0.1,                 # Likes proactive approach
                    "admin_team": 0.0,          # Neutral - doesn't affect their daily work much
                    "it_director": -0.1         # Slightly annoyed it's not their problem
                }
            ),
            Action(
                id="starbucks_budget",
                description="Propose an emergency Starbucks budget for the office.",
                directly_affects=["admin_team", "ceo"],
                resource_cost={"budget": 1000},
                difficulty=0.2,
                action_type="budget_request",
                relationship_deltas={
                    "facilities_manager": -0.1,  # Feels like their problem is being bypassed
                    "ceo": -0.2,                 # Concerned about setting expensive precedent
                    "admin_team": 0.3,           # Love the immediate coffee solution
                    "it_director": 0.0           # Indifferent to coffee source
                }
            ),
            Action(
                id="diy_repair",
                description="Attempt a DIY repair of the coffee machine.",
                directly_affects=["it_director"],
                resource_cost={"time": 1.0, "risk": 0.4},
                difficulty=0.6,
                action_type="technical",
                relationship_deltas={
                    "facilities_manager": -0.2,  # Worried about warranty and proper procedures
                    "ceo": 0.1,                  # Appreciates resourcefulness
                    "admin_team": 0.2,           # Hopeful for quick fix
                    "it_director": 0.3           # Respects technical problem-solving approach
                }
            ),
            Action(
                id="emergency_meeting",
                description="Call an emergency team meeting to address the crisis.",
                directly_affects=["ceo", "admin_team", "it_director"],
                resource_cost={"time": 0.3},
                difficulty=0.1,
                action_type="meeting",
                relationship_deltas={
                    "facilities_manager": 0.1,   # Appreciates being included in solution
                    "ceo": -0.1,                 # Slightly annoyed by meeting overhead for coffee issue
                    "admin_team": 0.1,           # Feel heard and included
                    "it_director": -0.1          # Views as inefficient for non-IT issue
                }
            ),
        ]

    def get_scenario_context(self) -> Dict[str, Any]:
        """Return context useful for AI generation or logging."""
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "stakeholder_descriptions": {
                "ceo": "Concerned about employee morale but focused on bigger issues",
                "office_manager": "Directly responsible for office amenities",
                "it_director": "Believes coffee machine is not an IT issue",
                "hr_manager": "Worried about impact on employee satisfaction",
                "finance_director": "Concerned about unnecessary expenses"  
            },
            "stakeholder_profiles": {
                "ceo": {
                    "name": "Alex Johnson",
                    "role": "CEO",
                    "background": "Concerned about employee morale but focused on bigger issues",
                    "priorities": ["company growth", "investor relations", "strategic planning"],
                    "trust": 0.70,
                    "respect": 0.80,
                    "influence": 0.90
                },
                "office_manager": {
                    "name": "Morgan Lee",
                    "role": "Office Manager",
                    "background": "Directly responsible for office amenities",
                    "priorities": ["employee comfort", "office functionality", "vendor management"],
                    "trust": 0.50,
                    "respect": 0.60,
                    "influence": 0.50
                },
                "it_director": {
                    "name": "Taylor Smith",
                    "role": "IT Director",
                    "background": "Believes coffee machine is not an IT issue",
                    "priorities": ["system uptime", "cybersecurity", "technical debt reduction"],
                    "trust": 0.60,
                    "respect": 0.70,
                    "influence": 0.60
                },
                "hr_manager": {
                    "name": "Jordan Rivera",
                    "role": "HR Manager",
                    "background": "Worried about impact on employee satisfaction",
                    "priorities": ["employee satisfaction", "retention", "workplace culture"],
                    "trust": 0.65,
                    "respect": 0.75,
                    "influence": 0.55
                },
                "finance_director": {
                    "name": "Casey Wong",
                    "role": "Finance Director",
                    "background": "Concerned about unnecessary expenses",
                    "priorities": ["cost control", "budget adherence", "ROI analysis"],
                    "trust": 0.65,
                    "respect": 0.70,
                    "influence": 0.75
                }
            },
            "previous_events": [
                "The coffee machine has been malfunctioning for weeks",
                "Several employees have complained about coffee quality",
                "A recent budget review has put all non-essential purchases on hold"
            ]
        }

    # FUTURE: Add multi-scenario progression for extended gameplay
